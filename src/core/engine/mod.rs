// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

pub(crate) mod concepts;
mod document_api;
pub(crate) mod knowledge;
pub(crate) mod multi_hop;
mod mutation;
pub(crate) mod query;
pub(crate) mod router;
pub(crate) mod shard;
pub(crate) mod structural;

use anyhow::{Context, Result};
use parking_lot::RwLock;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::admission::AdmissionControl;
use super::agency::goals::GoalStore;
use super::agency::planner::{Plan, Planner};
use super::agency::questions::{Question, QuestionGenerator};
use super::agency::self_modify::{ProposalKind, SelfModifier};
use super::atom_memory::AtomMemory;
use super::audit::{AuditLog, AuditOp};
use super::cognition::governor::{GovernanceReport, GovernorConfig, MemoryGovernor};
use super::cognition::r#loop::{CognitionConfig as CognitionLoopConfig, CognitionLoop, Insight};
use super::composite_memory::CompositeMemory;
use super::config::HmsConfig;
use super::decompose::Decomposer;
use super::diffusion::DiffusionFactorizer;
use super::encoding::encode_text_internal;
use super::entangled::EntangledHVec;
use super::graph::RelationStore;
use super::ivf::IVFIndex;
use super::role::RoleRegistry;
use super::rules::RuleStore;
use super::storage::PersistentArena;
use super::store_lock::StoreLock;
use super::text::TextProcessor;
use super::triple_store::TripleStore;
use super::types::{GraphPath, Relation, RelationType, TextMetrics};

use mutation::Mutation;
use shard::{ShardManager, ShardSet};

type SignFn<'a> = Box<dyn Fn(&[u8]) -> super::audit::SignatureBytes + 'a>;

/// Persistent operations take mutation_gate before ShardSet and component locks.
pub struct HmsCore {
    config: HmsConfig,
    cached_zt_key: Option<EntangledHVec>,
    mutation_gate: RwLock<()>,
    revision: std::sync::atomic::AtomicU64,
    pub(crate) arena: Arc<PersistentArena>,
    pub(crate) dimensions: usize,
    pub(crate) storage_path: PathBuf,
    shards: RwLock<ShardSet>,
    graph: RelationStore,
    documents: RwLock<std::collections::BTreeMap<String, super::documents::StoredDocument>>,
    derived: RwLock<std::collections::BTreeMap<String, Vec<String>>>,
    atom_memory: Option<Arc<AtomMemory>>,
    composite_memory: Option<Arc<CompositeMemory>>,
    triple_store: Option<Arc<TripleStore>>,
    role_registry: Option<RoleRegistry>,
    rule_store: Option<RuleStore>,
    decomposer: Option<Decomposer>,
    admission: Option<AdmissionControl>,
    cognition_loop: parking_lot::Mutex<Option<CognitionLoop>>,
    goal_store: Option<GoalStore>,
    self_modifier: Option<SelfModifier>,
    audit: Option<AuditLog>,
    #[cfg(feature = "security")]
    signing: Option<super::security::SigningManager>,
    #[cfg(feature = "security")]
    #[allow(dead_code)]
    encryption: Option<super::security::EncryptionManager>,
    #[cfg(feature = "security")]
    pub identity_registry: parking_lot::RwLock<super::security::identity::IdentityRegistry>,
    #[cfg(feature = "provenance")]
    provenance: Option<super::provenance::ProvenanceManager>,
    /// Experimental opt-in plastic relation store (lazily created on first use).
    #[cfg(feature = "experimental")]
    connection_graph: parking_lot::Mutex<Option<super::connection_graph::ConnectionGraph>>,
    /// Experimental opt-in phasor relational memory (lazily created on first use).
    #[cfg(feature = "experimental")]
    phase_graph: parking_lot::Mutex<Option<super::phase_graph::PhaseGraph>>,
    // Declared last so it is dropped after every mmap, index, and store
    // component. No subsequent instance can enter while teardown is flushing.
    _store_lock: StoreLock,
}

impl HmsCore {
    /// Create a new HMS instance. If `storage_path` is None, uses the current directory.
    pub fn new(
        dimensions: u32,
        storage_path: Option<String>,
        config: Option<HmsConfig>,
    ) -> Result<Self> {
        const MAX_DIMENSIONS: u32 = 1_000_000;
        if dimensions == 0 || dimensions > MAX_DIMENSIONS {
            return Err(anyhow::anyhow!(
                "dimensions must be between 1 and {} (got {})",
                MAX_DIMENSIONS,
                dimensions
            ));
        }
        let dim = dimensions as usize;
        let config = config.unwrap_or_default();
        config.validate()?;
        anyhow::ensure!(
            ((dim / 256).max(1) as f64 / config.privacy.epsilon).is_finite(),
            "privacy epsilon is too small for the configured dimensions"
        );
        #[cfg(feature = "security")]
        let mut config = config;

        let base_path = storage_path
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(".").to_path_buf());
        if !base_path.exists() {
            std::fs::create_dir_all(&base_path)?;
        }

        let store_lock = StoreLock::acquire(&base_path)?;
        super::schema::validate_store(&base_path, dim, &config)?;

        let arena = Arc::new(PersistentArena::new(base_path.join("vectors_data.bin"))?);

        let audit = if config.security.audit_enabled {
            Some(AuditLog::new(&base_path)?)
        } else {
            None
        };

        #[cfg(feature = "security")]
        let signing = if config.security.signing_enabled {
            let key_path = config
                .security
                .key_path
                .as_ref()
                .map(PathBuf::from)
                .unwrap_or_else(|| base_path.join("hms_signing.key"));
            Some(super::security::SigningManager::new(&key_path)?)
        } else {
            None
        };

        #[cfg(feature = "security")]
        let encryption = if config.security.encryption_enabled {
            use zeroize::Zeroize;
            let mut passphrase = config
                .security
                .encryption_passphrase
                .take()
                .or_else(|| {
                    config
                        .security
                        .encryption_passphrase_env
                        .as_deref()
                        .and_then(|name| std::env::var(name).ok())
                })
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "encryption requires encryption_passphrase or encryption_passphrase_env"
                    )
                })?;
            let manager = super::security::EncryptionManager::new(&passphrase, &base_path);
            passphrase.zeroize();
            Some(manager?)
        } else {
            None
        };

        #[cfg(feature = "provenance")]
        let provenance = if config.provenance.enabled {
            let key_path = config
                .provenance
                .key_path
                .as_ref()
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| base_path.join("hms_provenance.key"));
            let mgr = super::provenance::ProvenanceManager::new(&key_path, Some(&base_path))?;
            #[cfg(feature = "provenance-scitt")]
            let mgr = if let Some(ref endpoint) = config.provenance.scitt_endpoint {
                mgr.with_scitt_endpoint(endpoint.clone())
            } else {
                mgr
            };
            Some(mgr)
        } else {
            None
        };

        let shard_set = if config.shard.enabled && config.shard.shard_count > 1 {
            ShardSet::Multi(ShardManager::new(config.shard.shard_count, dim))
        } else if let Some(n) = Self::load_shard_meta(&base_path).filter(|&n| n > 1) {
            // A store that auto-sharded in a previous run persists its shard
            // count so the topology (and per-shard indices) reload correctly.
            ShardSet::Multi(ShardManager::new(n, dim))
        } else {
            ShardSet::Single(Box::new(shard::Shard::new(dim)))
        };

        let (atom_mem, comp_mem, tri_store, role_reg, rule_st, decomp, adm) =
            if config.meaning.enabled {
                let mc = &config.meaning;
                (
                    Some(Arc::new(AtomMemory::new(dim, mc.idf_clip_factor))),
                    Some(Arc::new(CompositeMemory::new(dim, mc.idf_clip_factor))),
                    Some(Arc::new(TripleStore::new())),
                    Some(RoleRegistry::new(dim)),
                    Some(RuleStore::new()),
                    Some(Decomposer::new()),
                    Some(AdmissionControl::new(mc.algebraic_max_fanout)),
                )
            } else {
                (None, None, None, None, None, None, None)
            };


        let cached_zt_key = config.privacy.zero_trust_key.as_ref().map(|zt_key| {
            let seed = fxhash::hash64(zt_key);
            let mut master_key = EntangledHVec::new_deterministic(dim, seed);
            for i in 1..25 {
                master_key = master_key.bind(&EntangledHVec::new_deterministic(dim, seed + i));
            }
            master_key
        });

        let core = Self {
            cached_zt_key,
            config: config.clone(),
            mutation_gate: RwLock::new(()),
            revision: std::sync::atomic::AtomicU64::new(0),
            arena,
            dimensions: dim,
            storage_path: base_path,
            shards: RwLock::new(shard_set),
            graph: RelationStore::new(),
            documents: RwLock::new(std::collections::BTreeMap::new()),
            derived: RwLock::new(std::collections::BTreeMap::new()),
            atom_memory: atom_mem,
            composite_memory: comp_mem,
            triple_store: tri_store,
            role_registry: role_reg,
            rule_store: rule_st,
            decomposer: decomp,
            admission: adm,
            cognition_loop: parking_lot::Mutex::new(None),
            goal_store: if config.meaning.enabled {
                Some(GoalStore::new())
            } else {
                None
            },
            self_modifier: if config.meaning.enabled {
                Some(SelfModifier::new())
            } else {
                None
            },
            audit,
            #[cfg(feature = "security")]
            signing,
            #[cfg(feature = "security")]
            encryption,
            #[cfg(feature = "security")]
            identity_registry: parking_lot::RwLock::new(super::security::identity::IdentityRegistry::new()),
            #[cfg(feature = "provenance")]
            provenance,
            #[cfg(feature = "experimental")]
            connection_graph: parking_lot::Mutex::new(None),
            #[cfg(feature = "experimental")]
            phase_graph: parking_lot::Mutex::new(None),
            _store_lock: store_lock,
        };

        core.load_from_log()?;
        core.load_indices()?;
        {
            let shards = core.shards.read();
            shards.try_for_each_shard(|s| s.rebuild_inverted_index(dim))?;
        }

        Ok(core)
    }

    fn shard_meta_path(base: &Path) -> PathBuf {
        base.join("shard_meta.json")
    }

    /// Read the persisted shard count written by a prior auto-shard, if any.
    fn load_shard_meta(base: &Path) -> Option<usize> {
        let data = std::fs::read_to_string(Self::shard_meta_path(base)).ok()?;
        let value: serde_json::Value = serde_json::from_str(&data).ok()?;
        value.get("shard_count")?.as_u64().map(|n| n as usize)
    }

    fn save_shard_meta(&self, shard_count: usize) -> Result<()> {
        let json = serde_json::json!({ "shard_count": shard_count });
        super::durable_file::atomic_write(
            &Self::shard_meta_path(&self.storage_path),
            &serde_json::to_vec(&json)?,
        )?;
        Ok(())
    }

    /// Per-shard NSG index path. Single-shard stores keep the legacy filename so
    /// existing on-disk indices continue to load.
    fn nsg_index_path(&self, shard_idx: usize, multi: bool) -> PathBuf {
        if multi {
            self.storage_path.join(format!("nsg_index_{shard_idx}.bin"))
        } else {
            self.storage_path.join("nsg_index.bin")
        }
    }

    fn ivf_index_path(&self, shard_idx: usize, multi: bool) -> PathBuf {
        if multi {
            self.storage_path.join(format!("ivf_index_{shard_idx}.bin"))
        } else {
            self.storage_path.join("ivf_index.bin")
        }
    }

    fn load_indices(&self) -> Result<()> {
        let metadata = self.storage_path.join("indices.json");
        let Ok(bytes) = std::fs::read(metadata) else {
            return Ok(());
        };
        let saved: serde_json::Value = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(error) => {
                tracing::warn!(%error, "index metadata invalid; using rebuilt inverted index");
                return Ok(());
            }
        };
        if saved["checkpoint"] != serde_json::to_value(self.arena.checkpoint())? {
            tracing::info!("stale ANN checkpoint; using rebuilt inverted index");
            return Ok(());
        }
        let shards = self.shards.read();
        let multi = shards.shard_count() > 1;
        if saved["shardCount"] != shards.shard_count() {
            return Ok(());
        }
        shards.try_for_each_shard_indexed(|i, shard| {
            let read_cache = |path: &Path, expected_crc: &serde_json::Value| -> Result<Vec<u8>> {
                let raw = std::fs::read(path)?;
                anyhow::ensure!(
                    expected_crc.as_u64() == Some(crc32fast::hash(&raw) as u64),
                    "index cache checksum mismatch"
                );
                self.maybe_decrypt(&raw)
            };
            let nsg_path = self.nsg_index_path(i, multi);
            if nsg_path.exists() && saved["shards"][i]["nsg"] == true {
                let result = (|| -> Result<super::nsg::NSGIndex> {
                    let data = read_cache(&nsg_path, &saved["shards"][i]["nsgCrc"])?;
                    let (nsg, consumed) =
                        bincode::serde::decode_from_slice(&data, bincode::config::standard())?;
                    anyhow::ensure!(consumed == data.len(), "trailing index cache data");
                    Ok(nsg)
                })();
                match result {
                    Ok(nsg) => *shard.nsg.write() = Some(nsg),
                    Err(error) => tracing::warn!(%error, shard = i, "discarding invalid NSG cache"),
                }
            }

            let ivf_path = self.ivf_index_path(i, multi);
            if ivf_path.exists() && saved["shards"][i]["ivf"] == true {
                let result = (|| -> Result<IVFIndex> {
                    let data = read_cache(&ivf_path, &saved["shards"][i]["ivfCrc"])?;
                    let (mut ivf, _): (IVFIndex, usize) =
                        bincode::serde::decode_from_slice(&data, bincode::config::standard())?;
                    ivf.lists = Some(super::ivf::inverted_list::InvertedLists::new());

                    let vectors = shard.vectors.read();
                    let registry = shard.registry.read();
                    for id in registry.iter() {
                        if let Some(vec) = vectors.get(id) {
                            ivf.insert(id, vec)?;
                        }
                    }
                    Ok(ivf)
                })();
                match result {
                    Ok(ivf) => *shard.ivf.write() = Some(ivf),
                    Err(error) => tracing::warn!(%error, shard = i, "discarding invalid IVF cache"),
                }
            }
            Ok(())
        })
    }

    /// Persist every shard's trained NSG/IVF index to disk. Applies to both
    /// single- and multi-shard stores; multi-shard indices are written to
    /// per-shard files so they reload after a restart.
    fn persist_indices(&self, shards: &ShardSet) -> Result<()> {
        let multi = shards.shard_count() > 1;
        let mut states = Vec::new();
        shards.try_for_each_shard_indexed(|i, shard| {
            let mut nsg_crc = None;
            let mut ivf_crc = None;
            if let Some(ref nsg) = *shard.nsg.read() {
                let data = bincode::serde::encode_to_vec(nsg, bincode::config::standard())?;
                let data = self.maybe_encrypt(&data)?;
                nsg_crc = Some(crc32fast::hash(&data));
                super::durable_file::atomic_write(
                    &self.nsg_index_path(i, multi),
                    &data,
                )?;
            }
            if let Some(ref ivf) = *shard.ivf.read() {
                let data = bincode::serde::encode_to_vec(ivf, bincode::config::standard())?;
                let data = self.maybe_encrypt(&data)?;
                ivf_crc = Some(crc32fast::hash(&data));
                super::durable_file::atomic_write(
                    &self.ivf_index_path(i, multi),
                    &data,
                )?;
            }
            states
                .push(serde_json::json!({"nsg": shard.nsg_trained(), "ivf": shard.ivf_trained(), "nsgCrc": nsg_crc, "ivfCrc": ivf_crc}));
            Ok(())
        })?;
        super::durable_file::atomic_write(
            &self.storage_path.join("indices.json"),
            &serde_json::to_vec(
                &serde_json::json!({"checkpoint": self.arena.checkpoint(), "shardCount": shards.shard_count(), "shards": states}),
            )?,
        )
    }

    /// Bundle vectors respecting the PrivacyConfig.
    /// When dp_enabled, uses Laplace noise with the configured epsilon.
    pub fn bundle<V: std::borrow::Borrow<EntangledHVec>>(&self, vectors: &[V]) -> EntangledHVec {
        if self.config.privacy.dp_enabled {
            EntangledHVec::bundle_dp_in_space(vectors, self.config.privacy.epsilon, self.dimensions)
        } else {
            EntangledHVec::bundle(vectors)
        }
    }

    fn maybe_encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        #[cfg(feature = "security")]
        if let Some(ref enc) = self.encryption {
            return enc.encrypt(data);
        }
        Ok(data.to_vec())
    }

    fn maybe_decrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        #[cfg(feature = "security")]
        if let Some(ref enc) = self.encryption {
            return enc.decrypt(data);
        }
        Ok(data.to_vec())
    }

    fn load_from_log(&self) -> Result<()> {
        let shards = self.shards.write();
        let mut offset = 0;
        while offset < self.arena.stats().used_bytes {
            let (payload, _) = self
                .arena_read_frame(offset)
                .with_context(|| format!("cannot replay arena frame at {offset}"))?;
            let json = payload.strip_prefix(mutation::TRANSACTION_MAGIC).context(
                "unsupported transaction format; re-encode original sources into a new store",
            )?;
            let mutations: Vec<Mutation> = serde_json::from_slice(json)?;
            for mutation in &mutations {
                self.validate_mutation(mutation)?;
            }
            self.apply_mutations(&mutations, &shards)?;
            offset = self.arena.next_offset(offset)?;
        }
        if let Some(atoms) = &self.atom_memory {
            atoms.rebuild_indices();
        }
        if let Some(composites) = &self.composite_memory {
            composites.rebuild_indices();
        }
        Ok(())
    }

    /// Returns the dimensionality of the hypervector space.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Encode text into a sparse hypervector using multi-scale n-grams and word tokens.
    pub fn encode_text(&self, text: &str) -> EntangledHVec {
        encode_text_internal(text, self.dimensions)
    }

    /// Compute word, sentence, syllable, and character-class counts for text.
    pub fn analyze_text(&self, text: &str) -> TextMetrics {
        TextProcessor::analyze(text)
    }

    /// Compute Flesch Reading Ease score from text metrics.
    pub fn calculate_readability(&self, metrics: &TextMetrics) -> f64 {
        TextProcessor::calculate_readability(metrics)
    }

    /// Delete a vector by ID. Returns true if it existed. Crash-safe: tombstone is persisted first.
    pub fn delete(&self, id: &str) -> Result<bool> {
        self.delete_with_reason(id, None)
    }

    pub fn delete_with_reason(
        &self,
        id: &str,
        #[allow(unused_variables)] reason: Option<&str>,
    ) -> Result<bool> {
        mutation::validate_public_id(id)?;
        anyhow::ensure!(
            !id.starts_with(super::documents::CHUNK_PREFIX),
            "use deleteDocument to delete document chunks"
        );
        let existed = self.shards.read().get_vector(id).is_some();
        self.commit(&[Mutation::Delete { id: id.to_owned() }])?;
        if let Some(ref audit) = self.audit {
            audit.record(AuditOp::Delete, id, self.sign_fn().as_deref())?;
        }
        #[cfg(feature = "provenance")]
        if let Some(ref mgr) = self.provenance {
            if self.config.provenance.auto_sign {
                mgr.record_deletion(id, reason)?;
            }
        }
        Ok(existed)
    }

    pub fn memorize_meaning(&self, id: &str, text: &str) -> Result<()> {
        self.memorize_meaning_with_source(id, text, None)
    }

    pub fn memorize_meaning_with_source(
        &self,
        id: &str,
        text: &str,
        #[allow(unused_variables)] source_uri: Option<&str>,
    ) -> Result<()> {
        mutation::validate_public_id(id)?;
        let mut mutations = vec![Mutation::Vector {
            id: id.to_owned(),
            vector: self.encode_text(text),
        }];
        let mut children = Vec::new();
        if self.config.meaning.auto_decompose {
            if let Some(decomposer) = &self.decomposer {
                for (i, unit) in decomposer.decompose(text).into_iter().enumerate() {
                    let child = format!("{}{}:{id}:{i}", mutation::MEANING_PREFIX, id.len());
                    children.push(child.clone());
                    mutations.push(Mutation::Triplet {
                        id: child,
                        subject: unit.subject,
                        relation: unit.relation,
                        object: unit.object,
                    });
                }
            }
        }
        if !children.is_empty() {
            mutations.push(Mutation::Derived {
                id: id.to_owned(),
                children,
            });
        }
        self.commit(&mutations)?;
        #[cfg(feature = "provenance")]
        if let Some(ref mgr) = self.provenance {
            if self.config.provenance.auto_sign {
                mgr.create_fact_provenance(id, text.as_bytes(), source_uri)?;
            }
        }
        Ok(())
    }

    /// Publish a verified generation while excluding every persistent mutation.
    pub fn compact(&self) -> Result<()> {
        let _transaction = self.mutation_gate.write();
        let shards = self.shards.write();
        let temporary = tempfile::Builder::new()
            .prefix(".compact-")
            .tempdir_in(&self.storage_path)?;
        {
            let arena = PersistentArena::new(temporary.path())?;
            let write = |mutation: Mutation| -> Result<()> {
                let mut payload = mutation::TRANSACTION_MAGIC.to_vec();
                payload.extend(serde_json::to_vec(&[mutation])?);
                arena.write_slice(&self.maybe_encrypt(&payload)?)?;
                Ok(())
            };
            let composite_ids: std::collections::HashSet<_> = self
                .composite_memory
                .as_ref()
                .map(|memory| {
                    memory
                        .inner()
                        .all_vectors()
                        .into_iter()
                        .map(|(_, id, _)| id)
                        .collect()
                })
                .unwrap_or_default();
            shards.try_for_each_shard(|shard| {
                for (id, vector) in shard.vectors.read().iter() {
                    if !id.starts_with(super::documents::CHUNK_PREFIX)
                        && !composite_ids.contains(id)
                    {
                        write(Mutation::Vector {
                            id: id.clone(),
                            vector: vector.clone(),
                        })?;
                    }
                }
                Ok(())
            })?;
            for doc in self.documents.read().values() {
                write(Mutation::Document(doc.clone()))?;
            }
            for relation in self.graph.snapshot() {
                write(Mutation::Relation(relation))?;
            }
            for relation_type in self.graph.type_snapshot() {
                write(Mutation::RelationType(relation_type))?;
            }
            if let Some(atoms) = &self.atom_memory {
                for (_, id, vector) in atoms.inner().all_vectors() {
                    write(Mutation::Atom { id, vector })?;
                }
            }
            if let Some(composites) = &self.composite_memory {
                for (_, id, vector) in composites.inner().all_vectors() {
                    write(Mutation::Composite { id, vector })?;
                }
            }
            if let Some(triples) = &self.triple_store {
                for record in triples.snapshot() {
                    write(Mutation::StoredTriple(record))?;
                }
            }
            if let Some(rules) = &self.rule_store {
                for rule in rules.all_rules() {
                    write(Mutation::Rule(rule))?;
                }
            }
            for (id, children) in self.derived.read().iter() {
                write(Mutation::Derived {
                    id: id.clone(),
                    children: children.clone(),
                })?;
            }
            arena.flush()?;
        }
        self.arena.replace_with_compacted(temporary.path())?;
        self.revision
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        self.persist_indices(&shards)?;
        if let Some(audit) = &self.audit {
            audit.record(AuditOp::Compact, "", self.sign_fn().as_deref())?;
        }
        Ok(())
    }

    /// Store a vector with the given ID. Persists to the arena log and updates all indices.
    pub fn memorize(&self, id: String, mut vector: EntangledHVec) -> Result<()> {
        mutation::validate_public_id(&id)?;
        anyhow::ensure!(
            !id.starts_with(super::documents::CHUNK_PREFIX),
            "chunk IDs are reserved for document ingestion"
        );

        // --- FHE-Lite / Zero-Trust Encryption ---
        if let Some(ref zt_key) = self.config.privacy.zero_trust_key {
            let seed = fxhash::hash64(zt_key);
            let mut master_key = EntangledHVec::new_deterministic(self.dimensions, seed);
            for i in 1..25 {
                master_key = master_key.bind(&EntangledHVec::new_deterministic(self.dimensions, seed + i));
            }
            vector = vector.bind(&master_key);
        }

        self.commit(&[Mutation::Vector {
            id: id.clone(),
            vector,
        }])?;
        if let Some(ref audit) = self.audit {
            audit.record(AuditOp::Memorize, &id, self.sign_fn().as_deref())?;
        }
        Ok(())
    }

    fn maybe_auto_shard(&self, count: u64) -> Result<()> {
        let cfg = &self.config.shard;
        if !cfg.enabled
            || cfg.shard_count > 0
            || cfg.auto_threshold == 0
            || count < cfg.auto_threshold as u64
        {
            return Ok(());
        }
        let (revision, snapshot) = {
            let _transaction = self.mutation_gate.read();
            let shards = self.shards.read();
            if shards.shard_count() > 1 {
                return Ok(());
            }
            (
                self.revision.load(std::sync::atomic::Ordering::Acquire),
                shards.collect_all_patterns(),
            )
        };
        let count = (count as usize / cfg.target_shard_size).clamp(2, 1024);
        let new_set = ShardSet::Multi(ShardManager::new(count, self.dimensions));
        for (id, vector) in snapshot {
            new_set.insert(id, vector, self.dimensions)?;
        }
        let _transaction = self.mutation_gate.write();
        anyhow::ensure!(
            revision == self.revision.load(std::sync::atomic::Ordering::Acquire),
            "store changed during sharding; retry maintenance"
        );
        self.save_shard_meta(count)?;
        *self.shards.write() = new_set;
        self.revision
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        self.persist_indices(&self.shards.read())
    }

    /// Convert a dense f32 vector to sparse and memorize it.
    pub fn memorize_vector(&self, id: String, dense: &[f32]) -> Result<()> {
        self.validate_dense(dense)?;
        let vector = EntangledHVec::from_dense(dense, self.dimensions);
        self.memorize(id, vector)
    }

    /// Encode a bounded scalar value as a hypervector and memorize it.
    pub fn memorize_scalar(&self, id: String, value: f64, min: f64, max: f64) -> Result<()> {
        let vector = EntangledHVec::from_scalar(value, min, max, self.dimensions);
        self.memorize(id, vector)
    }

    /// Returns the total number of stored vectors across all shards.
    pub fn vector_count(&self) -> u64 {
        self.shards.read().count()
    }

    // === Graph API ===

    pub fn add_relation(&self, rel: &Relation) -> Result<()> {
        self.commit(&[Mutation::Relation(rel.clone())])
    }

    pub fn remove_relation(
        &self,
        source_id: &str,
        relation_type: &str,
        target_id: &str,
    ) -> Result<bool> {
        let existed = self
            .graph
            .outgoing(source_id, Some(relation_type), 0.0)
            .iter()
            .any(|r| r.target_id == target_id);
        self.commit(&[Mutation::RemoveRelation {
            source: source_id.into(),
            relation: relation_type.into(),
            target: target_id.into(),
        }])?;
        Ok(existed)
    }

    pub fn declare_relation_type(&self, rel_type: RelationType) -> Result<()> {
        self.commit(&[Mutation::RelationType(rel_type)])
    }

    pub fn traverse(
        &self,
        start_id: &str,
        relation_type: Option<&str>,
        max_depth: u32,
        at_time: f64,
    ) -> Vec<GraphPath> {
        let _transaction = self.mutation_gate.read();
        let shards = self.shards.read();
        self.graph
            .traverse(start_id, relation_type, max_depth, at_time, &|a, b| {
                let vec_a = shards.get_vector(a);
                let vec_b = shards.get_vector(b);
                match (vec_a, vec_b) {
                    (Some(va), Some(vb)) => va.similarity(&vb),
                    _ => 0.0,
                }
            })
    }

    pub fn outgoing_relations(
        &self,
        source_id: &str,
        relation_type: Option<&str>,
        at_time: f64,
    ) -> Vec<Relation> {
        let _transaction = self.mutation_gate.read();
        self.graph.outgoing(source_id, relation_type, at_time)
    }

    pub fn incoming_relations(
        &self,
        target_id: &str,
        relation_type: Option<&str>,
        at_time: f64,
    ) -> Vec<Relation> {
        let _transaction = self.mutation_gate.read();
        self.graph.incoming(target_id, relation_type, at_time)
    }

    pub fn relation_count(&self) -> usize {
        let _transaction = self.mutation_gate.read();
        self.graph.count()
    }

    // === Federated Query ===

    /// Query across local instance and remote peers in parallel.
    ///
    /// Improvements over naive collect-sort-truncate:
    /// - **BinaryHeap merge**: O(N log k) top-k selection instead of O(N log N) sort.
    /// - **ID deduplication**: same document appearing in multiple peers is counted once
    ///   (highest similarity wins).
    /// - **Partial failure tolerance**: a failed peer logs a warning and is skipped
    ///   rather than aborting the entire query.
    pub fn federated_query(
        &self,
        peer_paths: &[String],
        query_vec: &EntangledHVec,
        k: u32,
    ) -> Result<Vec<super::types::RetrievalResult>> {
        use rayon::prelude::*;
        use std::collections::BinaryHeap;

        let k = k as usize;

        // Query local instance
        let local_results = self.query(query_vec, k as u32);

        // Query each peer in parallel; collect successes and log failures
        let peer_outcomes: Vec<(String, Result<Vec<super::types::RetrievalResult>>)> = peer_paths
            .par_iter()
            .map(|path| {
                let outcome = HmsCore::new(
                    self.dimensions as u32,
                    Some(path.clone()),
                    Some(self.config.clone()),
                )
                .map(|peer| peer.query(query_vec, k as u32));
                (path.clone(), outcome)
            })
            .collect();

        // Deduplicate by ID, keeping highest similarity per ID.
        let mut seen: fxhash::FxHashMap<String, f64> =
            fxhash::FxHashMap::with_capacity_and_hasher(k * 2, Default::default());

        // BinaryHeap with RetrievalResult's Ord (min-heap by similarity):
        // pop() removes the lowest-similarity item, so we maintain top-k.
        let mut heap: BinaryHeap<super::types::RetrievalResult> = BinaryHeap::with_capacity(k + 1);

        let mut insert = |r: super::types::RetrievalResult| {
            // Dedup: skip if we already have this ID with equal-or-higher similarity
            let dominated = seen
                .get(&r.id)
                .is_some_and(|&prev_sim| prev_sim >= r.similarity);
            if dominated {
                return;
            }
            seen.insert(r.id.clone(), r.similarity);
            heap.push(r);
            if heap.len() > k {
                if let Some(evicted) = heap.pop() {
                    seen.remove(&evicted.id);
                }
            }
        };

        for r in local_results {
            insert(r);
        }

        for (path, outcome) in peer_outcomes {
            match outcome {
                Ok(results) => {
                    for r in results {
                        insert(r);
                    }
                }
                Err(e) => {
                    tracing::warn!("federated query: peer {:?} failed, skipping: {}", path, e);
                }
            }
        }

        // Drain heap into sorted vec (highest similarity first)
        let mut results: Vec<super::types::RetrievalResult> = heap.into_sorted_vec();
        // into_sorted_vec uses the Ord (min-heap), so lowest similarity is first.
        // Reverse to get descending order.
        results.reverse();
        Ok(results)
    }

    // === Meaning Memory API ===

    /// Build the shared meaning-memory context, if all required subsystems are
    /// present. Returns `None` when meaning memory is not fully initialized.
    fn meaning_ctx(&self) -> Option<structural::MeaningContext<'_>> {
        let (atom_mem, comp_mem, tri, roles, adm) = match (
            &self.atom_memory,
            &self.composite_memory,
            &self.triple_store,
            &self.role_registry,
            &self.admission,
        ) {
            (Some(a), Some(c), Some(t), Some(r), Some(ad)) => (a, c, t, r, ad),
            _ => return None,
        };
        let mc = &self.config.meaning;
        Some(structural::MeaningContext {
            atom_memory: atom_mem,
            composite_memory: comp_mem,
            triple_store: tri,
            roles,
            admission: adm,
            beta: mc.beta,
            k: 64,
            max_iter: 3,
        })
    }

    pub fn structural_query(
        &self,
        known: &[(&str, &EntangledHVec)],
        target_role: &str,
    ) -> Vec<structural::StructuralResult> {
        let _transaction = self.mutation_gate.read();
        let ctx = match self.meaning_ctx() {
            Some(ctx) => ctx,
            None => return Vec::new(),
        };
        structural::fuzzy_structural_query(&ctx, known, target_role)
    }

    pub fn multi_hop(&self, start: &str, relations: &[&str]) -> Vec<multi_hop::MultiHopResult> {
        let _transaction = self.mutation_gate.read();
        let (ctx, rules) = match (self.meaning_ctx(), &self.rule_store) {
            (Some(ctx), Some(rules)) => (ctx, rules),
            _ => return Vec::new(),
        };
        let mc = &self.config.meaning;
        multi_hop::multi_hop_query(start, relations, &ctx, rules, mc.max_hop_depth)
    }

    pub fn meaning_cleanup(&self, noisy: &EntangledHVec) -> Option<(String, f64)> {
        let atom_mem = self.atom_memory.as_ref()?;
        let mc = &self.config.meaning;
        let result = atom_mem.cleanup(noisy, mc.beta, 64, 3);
        if result.found {
            Some((result.id, result.confidence))
        } else {
            None
        }
    }

    pub fn declare_rule(
        &self,
        name: &str,
        input_relations: Vec<String>,
        output_relation: String,
    ) -> Result<()> {
        self.commit(&[Mutation::Rule(super::rules::CompositionRule {
            name: name.to_string(),
            input_relations,
            output_relation,
        })])
    }

    pub fn meaning_enabled(&self) -> bool {
        self.config.meaning.enabled
    }

    pub fn meaning_atom_count(&self) -> usize {
        self.atom_memory.as_ref().map_or(0, |m| m.count())
    }

    pub fn meaning_composite_count(&self) -> usize {
        self.composite_memory.as_ref().map_or(0, |m| m.count())
    }

    pub fn meaning_triple_count(&self) -> usize {
        self.triple_store.as_ref().map_or(0, |t| t.count())
    }

    pub fn meaning_rule_count(&self) -> usize {
        self.rule_store.as_ref().map_or(0, |r| r.count())
    }

    pub fn register_role(&mut self, name: &str, shift: usize) -> anyhow::Result<()> {
        if let Some(ref mut roles) = self.role_registry {
            roles.register(name, shift)
        } else {
            Err(anyhow::anyhow!("meaning memory not enabled"))
        }
    }

    // === Cognition API ===

    pub fn start_cognition(&self) -> Result<()> {
        if self.cognition_running() {
            return Err(anyhow::anyhow!("cognition loop is already running"));
        }
        let atom_mem = self
            .atom_memory
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("meaning memory not enabled"))?;
        let tri_store = self
            .triple_store
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("meaning memory not enabled"))?;

        let cl = CognitionLoop::start(
            Arc::clone(atom_mem),
            Arc::clone(tri_store),
            self.cognition_loop_config(),
        );

        *self.cognition_loop.lock() = Some(cl);
        Ok(())
    }

    pub fn stop_cognition(&self) {
        if let Some(ref mut cl) = *self.cognition_loop.lock() {
            cl.stop();
        }
    }

    pub fn cognition_running(&self) -> bool {
        self.cognition_loop
            .lock()
            .as_ref()
            .is_some_and(|cl| cl.state().is_running())
    }

    pub fn cognition_cycle_count(&self) -> u64 {
        self.cognition_loop
            .lock()
            .as_ref()
            .map_or(0, |cl| cl.state().cycle_count())
    }

    pub fn take_insights(&self) -> Vec<Insight> {
        self.cognition_loop
            .lock()
            .as_ref()
            .map_or_else(Vec::new, |cl| cl.state().take_insights())
    }

    pub fn cognition_insight_count(&self) -> usize {
        self.cognition_loop
            .lock()
            .as_ref()
            .map_or(0, |cl| cl.state().insight_count())
    }

    fn cognition_loop_config(&self) -> CognitionLoopConfig {
        let cc = &self.config.cognition;
        CognitionLoopConfig {
            interval: std::time::Duration::from_secs(cc.interval_secs),
            min_pattern_freq: cc.min_pattern_freq,
            min_abstraction_members: cc.min_abstraction_members,
            min_shared_relations: cc.min_shared_relations,
            min_peer_coverage: cc.min_peer_coverage,
            hypothesis_beta: cc.hypothesis_beta,
            min_hypothesis_confidence: cc.min_hypothesis_confidence,
            min_analogy_relations: cc.min_analogy_relations,
        }
    }

    pub fn run_cognition_once(&self) -> Vec<Insight> {
        let (atom_mem, tri_store) = match (&self.atom_memory, &self.triple_store) {
            (Some(a), Some(t)) => (a, t),
            _ => return Vec::new(),
        };
        let cfg = self.cognition_loop_config();
        CognitionLoop::run_once(atom_mem, tri_store, &cfg)
    }

    pub fn govern_memory(&self) -> GovernanceReport {
        let (atom_mem, comp_mem, tri_store) = match (
            &self.atom_memory,
            &self.composite_memory,
            &self.triple_store,
        ) {
            (Some(a), Some(c), Some(t)) => (a, c, t),
            _ => return GovernanceReport::default(),
        };
        let cc = &self.config.cognition;
        let gov_config = GovernorConfig {
            duplicate_threshold: cc.governor_duplicate_threshold,
            max_scan_size: cc.governor_max_scan_size,
            forget_unreferenced_atoms: cc.governor_forget_unreferenced,
            refine_atoms: cc.refine_atoms,
            ..Default::default()
        };
        MemoryGovernor::govern(atom_mem, comp_mem, tri_store, &gov_config)
    }

    pub fn cognition_enabled(&self) -> bool {
        self.config.cognition.enabled
    }

    // === Agency API ===

    pub fn add_goal(
        &self,
        name: &str,
        description: &str,
        relevance: f64,
        urgency: f64,
        cost: f64,
    ) -> Option<usize> {
        let goal_store = self.goal_store.as_ref()?;
        let atom_mem = self.atom_memory.as_ref()?;
        let (_, vec) = atom_mem.get_or_insert(name);
        Some(goal_store.add(super::agency::goals::Goal {
            name: name.to_string(),
            description: description.to_string(),
            vector: vec,
            relevance,
            urgency,
            cost,
            active: true,
        }))
    }

    pub fn deactivate_goal(&self, name: &str) -> bool {
        self.goal_store
            .as_ref()
            .is_some_and(|gs| gs.deactivate(name))
    }

    pub fn active_goals(&self) -> Vec<(String, f64)> {
        self.goal_store.as_ref().map_or_else(Vec::new, |gs| {
            gs.prioritized()
                .iter()
                .map(|g| (g.name.clone(), g.utility()))
                .collect()
        })
    }

    pub fn plan_goal(&self, goal: &str, causal_relations: &[&str], max_depth: usize) -> Plan {
        let tri = match &self.triple_store {
            Some(t) => t,
            None => {
                return Plan {
                    goal: goal.to_string(),
                    actions: Vec::new(),
                    complete: false,
                    total_cost: 0.0,
                }
            }
        };
        Planner::backward_chain(tri, goal, causal_relations, max_depth)
    }

    pub fn generate_questions(&self) -> Vec<Question> {
        let (atom_mem, tri_store, goal_store) =
            match (&self.atom_memory, &self.triple_store, &self.goal_store) {
                (Some(a), Some(t), Some(g)) => (a, t, g),
                _ => return Vec::new(),
            };

        let cc = &self.config.cognition;
        let gaps = super::cognition::gaps::GapDetector::detect(
            tri_store,
            cc.min_shared_relations,
            cc.min_peer_coverage,
        );
        let hypotheses = super::cognition::hypothesis::HypothesisEngine::propose(
            &gaps,
            tri_store,
            atom_mem,
            cc.hypothesis_beta,
            cc.min_hypothesis_confidence,
        );

        let mut questions = Vec::new();
        questions.extend(QuestionGenerator::from_gaps(&gaps, atom_mem, goal_store));
        questions.extend(QuestionGenerator::from_hypotheses(
            &hypotheses,
            atom_mem,
            goal_store,
        ));
        QuestionGenerator::prioritize(questions)
    }

    pub fn propose_rule(
        &self,
        name: &str,
        input_relations: Vec<String>,
        output_relation: &str,
        reason: &str,
    ) -> Option<usize> {
        let sm = self.self_modifier.as_ref()?;
        Some(sm.propose(
            ProposalKind::AddRule {
                name: name.to_string(),
                input_relations,
                output_relation: output_relation.to_string(),
            },
            reason.to_string(),
        ))
    }

    pub fn approve_proposal(&self, id: usize) -> bool {
        self.self_modifier.as_ref().is_some_and(|sm| sm.approve(id))
    }

    pub fn reject_proposal(&self, id: usize) -> bool {
        self.self_modifier.as_ref().is_some_and(|sm| sm.reject(id))
    }

    pub fn pending_proposals(&self) -> usize {
        self.self_modifier
            .as_ref()
            .map_or(0, |sm| sm.pending_count())
    }

    pub fn goal_count(&self) -> usize {
        self.goal_store.as_ref().map_or(0, |gs| gs.count())
    }

    pub fn active_goal_count(&self) -> usize {
        self.goal_store.as_ref().map_or(0, |gs| gs.active_count())
    }

    /// Returns true if the IVF index has been trained.
    pub fn ivf_trained(&self) -> bool {
        self.shards.read().ivf_trained()
    }

    /// Train the IVF index on current vectors. Persists the index to disk.
    pub fn train_ivf(&self) -> Result<()> {
        let (revision, snapshots) = {
            let _transaction = self.mutation_gate.read();
            let shards = self.shards.read();
            let mut snapshots = Vec::new();
            shards.for_each_shard(|shard| snapshots.push(shard.load_all_vectors()));
            (
                self.revision.load(std::sync::atomic::Ordering::Acquire),
                snapshots,
            )
        };
        let mut indices = Vec::with_capacity(snapshots.len());
        for (ids, vectors) in snapshots {
            indices.push(if ids.is_empty() {
                None
            } else {
                Some(IVFIndex::train(
                    &vectors,
                    &ids,
                    self.dimensions,
                    &self.config.ivf,
                )?)
            });
        }
        let _transaction = self.mutation_gate.write();
        anyhow::ensure!(
            revision == self.revision.load(std::sync::atomic::Ordering::Acquire),
            "store changed during training; retry maintenance"
        );
        let shards = self.shards.read();
        shards.try_for_each_shard_indexed(|i, shard| {
            *shard.ivf.write() = indices[i].take();
            Ok(())
        })?;
        self.persist_indices(&shards)
    }

    /// Returns true if the NSG graph index has been trained.
    pub fn nsg_trained(&self) -> bool {
        self.shards.read().nsg_trained()
    }

    /// Train the NSG graph index on current vectors. Persists the index to disk.
    pub fn train_nsg(&self) -> Result<()> {
        let (revision, snapshots) = {
            let _transaction = self.mutation_gate.read();
            let shards = self.shards.read();
            let mut snapshots = Vec::new();
            shards.for_each_shard(|shard| snapshots.push(shard.load_all_vectors()));
            (
                self.revision.load(std::sync::atomic::Ordering::Acquire),
                snapshots,
            )
        };
        let mut indices = Vec::with_capacity(snapshots.len());
        for (ids, vectors) in snapshots {
            indices.push(if ids.is_empty() {
                None
            } else {
                Some(super::nsg::training::train(
                    &vectors,
                    &ids,
                    &self.config.nsg,
                )?)
            });
        }
        let _transaction = self.mutation_gate.write();
        anyhow::ensure!(
            revision == self.revision.load(std::sync::atomic::Ordering::Acquire),
            "store changed during training; retry maintenance"
        );
        let shards = self.shards.read();
        shards.try_for_each_shard_indexed(|i, shard| {
            *shard.nsg.write() = indices[i].take();
            Ok(())
        })?;
        self.persist_indices(&shards)
    }

    fn arena_write(&self, data: &[u8]) -> Result<usize> {
        let payload = self.maybe_encrypt(data)?;
        self.arena.write_slice(&payload)
    }

    fn arena_read_frame(&self, offset: usize) -> Result<(Vec<u8>, u32)> {
        let (data, version) = self.arena.read_frame(offset)?;
        let payload = self.maybe_decrypt(&data)?;
        Ok((payload, version))
    }

    fn sign_fn(&self) -> Option<SignFn<'_>> {
        #[cfg(feature = "security")]
        {
            self.signing
                .as_ref()
                .map(|s| Box::new(move |data: &[u8]| s.sign(data)) as SignFn<'_>)
        }
        #[cfg(not(feature = "security"))]
        {
            None
        }
    }

    /// Query the audit log for entries since `timestamp_ms`.
    /// Returns an empty vec if audit logging is disabled.
    pub fn audit_since(&self, timestamp_ms: u64) -> Result<Vec<super::audit::AuditEntry>> {
        match self.audit {
            Some(ref audit) => audit.entries_since(timestamp_ms),
            None => Ok(Vec::new()),
        }
    }

    // === Provenance API ===

    #[cfg(feature = "provenance")]
    pub fn provenance_enabled(&self) -> bool {
        self.provenance.is_some()
    }

    #[cfg(feature = "provenance")]
    pub fn create_fact_provenance(
        &self,
        fact_id: &str,
        content: &[u8],
        source_uri: Option<&str>,
    ) -> Result<super::provenance::types::ProvenanceRecord> {
        let mgr = self
            .provenance
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("provenance not enabled"))?;
        mgr.create_fact_provenance(fact_id, content, source_uri)
    }

    #[cfg(feature = "provenance")]
    pub fn create_triple_provenance(
        &self,
        params: &super::provenance::TripleProvenanceParams<'_>,
    ) -> Result<super::provenance::types::ProvenanceRecord> {
        let mgr = self
            .provenance
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("provenance not enabled"))?;
        mgr.create_triple_provenance(params)
    }

    #[cfg(feature = "provenance")]
    pub fn create_store_manifest(
        &self,
        store_id: &str,
        store_data: &[u8],
        fact_count: usize,
        dimensions: u32,
        title: Option<&str>,
    ) -> Result<super::provenance::types::StoreManifest> {
        let mgr = self
            .provenance
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("provenance not enabled"))?;
        mgr.create_store_manifest(store_id, store_data, fact_count, dimensions, title)
    }

    #[cfg(feature = "provenance")]
    pub fn verify_fact_provenance(
        &self,
        record: &super::provenance::types::ProvenanceRecord,
        trust: &super::provenance::trust::TrustStore,
    ) -> Result<super::provenance::types::VerificationResult> {
        let mgr = self
            .provenance
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("provenance not enabled"))?;
        mgr.verify_fact_provenance(record, trust)
    }

    /// A trust store containing this instance's own provenance key, for
    /// authenticating records it signed. `None` if provenance is disabled.
    #[cfg(feature = "provenance")]
    pub fn provenance_self_trust(&self) -> Option<super::provenance::trust::TrustStore> {
        self.provenance.as_ref().map(|mgr| mgr.self_trust())
    }

    #[cfg(feature = "provenance")]
    pub fn verify_store_manifest(
        &self,
        manifest: &super::provenance::types::StoreManifest,
        store_data: Option<&[u8]>,
        trust: &super::provenance::trust::TrustStore,
    ) -> Result<super::provenance::types::VerificationResult> {
        let mgr = self
            .provenance
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("provenance not enabled"))?;
        mgr.verify_store_manifest(manifest, store_data, trust)
    }

    #[cfg(feature = "provenance")]
    pub fn issuer_did(&self) -> Option<&str> {
        self.provenance.as_ref().map(|mgr| mgr.issuer_did())
    }

    #[cfg(feature = "provenance")]
    pub fn get_provenance(
        &self,
        fact_id: &str,
    ) -> Option<super::provenance::types::ProvenanceRecord> {
        self.provenance.as_ref()?.get_record(fact_id)
    }

    #[cfg(feature = "provenance")]
    pub fn provenance_count(&self) -> usize {
        self.provenance.as_ref().map_or(0, |mgr| mgr.record_count())
    }

    #[cfg(feature = "provenance")]
    pub fn create_self_manifest(
        &self,
        title: Option<&str>,
    ) -> Result<super::provenance::types::StoreManifest> {
        let mgr = self
            .provenance
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("provenance not enabled"))?;
        let store_data =
            std::fs::read(self.storage_path.join("vectors_data.bin")).unwrap_or_default();
        let store_id = self
            .storage_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("default");
        mgr.create_store_manifest(
            store_id,
            &store_data,
            self.vector_count() as usize,
            self.dimensions as u32,
            title,
        )
    }

    #[cfg(feature = "provenance")]
    pub fn revoke_credential(&self, status_index: u64) -> Result<()> {
        let mgr = self
            .provenance
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("provenance not enabled"))?;
        mgr.revoke_credential(status_index)
    }

    #[cfg(feature = "provenance")]
    pub fn is_credential_revoked(&self, status_index: u64) -> bool {
        self.provenance
            .as_ref()
            .is_some_and(|mgr| mgr.is_revoked(status_index))
    }

    #[cfg(feature = "provenance")]
    pub fn verify_provenance_log(&self) -> Result<bool> {
        let mgr = self
            .provenance
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("provenance not enabled"))?;
        mgr.verify_log_integrity()
    }

    #[cfg(feature = "provenance")]
    pub fn create_batch_provenance(
        &self,
        items: &[(&str, &[u8], Option<&str>)],
    ) -> Result<Vec<super::provenance::types::ProvenanceRecord>> {
        let mgr = self
            .provenance
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("provenance not enabled"))?;
        mgr.create_batch_provenance(items)
    }

    #[cfg(feature = "provenance")]
    pub fn create_sigstore_bundle(
        &self,
        content: &[u8],
        identity: Option<&str>,
    ) -> Result<super::provenance::sigstore::SigstoreBundle> {
        let mgr = self
            .provenance
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("provenance not enabled"))?;
        mgr.create_sigstore_bundle(content, identity)
    }

    #[cfg(feature = "provenance")]
    pub fn verify_sigstore_bundle(
        &self,
        bundle: &super::provenance::sigstore::SigstoreBundle,
        content: &[u8],
        trusted_key: &ed25519_dalek::VerifyingKey,
    ) -> Result<()> {
        let mgr = self
            .provenance
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("provenance not enabled"))?;
        mgr.verify_sigstore_bundle(bundle, content, trusted_key)
    }

    #[cfg(feature = "provenance")]
    pub fn create_cawg_assertion(
        &self,
        referenced: &[(String, String, Vec<u8>)],
        display_name: &str,
    ) -> Result<serde_json::Value> {
        let mgr = self
            .provenance
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("provenance not enabled"))?;
        mgr.create_cawg_assertion(referenced, display_name)
    }

    #[cfg(feature = "provenance")]
    pub fn verify_cawg_assertion(
        &self,
        assertion: &serde_json::Value,
        trust: &super::provenance::trust::TrustStore,
    ) -> Result<serde_json::Value> {
        let mgr = self
            .provenance
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("provenance not enabled"))?;
        mgr.verify_cawg_assertion(assertion, trust)
    }

    /// Decompose a product vector into factors from domain codebooks using diffusion.
    pub fn factorize_diffusion(
        &self,
        product: &EntangledHVec,
        domain_codebooks: &[Vec<EntangledHVec>],
        max_iter: usize,
    ) -> Vec<Option<EntangledHVec>> {
        DiffusionFactorizer::factorize(&self.config.diffusion, product, domain_codebooks, max_iter)
    }

    /// Lock the connection graph, lazily opening it over its durable append-only
    /// event log at `{storage}/connection_graph.log` on first use so plastic
    /// state survives restart. Falls back to in-memory if the log is unopenable.
    #[cfg(feature = "experimental")]
    fn connection_graph_lazy(
        &self,
    ) -> parking_lot::MutexGuard<'_, Option<super::connection_graph::ConnectionGraph>> {
        let mut guard = self.connection_graph.lock();
        if guard.is_none() {
            let path = self.storage_path.join("connection_graph.log");
            let graph = super::connection_graph::ConnectionGraph::open(
                self.dimensions,
                super::connection_graph::GraphConfig::default(),
                &path,
            )
            .unwrap_or_else(|e| {
                tracing::warn!("connection graph persistence unavailable ({e}); in-memory only");
                super::connection_graph::ConnectionGraph::new(self.dimensions)
            });
            *guard = Some(graph);
        }
        guard
    }

    /// Assert a `(subject, relation, object)` edge into the experimental plastic
    /// connection graph. Opt-in and additive: independent of the sparse-binary
    /// store. Durable — the edge is appended to the graph's event log.
    #[cfg(feature = "experimental")]
    pub fn relate(&self, subject: &str, relation: &str, object: &str) {
        let mut guard = self.connection_graph_lazy();
        if let Some(g) = guard.as_mut() {
            g.store(subject, relation, object);
        }
    }

    /// Query a relation in the plastic connection graph, returning its presence
    /// score. This is a *mutating* read (the observer effect): it reinforces the
    /// queried relation and lets the untouched background decay, all recorded as
    /// verifiable, durable events.
    #[cfg(feature = "experimental")]
    pub fn query_relation(&self, subject: &str, relation: &str, object: &str) -> f64 {
        let mut guard = self.connection_graph_lazy();
        guard
            .as_mut()
            .map_or(0.0, |g| g.query(subject, relation, object))
    }

    /// Dimension and phase-quantization of the experimental phasor memory. A
    /// modest dim keeps the (dim * n_phases) histogram small; retrieval works
    /// well past this substrate's load in the validation experiments.
    #[cfg(feature = "experimental")]
    fn phase_graph_lazy(
        &self,
    ) -> parking_lot::MutexGuard<'_, Option<super::phase_graph::PhaseGraph>> {
        let mut guard = self.phase_graph.lock();
        if guard.is_none() {
            *guard = Some(super::phase_graph::PhaseGraph::new(2048, 256));
        }
        guard
    }

    /// Assert `(subject, relation, object)` into the experimental phasor memory,
    /// which supports relation algebra (rotation binding) and associative
    /// retrieval the sparse-binary store cannot. Opt-in and additive.
    #[cfg(feature = "experimental")]
    pub fn relate_phase(&self, subject: &str, relation: &str, object: &str) {
        self.phase_graph_lazy()
            .as_mut()
            .expect("lazily initialized")
            .relate(subject, relation, object);
    }

    /// Recover the object of `(subject, relation)` from the phasor memory.
    #[cfg(feature = "experimental")]
    pub fn phase_retrieve_object(&self, subject: &str, relation: &str) -> Option<String> {
        self.phase_graph_lazy()
            .as_ref()
            .and_then(|g| g.retrieve_object(subject, relation).map(String::from))
    }

    /// Recover the subject of `(relation, object)` — the inverse query — from the
    /// phasor memory.
    #[cfg(feature = "experimental")]
    pub fn phase_retrieve_subject(&self, relation: &str, object: &str) -> Option<String> {
        self.phase_graph_lazy()
            .as_ref()
            .and_then(|g| g.retrieve_subject(relation, object).map(String::from))
    }

    /// Multi-hop reasoning over the phasor memory: follow `relations` from
    /// `start`, retrieving at each hop, returning the final entity.
    #[cfg(feature = "experimental")]
    pub fn phase_retrieve_path(&self, start: &str, relations: &[&str]) -> Option<String> {
        self.phase_graph_lazy()
            .as_ref()
            .and_then(|g| g.retrieve_path(start, relations))
    }
}

#[cfg(all(test, feature = "experimental"))]
mod connection_graph_tests {
    use super::*;

    #[test]
    fn relate_and_query_through_engine() {
        let dir = tempfile::tempdir().unwrap();
        let hms = HmsCore::new(4096, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

        // Query before any relation exists -> graph unused -> 0.0.
        assert_eq!(hms.query_relation("paris", "capital_of", "france"), 0.0);

        hms.relate("paris", "capital_of", "france");
        let present = hms.query_relation("paris", "capital_of", "france");
        let absent = hms.query_relation("berlin", "capital_of", "spain");
        assert!(
            present > absent,
            "asserted relation ({present}) must score above an absent one ({absent})"
        );
    }

    #[test]
    fn connection_graph_survives_restart() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_string_lossy().to_string();

        {
            let hms = HmsCore::new(4096, Some(path.clone()), None).unwrap();
            hms.relate("paris", "capital_of", "france");
        } // engine dropped -> connection graph's BufWriter flushed on drop

        // New engine over the same storage: the relation must reload from the log.
        let hms2 = HmsCore::new(4096, Some(path), None).unwrap();
        let present = hms2.query_relation("paris", "capital_of", "france");
        let absent = hms2.query_relation("madrid", "capital_of", "italy");
        assert!(
            present > absent,
            "relation must survive restart: reloaded ({present}) !> absent ({absent})"
        );
    }

    #[test]
    fn phase_graph_relate_and_retrieve() {
        let dir = tempfile::tempdir().unwrap();
        let hms = HmsCore::new(4096, Some(dir.path().to_string_lossy().to_string()), None).unwrap();
        hms.relate_phase("paris", "capital_of", "france");
        hms.relate_phase("berlin", "capital_of", "germany");
        // forward retrieval and inverse (subject) retrieval from the same field.
        assert_eq!(
            hms.phase_retrieve_object("paris", "capital_of").as_deref(),
            Some("france")
        );
        assert_eq!(
            hms.phase_retrieve_subject("capital_of", "germany")
                .as_deref(),
            Some("berlin")
        );
    }
}
