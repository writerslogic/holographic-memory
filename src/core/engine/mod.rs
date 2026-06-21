// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod concepts;
pub(crate) mod knowledge;
pub(crate) mod multi_hop;
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
use super::text::TextProcessor;
use super::triple_store::TripleStore;
use super::types::{GraphPath, Relation, RelationType, TextMetrics};

use shard::{ShardManager, ShardSet};

type SignFn<'a> = Box<dyn Fn(&[u8]) -> super::audit::SignatureBytes + 'a>;

/// Lock ordering: ShardSet (read) -> Shard.vectors -> Shard.ivf -> Shard.nsg.
/// Arena lock is independent (managed internally by PersistentArena).
pub struct HmsCore {
    config: HmsConfig,
    pub(crate) arena: Arc<PersistentArena>,
    pub(crate) dimensions: usize,
    pub(crate) storage_path: PathBuf,
    shards: RwLock<ShardSet>,
    graph: RelationStore,
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
    #[cfg(feature = "provenance")]
    provenance: Option<super::provenance::ProvenanceManager>,
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

        let base_path = storage_path
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(".").to_path_buf());
        if !base_path.exists() {
            std::fs::create_dir_all(&base_path)?;
        }

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
            let passphrase = config
                .security
                .encryption_passphrase
                .as_deref()
                .ok_or_else(|| {
                    anyhow::anyhow!("encryption_passphrase required when encryption is enabled")
                })?;
            Some(super::security::EncryptionManager::new(
                passphrase, &base_path,
            )?)
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

        let core = Self {
            config: config.clone(),
            arena,
            dimensions: dim,
            storage_path: base_path,
            shards: RwLock::new(shard_set),
            graph: RelationStore::new(),
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
            #[cfg(feature = "provenance")]
            provenance,
        };

        core.load_from_log()?;
        core.load_indices()?;
        {
            let shards = core.shards.read();
            shards.try_for_each_shard(|s| s.rebuild_inverted_index(dim))?;
        }

        Ok(core)
    }

    fn load_indices(&self) -> Result<()> {
        let nsg_path = self.storage_path.join("nsg_index.bin");
        if nsg_path.exists() {
            let raw = std::fs::read(&nsg_path)?;
            let data = self.maybe_decrypt(&raw)?;
            let nsg: super::nsg::NSGIndex = bincode::deserialize(&data)?;
            // Load NSG into the first (or only) shard
            let shards = self.shards.read();
            if let ShardSet::Single(ref shard) = *shards {
                *shard.nsg.write() = Some(nsg);
            }
        }

        let ivf_path = self.storage_path.join("ivf_index.bin");
        if ivf_path.exists() {
            let raw = std::fs::read(&ivf_path)?;
            let data = self.maybe_decrypt(&raw)?;
            let mut ivf: IVFIndex = bincode::deserialize(&data)?;
            ivf.lists = Some(super::ivf::inverted_list::InvertedLists::new());

            let shards = self.shards.read();
            if let ShardSet::Single(ref shard) = *shards {
                let vectors = shard.vectors.read();
                let registry = shard.registry.read();
                for id in registry.iter() {
                    if let Some(vec) = vectors.get(id) {
                        ivf.insert(id, vec)?;
                    }
                }
                *shard.ivf.write() = Some(ivf);
            }
        }

        Ok(())
    }

    fn save_nsg(&self, nsg: &super::nsg::NSGIndex) -> Result<()> {
        let data = bincode::serialize(nsg)?;
        std::fs::write(
            self.storage_path.join("nsg_index.bin"),
            self.maybe_encrypt(&data)?,
        )?;
        Ok(())
    }

    fn save_ivf(&self, ivf: &IVFIndex) -> Result<()> {
        let data = bincode::serialize(ivf)?;
        std::fs::write(
            self.storage_path.join("ivf_index.bin"),
            self.maybe_encrypt(&data)?,
        )?;
        Ok(())
    }

    /// Bundle vectors respecting the PrivacyConfig.
    /// When dp_enabled, uses Laplace noise with the configured epsilon.
    pub fn bundle<V: std::borrow::Borrow<EntangledHVec>>(&self, vectors: &[V]) -> EntangledHVec {
        if self.config.privacy.dp_enabled {
            EntangledHVec::bundle_dp(vectors, self.config.privacy.epsilon)
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
        let shards = self.shards.read();

        let mut offset = 0;
        while let Ok((payload, _version)) = self.arena_read_frame(offset) {
            if let Some((id, vec)) =
                super::atom_memory::AtomMemory::deserialize_atom(&payload, self.dimensions)
            {
                if let Some(ref atom_mem) = self.atom_memory {
                    atom_mem.load_atom(id, vec);
                }
            } else if let Some((id, vec)) =
                super::composite_memory::CompositeMemory::deserialize_composite(
                    &payload,
                    self.dimensions,
                )
            {
                if let Some(ref comp_mem) = self.composite_memory {
                    comp_mem.load_composite(id, vec);
                }
            } else if let Some(record) =
                super::triple_store::TripleStore::deserialize_triple(&payload)
            {
                if let Some(ref tri_store) = self.triple_store {
                    tri_store.load_triple(record);
                }
            } else if let Some(rule) = super::rules::RuleStore::deserialize_rule(&payload) {
                if let Some(ref rule_store) = self.rule_store {
                    rule_store.load_rule(rule);
                }
            } else if let Some(rel) = RelationStore::deserialize_relation(&payload) {
                self.graph.load_relation(&rel);
            } else {
                let (id, vector) = Self::parse_log_payload(self.dimensions, &payload);
                if vector.dim == 0 {
                    shards.remove(&id, self.dimensions)?;
                } else {
                    shards.insert(id, vector, self.dimensions)?;
                }
            }
            offset = match self.arena.next_offset(offset) {
                Ok(next) => next,
                Err(_) => break,
            };
        }

        // Rebuild registries from live vectors (ensure deterministic ordering)
        shards.for_each_shard(|shard| {
            let vectors = shard.vectors.read();
            let mut reg = shard.registry.write();
            let mut live_ids: Vec<String> = vectors.keys().cloned().collect();
            live_ids.sort();
            *reg = live_ids;
            shard
                .vector_count
                .store(reg.len() as u64, std::sync::atomic::Ordering::SeqCst);
        });

        if let Some(ref atom_mem) = self.atom_memory {
            atom_mem.rebuild_indices();
        }
        if let Some(ref comp_mem) = self.composite_memory {
            comp_mem.rebuild_indices();
        }

        Ok(())
    }

    fn parse_log_payload(dimensions: usize, payload: &[u8]) -> (String, EntangledHVec) {
        if payload.len() < 6 {
            return (String::new(), EntangledHVec::from_indices(vec![], 0));
        }
        let id_len = u16::from_le_bytes([payload[0], payload[1]]) as usize;
        let id_end = 2 + id_len;
        if payload.len() < id_end + 4 {
            return (String::new(), EntangledHVec::from_indices(vec![], 0));
        }
        let id = match std::str::from_utf8(&payload[2..id_end]) {
            Ok(s) => s.to_owned(),
            Err(_) => String::from_utf8_lossy(&payload[2..id_end]).into_owned(),
        };
        let delta_count_raw = u32::from_le_bytes(match payload[id_end..id_end + 4].try_into() {
            Ok(b) => b,
            Err(_) => return (id, EntangledHVec::from_indices(vec![], 0)),
        });

        if delta_count_raw == Self::TOMBSTONE_MARKER {
            return (id, EntangledHVec::from_indices(vec![], 0));
        }

        let delta_count = delta_count_raw as usize;
        if delta_count == 0 {
            return (id, EntangledHVec::from_indices(vec![], dimensions));
        }

        let deltas_start = id_end + 4;
        let deltas_end = deltas_start + delta_count * 4;
        if payload.len() < deltas_end {
            return (id, EntangledHVec::from_indices(vec![], 0));
        }
        let deltas: Vec<u32> = payload[deltas_start..deltas_end]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().expect("chunks_exact(4)")))
            .collect();
        (id, EntangledHVec::from_deltas(&deltas, dimensions))
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

    fn serialize_log_entry(id: &str, vector: &EntangledHVec) -> Result<Vec<u8>> {
        let id_bytes = id.as_bytes();
        if id_bytes.len() > u16::MAX as usize {
            return Err(anyhow::anyhow!(
                "ID too long: {} bytes (max {})",
                id_bytes.len(),
                u16::MAX
            ));
        }
        let deltas = vector.to_deltas();
        let mut entry = Vec::with_capacity(2 + id_bytes.len() + 4 + deltas.len() * 4);
        entry.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
        entry.extend_from_slice(id_bytes);
        entry.extend_from_slice(&(deltas.len() as u32).to_le_bytes());
        for &d in &deltas {
            entry.extend_from_slice(&d.to_le_bytes());
        }
        Ok(entry)
    }

    const TOMBSTONE_MARKER: u32 = u32::MAX;

    fn serialize_tombstone(id: &str) -> Result<Vec<u8>> {
        let id_bytes = id.as_bytes();
        if id_bytes.len() > u16::MAX as usize {
            return Err(anyhow::anyhow!(
                "ID too long: {} bytes (max {})",
                id_bytes.len(),
                u16::MAX
            ));
        }
        let mut entry = Vec::with_capacity(2 + id_bytes.len() + 4);
        entry.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
        entry.extend_from_slice(id_bytes);
        entry.extend_from_slice(&Self::TOMBSTONE_MARKER.to_le_bytes());
        Ok(entry)
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
        self.arena_write(&Self::serialize_tombstone(id)?)?;

        if let Some(ref audit) = self.audit {
            audit.record(AuditOp::Delete, id, self.sign_fn().as_deref())?;
        }

        #[cfg(feature = "provenance")]
        if let Some(ref mgr) = self.provenance {
            if self.config.provenance.auto_sign {
                if let Err(e) = mgr.record_deletion(id, reason) {
                    tracing::warn!("provenance deletion record failed for {id}: {e}");
                }
            }
        }

        if let Some(ref atom_mem) = self.atom_memory {
            atom_mem.delete(id);
        }

        let shards = self.shards.read();
        if !shards.remove(id, self.dimensions)? {
            return Ok(false);
        }
        Ok(true)
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
        let vector = self.encode_text(text);
        self.memorize(id.to_string(), vector)?;

        #[cfg(feature = "provenance")]
        if let Some(ref mgr) = self.provenance {
            if self.config.provenance.auto_sign {
                if let Err(e) = mgr.create_fact_provenance(id, text.as_bytes(), source_uri) {
                    tracing::warn!("provenance signing failed for {id}: {e}");
                }
            }
        }

        if let (
            Some(ref decomposer),
            Some(ref atom_mem),
            Some(ref comp_mem),
            Some(ref tri_store),
            Some(ref roles),
        ) = (
            &self.decomposer,
            &self.atom_memory,
            &self.composite_memory,
            &self.triple_store,
            &self.role_registry,
        ) {
            if self.config.meaning.auto_decompose {
                let units = decomposer.decompose(text);
                for unit in &units {
                    let (_, s_vec) = atom_mem.get_or_insert(&unit.subject);
                    let (_, r_vec) = atom_mem.get_or_insert(&unit.relation);
                    let (_, o_vec) = atom_mem.get_or_insert(&unit.object);

                    self.arena_write(&super::atom_memory::AtomMemory::serialize_atom(
                        &unit.subject,
                        &s_vec,
                    ))?;
                    self.arena_write(&super::atom_memory::AtomMemory::serialize_atom(
                        &unit.relation,
                        &r_vec,
                    ))?;
                    self.arena_write(&super::atom_memory::AtomMemory::serialize_atom(
                        &unit.object,
                        &o_vec,
                    ))?;

                    let composite = roles.compose_triple(&s_vec, &r_vec, &o_vec);
                    let comp_id =
                        format!("{}:{}:{}:{}", id, unit.subject, unit.relation, unit.object);
                    comp_mem.insert(comp_id.clone(), composite.clone());

                    self.arena_write(
                        &super::composite_memory::CompositeMemory::serialize_composite(
                            &comp_id, &composite,
                        ),
                    )?;

                    tri_store.add(&unit.subject, &unit.relation, &unit.object, &comp_id);
                    self.arena_write(&super::triple_store::TripleStore::serialize_triple(
                        &super::triple_store::TripleRecord {
                            subject_id: unit.subject.clone(),
                            relation_id: unit.relation.clone(),
                            object_id: unit.object.clone(),
                            composite_id: comp_id.clone(),
                            deleted: false,
                        },
                    ))?;

                    #[cfg(feature = "provenance")]
                    if let Some(ref mgr) = self.provenance {
                        if self.config.provenance.auto_sign {
                            if let Err(e) = mgr.create_triple_provenance(
                                &super::provenance::TripleProvenanceParams {
                                    fact_id: &comp_id,
                                    content: text.as_bytes(),
                                    dimensions: self.dimensions as u32,
                                    subject: &unit.subject,
                                    relation: &unit.relation,
                                    object: &unit.object,
                                    source_uri,
                                },
                            ) {
                                tracing::warn!("provenance signing failed for {comp_id}: {e}");
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Compact the arena log by rewriting only live vectors. Blocks all writes during compaction.
    pub fn compact(&self) -> Result<()> {
        // Hold the shards write lock for the entire compaction to block
        // concurrent memorize/delete. This guarantees the snapshot is
        // consistent with what gets swapped in.
        let shards = self.shards.write();

        let mut snapshot = Vec::new();
        shards.for_each_shard(|shard| {
            let vectors = shard.vectors.read();
            let registry = shard.registry.read();
            for id in registry.iter() {
                if let Some(v) = vectors.get(id) {
                    snapshot.push((id.clone(), v.clone()));
                }
            }
        });

        let temp_dir = self.storage_path.join(format!(
            ".compact_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros()
        ));

        let relation_snapshot = self.graph.snapshot();

        {
            let temp_arena = PersistentArena::new(&temp_dir)?;
            for (id, vector) in &snapshot {
                let entry = Self::serialize_log_entry(id, vector)?;
                let payload = self.maybe_encrypt(&entry)?;
                temp_arena.write_slice(&payload)?;
            }
            for rel in &relation_snapshot {
                let entry = RelationStore::serialize_relation(rel);
                let payload = self.maybe_encrypt(&entry)?;
                temp_arena.write_slice(&payload)?;
            }
            if let Some(ref atom_mem) = self.atom_memory {
                for (_, id, vec) in atom_mem.inner().all_vectors() {
                    let entry = super::atom_memory::AtomMemory::serialize_atom(&id, &vec);
                    let payload = self.maybe_encrypt(&entry)?;
                    temp_arena.write_slice(&payload)?;
                }
            }
            if let Some(ref comp_mem) = self.composite_memory {
                for (_, id, vec) in comp_mem.inner().all_vectors() {
                    let entry =
                        super::composite_memory::CompositeMemory::serialize_composite(&id, &vec);
                    let payload = self.maybe_encrypt(&entry)?;
                    temp_arena.write_slice(&payload)?;
                }
            }
            if let Some(ref tri_store) = self.triple_store {
                for record in tri_store.snapshot() {
                    let entry = super::triple_store::TripleStore::serialize_triple(&record);
                    let payload = self.maybe_encrypt(&entry)?;
                    temp_arena.write_slice(&payload)?;
                }
            }
            if let Some(ref rule_store) = self.rule_store {
                for rule in rule_store.all_rules() {
                    let entry = super::rules::RuleStore::serialize_rule(&rule);
                    let payload = self.maybe_encrypt(&entry)?;
                    temp_arena.write_slice(&payload)?;
                }
            }
        }

        self.arena.replace_with_compacted(&temp_dir)?;

        if let Some(ref audit) = self.audit {
            audit.record(AuditOp::Compact, "", self.sign_fn().as_deref())?;
        }

        if let ShardSet::Single(ref shard) = *shards {
            if let Some(ref nsg) = *shard.nsg.read() {
                self.save_nsg(nsg)?;
            }
            if let Some(ref ivf) = *shard.ivf.read() {
                self.save_ivf(ivf)?;
            }
        }

        Ok(())
    }

    /// Store a vector with the given ID. Persists to the arena log and updates all indices.
    pub fn memorize(&self, id: String, vector: EntangledHVec) -> Result<()> {
        let entry = Self::serialize_log_entry(&id, &vector)?;
        self.arena_write(&entry)?;

        if let Some(ref audit) = self.audit {
            audit.record(AuditOp::Memorize, &id, self.sign_fn().as_deref())?;
        }

        if let Some(ref atom_mem) = self.atom_memory {
            atom_mem.insert_with_vec(&id, &vector);
        }

        let count = {
            let shards = self.shards.read();
            shards.insert(id, vector, self.dimensions)?;
            shards.count()
        };

        if self.config.ivf.enabled
            && self.config.ivf.auto_threshold > 0
            && count == self.config.ivf.auto_threshold as u64
        {
            self.train_ivf().context("Auto-train IVF failed")?;
        } else if self.config.nsg.auto_threshold > 0
            && count == self.config.nsg.auto_threshold as u64
        {
            self.train_nsg().context("Auto-train NSG failed")?;
        } else {
            self.maybe_auto_shard(count);
        }

        Ok(())
    }

    fn maybe_auto_shard(&self, count: u64) {
        let cfg = &self.config.shard;
        if !cfg.enabled || cfg.shard_count > 0 || cfg.auto_threshold == 0 {
            return;
        }
        if count < cfg.auto_threshold as u64 {
            return;
        }

        // Upgrade from Single to Multi: snapshot vectors, build new shards, swap.
        let mut shards = self.shards.write();
        let snapshot: Vec<(String, EntangledHVec)> = {
            match *shards {
                ShardSet::Single(ref old_shard) => {
                    let vectors = old_shard.vectors.read();
                    vectors
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect()
                }
                ShardSet::Multi(_) => return,
            }
        };

        let n_shards = (count as usize / cfg.target_shard_size).max(2);
        let mgr = ShardManager::new(n_shards, self.dimensions);
        for (id, vec) in snapshot {
            let target = mgr.shard_for(&id);
            let shard = &mgr.shards[target];
            shard.vectors.write().insert(id.clone(), vec.clone());
            shard.registry.write().push(id);
        }
        for shard in &mgr.shards {
            let count = shard.vectors.read().len() as u64;
            shard
                .vector_count
                .store(count, std::sync::atomic::Ordering::SeqCst);
            let _ = shard.rebuild_inverted_index(self.dimensions);
        }

        *shards = ShardSet::Multi(mgr);
    }

    /// Convert a dense f32 vector to sparse and memorize it.
    pub fn memorize_vector(&self, id: String, dense: &[f32]) -> Result<()> {
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
        let entry = RelationStore::serialize_relation(rel);
        self.arena_write(&entry)?;
        self.graph.add(rel);
        if let Some(ref audit) = self.audit {
            let label = format!("{}->{}:{}", rel.source_id, rel.target_id, rel.relation_type);
            audit.record(AuditOp::Memorize, &label, self.sign_fn().as_deref())?;
        }
        Ok(())
    }

    pub fn remove_relation(&self, source_id: &str, relation_type: &str, target_id: &str) -> bool {
        self.graph.remove(source_id, relation_type, target_id)
    }

    pub fn declare_relation_type(&self, rel_type: RelationType) {
        self.graph.declare_type(rel_type);
    }

    pub fn traverse(
        &self,
        start_id: &str,
        relation_type: Option<&str>,
        max_depth: u32,
        at_time: f64,
    ) -> Vec<GraphPath> {
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
        self.graph.outgoing(source_id, relation_type, at_time)
    }

    pub fn incoming_relations(
        &self,
        target_id: &str,
        relation_type: Option<&str>,
        at_time: f64,
    ) -> Vec<Relation> {
        self.graph.incoming(target_id, relation_type, at_time)
    }

    pub fn relation_count(&self) -> usize {
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

    pub fn structural_query(
        &self,
        known: &[(&str, &EntangledHVec)],
        target_role: &str,
    ) -> Vec<structural::StructuralResult> {
        let (atom_mem, comp_mem, tri, roles, adm) = match (
            &self.atom_memory,
            &self.composite_memory,
            &self.triple_store,
            &self.role_registry,
            &self.admission,
        ) {
            (Some(a), Some(c), Some(t), Some(r), Some(ad)) => (a, c, t, r, ad),
            _ => return Vec::new(),
        };
        let mc = &self.config.meaning;
        let ctx = structural::MeaningContext {
            atom_memory: atom_mem,
            composite_memory: comp_mem,
            triple_store: tri,
            roles,
            admission: adm,
            beta: mc.beta,
            k: 64,
            max_iter: 3,
        };
        structural::fuzzy_structural_query(&ctx, known, target_role)
    }

    pub fn multi_hop(&self, start: &str, relations: &[&str]) -> Vec<multi_hop::MultiHopResult> {
        let (atom_mem, comp_mem, tri, roles, adm, rules) = match (
            &self.atom_memory,
            &self.composite_memory,
            &self.triple_store,
            &self.role_registry,
            &self.admission,
            &self.rule_store,
        ) {
            (Some(a), Some(c), Some(t), Some(r), Some(ad), Some(ru)) => (a, c, t, r, ad, ru),
            _ => return Vec::new(),
        };
        let mc = &self.config.meaning;
        let ctx = structural::MeaningContext {
            atom_memory: atom_mem,
            composite_memory: comp_mem,
            triple_store: tri,
            roles,
            admission: adm,
            beta: mc.beta,
            k: 64,
            max_iter: 3,
        };
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

    pub fn declare_rule(&self, name: &str, input_relations: Vec<String>, output_relation: String) {
        if let Some(ref rules) = self.rule_store {
            rules.add_rule(super::rules::CompositionRule {
                name: name.to_string(),
                input_relations,
                output_relation,
            });
        }
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
        let shards = self.shards.read();
        shards.try_for_each_shard(|shard| {
            let (ids, vectors) = shard.load_all_vectors();
            if ids.is_empty() {
                return Ok(());
            }

            if let Some(ref mut existing) = *shard.ivf.write() {
                if let Some(ref lists) = existing.lists {
                    lists.clear_all()?;
                }
            }

            let index = IVFIndex::train(&vectors, &ids, self.dimensions, &self.config.ivf)?;
            *shard.ivf.write() = Some(index);
            Ok(())
        })?;

        if let ShardSet::Single(ref shard) = *shards {
            if let Some(ref ivf) = *shard.ivf.read() {
                self.save_ivf(ivf)?;
            }
        }
        Ok(())
    }

    /// Returns true if the NSG graph index has been trained.
    pub fn nsg_trained(&self) -> bool {
        self.shards.read().nsg_trained()
    }

    /// Train the NSG graph index on current vectors. Persists the index to disk.
    pub fn train_nsg(&self) -> Result<()> {
        let shards = self.shards.read();
        shards.try_for_each_shard(|shard| {
            let (ids, vectors) = shard.load_all_vectors();
            if ids.is_empty() {
                return Ok(());
            }

            let index = super::nsg::training::train(&vectors, &ids, &self.config.nsg)?;
            *shard.nsg.write() = Some(index);
            Ok(())
        })?;

        if let ShardSet::Single(ref shard) = *shards {
            if let Some(ref nsg) = *shard.nsg.read() {
                self.save_nsg(nsg)?;
            }
        }
        Ok(())
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
    ) -> Result<super::provenance::types::VerificationResult> {
        let mgr = self
            .provenance
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("provenance not enabled"))?;
        mgr.verify_fact_provenance(record)
    }

    #[cfg(feature = "provenance")]
    pub fn verify_store_manifest(
        &self,
        manifest: &super::provenance::types::StoreManifest,
        store_data: Option<&[u8]>,
    ) -> Result<super::provenance::types::VerificationResult> {
        let mgr = self
            .provenance
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("provenance not enabled"))?;
        mgr.verify_store_manifest(manifest, store_data)
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
    ) -> Result<()> {
        let mgr = self
            .provenance
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("provenance not enabled"))?;
        mgr.verify_sigstore_bundle(bundle, content)
    }

    #[cfg(feature = "provenance")]
    pub fn create_cawg_assertion(
        &self,
        referenced_assertions: Vec<super::provenance::cawg::HashedUri>,
        display_name: Option<&str>,
        provider: Option<&str>,
    ) -> Result<super::provenance::cawg::IdentityAssertion> {
        let mgr = self
            .provenance
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("provenance not enabled"))?;
        mgr.create_cawg_assertion(referenced_assertions, display_name, provider)
    }

    #[cfg(feature = "provenance")]
    pub fn verify_cawg_assertion(
        &self,
        assertion: &super::provenance::cawg::IdentityAssertion,
    ) -> Result<()> {
        let mgr = self
            .provenance
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("provenance not enabled"))?;
        mgr.verify_cawg_assertion(assertion)
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
}
