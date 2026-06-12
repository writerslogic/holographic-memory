pub(crate) mod concepts;
pub(crate) mod knowledge;
pub(crate) mod query;
pub(crate) mod router;

use anyhow::{Context, Result};
use parking_lot::RwLock;
use fxhash::FxHashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;

use super::config::HmsConfig;
use super::diffusion::{DiffusionConfig, DiffusionFactorizer};
use super::encoding::encode_text_internal;
use super::entangled::EntangledHVec;
use super::index::inverted::{Accumulator, SparseInvertedIndex};
use super::ivf::IVFIndex;
use super::nsg::NSGIndex;
use super::storage::PersistentArena;
use super::text::TextProcessor;
use super::types::TextMetrics;

/// Lock ordering: vectors -> ivf -> nsg. All code acquiring multiple locks must follow this order.
pub struct HmsCore {
    /// Global configuration for indexing and sharding.
    config: HmsConfig,
    /// Persistent append-only arena for storing raw vectors.
    pub(crate) arena: Arc<PersistentArena>,
    /// Number of dimensions in the hyperdimensional space.
    pub(crate) dimensions: usize,
    /// Path to the storage directory.
    pub(crate) storage_path: PathBuf,
    /// Inverted File (IVF) index for coarse-grained quantization.
    ivf: Arc<RwLock<Option<IVFIndex>>>,
    /// Navigable Small World (NSG) graph for fast proximity search.
    nsg: Arc<RwLock<Option<NSGIndex>>>,
    /// Sparse Inverted Index for high-sparsity term-based retrieval.
    pub(crate) inverted: Arc<RwLock<SparseInvertedIndex>>,
    /// Thread-local accumulator for score aggregation during inverted index queries.
    pub(crate) accumulator: Arc<parking_lot::Mutex<Accumulator>>,
    /// Atomic counter for the total number of non-deleted vectors.
    vector_count: AtomicU64,
    /// Absolute source of truth in RAM: ID -> Vector
    pub(crate) vectors: Arc<RwLock<FxHashMap<String, EntangledHVec>>>,
    /// Maps internal doc_id (index) to String ID.
    pub(crate) registry: Arc<RwLock<Vec<String>>>,
}

impl HmsCore {
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

        // Calculate m = D/256
        let m = (dim / 256).max(1);

        let core = Self {
            config: config.clone(),
            arena,
            dimensions: dim,
            storage_path: base_path,
            ivf: Arc::new(RwLock::new(None)),
            nsg: Arc::new(RwLock::new(None)),
            inverted: Arc::new(RwLock::new(SparseInvertedIndex::new(dim, m))),
            accumulator: Arc::new(parking_lot::Mutex::new(Accumulator::new(1024))),
            vector_count: AtomicU64::new(0),
            vectors: Arc::new(RwLock::new(FxHashMap::default())),
            registry: Arc::new(RwLock::new(Vec::new())),
        };

        core.load_from_log()?;
        core.load_indices()?; // Load persisted NSG/IVF from bin files
        core.rebuild_inverted_index()?;

        Ok(core)
    }

    fn rebuild_inverted_index(&self) -> Result<()> {
        let vectors = self.vectors.read();
        let registry = self.registry.read();
        let mut inv = self.inverted.write();

        // Reset index
        *inv = SparseInvertedIndex::new(self.dimensions, (self.dimensions / 256).max(1));

        for (i, id) in registry.iter().enumerate() {
            if let Some(vec) = vectors.get(id) {
                inv.add_doc(i as u32, &vec.indices);
            }
        }
        inv.finalize();

        // Also resize accumulator to match registry size
        let mut acc = self.accumulator.lock();
        *acc = Accumulator::new(registry.len().max(1024));

        Ok(())
    }

    fn load_indices(&self) -> Result<()> {
        // 1. Load NSG if exists
        let nsg_path = self.storage_path.join("nsg_index.bin");
        if nsg_path.exists() {
            let data = std::fs::read(&nsg_path)?;
            let nsg: NSGIndex = bincode::deserialize(&data)?;
            *self.nsg.write() = Some(nsg);
        }

        // 2. Load IVF if exists
        let ivf_path = self.storage_path.join("ivf_index.bin");
        if ivf_path.exists() {
            let data = std::fs::read(&ivf_path)?;
            let mut ivf: IVFIndex = bincode::deserialize(&data)?;
            // In-memory inverted lists don't need a DB handle
            ivf.lists = Some(super::ivf::inverted_list::InvertedLists::new());
            // Re-fill inverted lists from vectors
            let vectors = self.vectors.read();
            let registry = self.registry.read();
            for id in registry.iter() {
                if let Some(vec) = vectors.get(id) {
                    ivf.insert(id, vec)?;
                }
            }
            *self.ivf.write() = Some(ivf);
        }

        Ok(())
    }

    fn save_nsg(&self, nsg: &NSGIndex) -> Result<()> {
        let data = bincode::serialize(nsg)?;
        std::fs::write(self.storage_path.join("nsg_index.bin"), data)?;
        Ok(())
    }

    fn save_ivf(&self, ivf: &IVFIndex) -> Result<()> {
        let data = bincode::serialize(ivf)?;
        std::fs::write(self.storage_path.join("ivf_index.bin"), data)?;
        Ok(())
    }

    fn load_from_log(&self) -> Result<()> {
        let mut vectors = self.vectors.write();
        let mut registry = self.registry.write();

        let mut offset = 0;
        while let Ok((payload, _version)) = self.arena.read_frame(offset) {
            let (id, vector) = self.parse_log_payload(&payload);
            if vector.dim == 0 {
                vectors.remove(&id);
            } else {
                vectors.insert(id.clone(), vector);
            }
            offset = match self.arena.next_offset(offset) {
                Ok(next) => next,
                Err(_) => break,
            };
        }

        // Rebuild registry from live vectors.
        // For NSG/IVF indices, the order in registry must be stable.
        let mut live_ids: Vec<String> = vectors.keys().cloned().collect();
        live_ids.sort(); // Ensure deterministic internal indexing
        *registry = live_ids;
        self.vector_count.store(registry.len() as u64, AtomicOrdering::SeqCst);
        Ok(())
    }

    fn parse_log_payload(&self, payload: &[u8]) -> (String, EntangledHVec) {
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
            return (id, EntangledHVec::from_indices(vec![], self.dimensions));
        }

        let deltas_start = id_end + 4;
        let deltas_end = deltas_start + delta_count * 4;
        if payload.len() < deltas_end {
            return (id, EntangledHVec::from_indices(vec![], 0));
        }
        let deltas: Vec<u32> = payload[deltas_start..deltas_end]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (id, EntangledHVec::from_deltas(&deltas, self.dimensions))
    }

    /// Load all vectors and their associated metadata.
    fn load_all_vectors(&self) -> (Vec<String>, Vec<EntangledHVec>) {
        let vectors = self.vectors.read();
        let registry = self.registry.read();
        let mut ids = Vec::with_capacity(registry.len());
        let mut out_vectors = Vec::with_capacity(registry.len());

        for id in registry.iter() {
            if let Some(vec) = vectors.get(id) {
                ids.push(id.clone());
                out_vectors.push(vec.clone());
            }
        }

        (ids, out_vectors)
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn encode_text(&self, text: &str) -> EntangledHVec {
        encode_text_internal(text, self.dimensions)
    }

    pub fn analyze_text(&self, text: &str) -> TextMetrics {
        TextProcessor::analyze(text)
    }

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

    fn serialize_tombstone(id: &str) -> Vec<u8> {
        let id_bytes = id.as_bytes();
        let mut entry = Vec::with_capacity(2 + id_bytes.len() + 4);
        entry.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
        entry.extend_from_slice(id_bytes);
        entry.extend_from_slice(&Self::TOMBSTONE_MARKER.to_le_bytes());
        entry
    }

    pub fn delete(&self, id: &str) -> Result<bool> {
        let mut vectors = self.vectors.write();
        let mut reg = self.registry.write();

        if vectors.remove(id).is_none() {
            return Ok(false);
        }

        // Persist tombstone before updating in-memory state further.
        // If this fails, we've already removed from memory — but the
        // next load_from_log will still see the original entry. Accept
        // this minor inconsistency (the entry reappears after crash).
        self.arena.write_slice(&Self::serialize_tombstone(id))?;

        reg.retain(|r| r != id);
        self.vector_count
            .store(reg.len() as u64, AtomicOrdering::SeqCst);

        // Drop write locks before rebuilding inverted index
        drop(reg);
        drop(vectors);

        self.rebuild_inverted_index()?;
        Ok(true)
    }

    pub fn compact(&self) -> Result<()> {
        // Snapshot live vectors under read lock
        let snapshot: Vec<(String, EntangledHVec)> = {
            let vectors = self.vectors.read();
            let registry = self.registry.read();
            registry
                .iter()
                .filter_map(|id| vectors.get(id).map(|v| (id.clone(), v.clone())))
                .collect()
        };

        let temp_dir = self.storage_path.join(format!(
            ".compact_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros()
        ));

        // Write all live vectors to a fresh arena
        {
            let temp_arena = PersistentArena::new(&temp_dir)?;
            for (id, vector) in &snapshot {
                let entry = Self::serialize_log_entry(id, vector)?;
                temp_arena.write_slice(&entry)?;
            }
        } // temp_arena dropped here, releasing file handles

        // Atomically swap arena files
        self.arena.replace_with_compacted(&temp_dir)?;

        // Persist indices to the (now clean) storage path
        if let Some(ref nsg) = *self.nsg.read() {
            self.save_nsg(nsg)?;
        }
        if let Some(ref ivf) = *self.ivf.read() {
            self.save_ivf(ivf)?;
        }

        Ok(())
    }

    pub fn memorize(&self, id: String, vector: EntangledHVec) -> Result<()> {
        let mut vectors = self.vectors.write();
        let mut reg = self.registry.write();

        let entry = Self::serialize_log_entry(&id, &vector)?;
        self.arena.write_slice(&entry)?;

        let is_replacement = vectors.contains_key(&id);
        vectors.insert(id.clone(), vector.clone());

        if !is_replacement {
            reg.push(id.clone());
            let count = reg.len() as u64;
            self.vector_count.store(count, AtomicOrdering::SeqCst);

            let mut inv = self.inverted.write();
            inv.add_doc((count - 1) as u32, &vector.indices);
        } else {
            self.rebuild_inverted_index()?;
        }

        let count = reg.len() as u64;

        if let Some(ref mut ivf) = *self.ivf.write() {
            ivf.insert(&id, &vector)?;
        }

        if let Some(ref mut nsg) = *self.nsg.write() {
            nsg.insert(&id, &vector)?;
        }

        if self.config.ivf.enabled
            && self.config.ivf.auto_threshold > 0
            && count == self.config.ivf.auto_threshold as u64
        {
            self.train_ivf().context("Auto-train IVF failed")?;
        }

        if self.config.nsg.auto_threshold > 0 && count == self.config.nsg.auto_threshold as u64 {
            self.train_nsg().context("Auto-train NSG failed")?;
        }

        Ok(())
    }

    pub fn memorize_vector(&self, id: String, dense: &[f32]) -> Result<()> {
        let vector = EntangledHVec::from_dense(dense, self.dimensions);
        self.memorize(id, vector)
    }

    pub fn memorize_scalar(&self, id: String, value: f64, min: f64, max: f64) -> Result<()> {
        let vector = EntangledHVec::from_scalar(value, min, max, self.dimensions);
        self.memorize(id, vector)
    }

    pub fn vector_count(&self) -> u64 {
        self.vector_count.load(AtomicOrdering::Relaxed)
    }

    pub fn ivf_trained(&self) -> bool {
        self.ivf.read().as_ref().is_some_and(|ivf| ivf.is_trained())
    }

    pub fn train_ivf(&self) -> Result<()> {
        let (ids, vectors) = self.load_all_vectors();
        if ids.is_empty() {
            return Ok(());
        }

        // Clear existing inverted lists if any to prevent corruption
        if let Some(ref mut existing) = *self.ivf.write() {
            if let Some(ref lists) = existing.lists {
                lists.clear_all()?;
            }
        }

        let index = IVFIndex::train(
            &vectors,
            &ids,
            self.dimensions,
            &self.config.ivf,
        )?;

        self.save_ivf(&index)?; // PERSIST
        *self.ivf.write() = Some(index);
        Ok(())
    }

    pub fn nsg_trained(&self) -> bool {
        self.nsg.read().as_ref().is_some_and(|nsg| nsg.is_trained())
    }

    pub fn train_nsg(&self) -> Result<()> {
        let (ids, vectors) = self.load_all_vectors();
        if ids.is_empty() {
            return Ok(());
        }

        let index = super::nsg::training::train(
            &vectors,
            &ids,
            &self.config.nsg,
        )?;

        self.save_nsg(&index)?; // PERSIST
        *self.nsg.write() = Some(index);
        Ok(())
    }

    pub fn factorize_diffusion(
        &self,
        product: &EntangledHVec,
        domain_codebooks: &[Vec<EntangledHVec>],
        max_iter: usize,
    ) -> Vec<Option<EntangledHVec>> {
        let config = DiffusionConfig::default();
        DiffusionFactorizer::factorize(&config, product, domain_codebooks, max_iter)
    }
}
