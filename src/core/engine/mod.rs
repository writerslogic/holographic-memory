pub(crate) mod concepts;
pub(crate) mod knowledge;
pub(crate) mod maintenance;
pub(crate) mod query;
pub(crate) mod router;

use anyhow::{Context, Result};
use parking_lot::RwLock;
use redb::{Database, ReadableTable, TableDefinition};
use fxhash::{FxHashMap, FxHashSet};
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

pub(crate) const REGISTRY_TABLE: TableDefinition<&str, u64> = TableDefinition::new("registry");
pub(crate) const DELETED_IDS_TABLE: TableDefinition<&str, u64> =
    TableDefinition::new("deleted_ids");
pub(crate) const NSG_INDEX_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("nsg_index");
pub(crate) const IVF_INDEX_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("ivf_index");

/// Lock ordering: registry -> ivf -> nsg. All code acquiring multiple locks must follow this order.
pub struct HmsCore {
    /// Global configuration for indexing and sharding.
    config: HmsConfig,
    /// Persistent append-only arena for storing raw vectors.
    pub(crate) arena: Arc<PersistentArena>,
    /// Number of dimensions in the hyperdimensional space.
    pub(crate) dimensions: usize,
    /// Persistent key-value store for registry and index metadata.
    pub(crate) db: Arc<Database>,
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
    /// In-memory registry of ID-to-offset mappings (ordered).
    /// Used for mapping internal doc_id back to String ID.
    registry: Arc<RwLock<Vec<(String, usize)>>>,
    /// Fast lookup from String ID to arena offset.
    id_to_offset: Arc<RwLock<FxHashMap<String, usize>>>,
    /// Set of non-deleted IDs for authoritative filtering.
    live_ids: Arc<RwLock<FxHashSet<String>>>,
}

impl HmsCore {
    pub fn new(
        dimensions: u32,
        storage_path: Option<String>,
        config: Option<HmsConfig>,
    ) -> Result<Self> {
        // Security: bound dimensions to prevent OOM from malicious input.
        // 1M dimensions * 4 bytes/index = 4 MB per vector, which is reasonable.
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

        let db_path = base_path.join("meta.redb");
        let db = Arc::new(Database::create(db_path)?);

        // Ensure tables exist
        let write_tx = db.begin_write()?;
        {
            let _ = write_tx.open_table(REGISTRY_TABLE)?;
            let _ = write_tx.open_table(DELETED_IDS_TABLE)?;
            let _ = write_tx.open_table(NSG_INDEX_TABLE)?;
            let _ = write_tx.open_table(IVF_INDEX_TABLE)?;
        }
        write_tx.commit()?;

        let arena = Arc::new(PersistentArena::new(base_path.join("vectors_data.bin"))?);

        // Calculate m = D/256
        let m = (dim / 256).max(1);

        let core = Self {
            config: config.clone(),
            arena,
            dimensions: dim,
            db: db.clone(),
            ivf: Arc::new(RwLock::new(None)),
            nsg: Arc::new(RwLock::new(None)),
            inverted: Arc::new(RwLock::new(SparseInvertedIndex::new(dim, m))),
            accumulator: Arc::new(parking_lot::Mutex::new(Accumulator::new(1024))),
            vector_count: AtomicU64::new(0),
            registry: Arc::new(RwLock::new(Vec::new())),
            id_to_offset: Arc::new(RwLock::new(FxHashMap::default())),
            live_ids: Arc::new(RwLock::new(FxHashSet::default())),
        };

        core.load_registry()?;
        core.load_indices()?; // Load persisted NSG/IVF
        core.rebuild_inverted_index()?;

        Ok(core)
    }

    fn rebuild_inverted_index(&self) -> Result<()> {
        let (ids, vectors, _) = self.load_all_vectors();
        let mut inv = self.inverted.write();

        // Reset index
        *inv = SparseInvertedIndex::new(self.dimensions, (self.dimensions / 256).max(1));

        for (i, vec) in vectors.iter().enumerate() {
            inv.add_doc(i as u32, &vec.indices);
        }
        inv.finalize();

        // Also resize accumulator to match registry size
        let mut acc = self.accumulator.lock();
        *acc = Accumulator::new(ids.len().max(1024));

        Ok(())
    }

    fn load_indices(&self) -> Result<()> {
        let read_tx = self.db.begin_read()?;

        // 1. Load NSG if exists
        let nsg_table = read_tx.open_table(NSG_INDEX_TABLE)?;
        if let Some(data) = nsg_table.get("main")? {
            let nsg: NSGIndex = bincode::deserialize(data.value())?;
            *self.nsg.write() = Some(nsg);
        }

        // 2. Load IVF if exists
        let ivf_table = read_tx.open_table(IVF_INDEX_TABLE)?;
        if let Some(data) = ivf_table.get("main")? {
            let mut ivf: IVFIndex = bincode::deserialize(data.value())?;
            // Re-attach database handle to inverted lists
            ivf.lists = Some(super::ivf::inverted_list::InvertedLists::new(
                self.db.clone(),
            )?);
            *self.ivf.write() = Some(ivf);
        }

        Ok(())
    }

    fn save_nsg(&self, nsg: &NSGIndex) -> Result<()> {
        let data = bincode::serialize(nsg)?;
        let write_tx = self.db.begin_write()?;
        {
            let mut table = write_tx.open_table(NSG_INDEX_TABLE)?;
            table.insert("main", data.as_slice())?;
        }
        write_tx.commit()?;
        Ok(())
    }

    fn save_ivf(&self, ivf: &IVFIndex) -> Result<()> {
        let data = bincode::serialize(ivf)?;
        let write_tx = self.db.begin_write()?;
        {
            let mut table = write_tx.open_table(IVF_INDEX_TABLE)?;
            table.insert("main", data.as_slice())?;
        }
        write_tx.commit()?;
        Ok(())
    }

    fn load_registry(&self) -> Result<()> {
        let read_tx = self.db.begin_read()?;
        let table = read_tx.open_table(REGISTRY_TABLE)?;

        let mut reg = self.registry.write();
        let mut ito = self.id_to_offset.write();
        let mut live = self.live_ids.write();
        let mut count = 0;

        for item in table.iter()? {
            let (id, offset) = item?;
            let id_str = id.value().to_string();
            let off = offset.value() as usize;
            reg.push((id_str.clone(), off));
            ito.insert(id_str.clone(), off);
            live.insert(id_str);
            count += 1;
        }

        self.vector_count.store(count, AtomicOrdering::SeqCst);
        Ok(())
    }

    /// Load all vectors and their associated metadata from the registry.
    /// Returns (IDs, Vectors, Offsets).
    fn load_all_vectors(&self) -> (Vec<String>, Vec<EntangledHVec>, Vec<usize>) {
        let registry = self.registry.read();
        let mut ids = Vec::with_capacity(registry.len());
        let mut vectors = Vec::with_capacity(registry.len());
        let mut offsets = Vec::with_capacity(registry.len());

        for (id, offset) in registry.iter() {
            let (_, vec) = self.read_entry(*offset);
            ids.push(id.clone());
            vectors.push(vec);
            offsets.push(*offset);
        }

        (ids, vectors, offsets)
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

    pub fn delete(&self, id: &str) -> Result<()> {
        let write_tx = self.db.begin_write()?;
        {
            let mut table = write_tx.open_table(DELETED_IDS_TABLE)?;
            table.insert(id, 1)?;
            // Also remove from persistent registry so reloads skip it
            let mut reg_table = write_tx.open_table(REGISTRY_TABLE)?;
            reg_table.remove(id)?;
        }
        write_tx.commit()?;

        // Remove from in-memory registry immediately so all query paths
        // (brute-force, NSG result filtering, IVF result filtering) skip it.
        // Update vector_count under the same write lock to avoid TOCTOU race
        // with concurrent memorize() calls.
        {
            let mut reg = self.registry.write();
            let mut ito = self.id_to_offset.write();
            reg.retain(|(eid, _)| eid != id);
            ito.remove(id);
            self.live_ids.write().remove(id);
            self.vector_count
                .store(reg.len() as u64, AtomicOrdering::SeqCst);
        }

        self.rebuild_inverted_index()?;

        Ok(())
    }

    pub fn memorize(&self, id: String, vector: EntangledHVec) -> Result<()> {
        // Check if ID is deleted before memorizing
        {
            let read_tx = self.db.begin_read()?;
            let table = read_tx.open_table(DELETED_IDS_TABLE)?;
            if table.get(id.as_str())?.is_some() {
                return Err(anyhow::anyhow!("Cannot memorize deleted ID: {}", id));
            }
        }

        let id_bytes = id.as_bytes();
        if id_bytes.len() > u16::MAX as usize {
            return Err(anyhow::anyhow!(
                "ID too long: {} bytes (max {})",
                id_bytes.len(),
                u16::MAX
            ));
        }
        let len_bytes = (id_bytes.len() as u16).to_le_bytes();

        let deltas = vector.to_deltas();
        let delta_count = (deltas.len() as u32).to_le_bytes();

        let mut entry = Vec::with_capacity(2 + id_bytes.len() + 4 + deltas.len() * 4);
        entry.extend_from_slice(&len_bytes);
        entry.extend_from_slice(id_bytes);
        entry.extend_from_slice(&delta_count);
        for &d in &deltas {
            entry.extend_from_slice(&d.to_le_bytes());
        }

        let offset = self.arena.write_slice(&entry)?;

        // Add to persistent registry
        {
            let write_tx = self.db.begin_write()?;
            {
                let mut table = write_tx.open_table(REGISTRY_TABLE)?;
                table.insert(id.as_str(), offset as u64)?;
            }
            write_tx.commit()?;
        }

        // Add to in-memory registry (deduplicate: remove old entry if same ID exists).
        // Update vector_count under the same write lock to avoid TOCTOU race
        // with concurrent delete() calls.
        let (count, is_replacement) = {
            let mut reg = self.registry.write();
            let mut ito = self.id_to_offset.write();
            let exists = reg.iter().any(|(eid, _)| eid == &id);
            reg.retain(|(existing_id, _)| existing_id != &id);
            ito.remove(&id);
            reg.push((id.clone(), offset));
            ito.insert(id.clone(), offset);
            self.live_ids.write().insert(id.clone());
            let count = reg.len() as u64;
            self.vector_count.store(count, AtomicOrdering::SeqCst);
            (count, exists)
        };

        if is_replacement {
            self.rebuild_inverted_index()?;
        } else {
            let mut inv = self.inverted.write();
            // count-1 is the index of the newly pushed ID in registry
            inv.add_doc((count - 1) as u32, &vector.indices);
        }

        // Insert into IVF if trained. Note: old entries for the same ID remain in
        // the index but are filtered out during query via the live registry.
        // Full reclamation happens during `compact()`.
        if let Some(ref mut ivf) = *self.ivf.write() {
            ivf.insert(&id, &vector, offset)
                .context("IVF insert failed")?;
        }

        // Insert into NSG if trained
        if let Some(ref mut nsg) = *self.nsg.write() {
            nsg.insert(&id, &vector, offset)
                .context("NSG insert failed")?;
        }

        if self.config.ivf.enabled
            && self.config.ivf.auto_threshold > 0
            && count == self.config.ivf.auto_threshold as u64
        {
            self.train_ivf_internal().context("Auto-train IVF failed")?;
        }

        if self.config.nsg.auto_threshold > 0 && count == self.config.nsg.auto_threshold as u64 {
            self.train_nsg_internal().context("Auto-train NSG failed")?;
        }

        Ok(())
    }

    fn read_entry(&self, offset: usize) -> (String, EntangledHVec) {
        if let Ok((payload, _version)) = self.arena.read_frame(offset) {
            if payload.len() < 6 {
                return (
                    String::new(),
                    EntangledHVec::from_indices(vec![], self.dimensions),
                );
            }
            let id_len = u16::from_le_bytes([payload[0], payload[1]]) as usize;
            let id_end = 2 + id_len;
            if payload.len() < id_end + 4 {
                return (
                    String::new(),
                    EntangledHVec::from_indices(vec![], self.dimensions),
                );
            }
            let id = String::from_utf8_lossy(&payload[2..id_end]).to_string();
            let delta_count = u32::from_le_bytes(match payload[id_end..id_end + 4].try_into() {
                Ok(b) => b,
                Err(_) => {
                    return (id, EntangledHVec::from_indices(vec![], self.dimensions));
                }
            }) as usize;
            let deltas_start = id_end + 4;
            let deltas_end = deltas_start + delta_count * 4;
            if payload.len() < deltas_end {
                return (id, EntangledHVec::from_indices(vec![], self.dimensions));
            }
            let deltas: Vec<u32> = payload[deltas_start..deltas_end]
                .chunks_exact(4)
                .map(|c| {
                    // chunks_exact(4) guarantees exactly 4 bytes per chunk
                    u32::from_le_bytes(c.try_into().unwrap())
                })
                .collect();
            (id, EntangledHVec::from_deltas(&deltas, self.dimensions))
        } else {
            (
                String::new(),
                EntangledHVec::from_indices(vec![], self.dimensions),
            )
        }
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
        self.train_ivf_internal()
    }

    fn train_ivf_internal(&self) -> Result<()> {
        let (ids, vectors, offsets) = self.load_all_vectors();
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
            &offsets,
            &ids,
            self.dimensions,
            &self.config.ivf,
            self.db.clone(),
        )?;

        self.save_ivf(&index)?; // PERSIST
        *self.ivf.write() = Some(index);
        Ok(())
    }

    pub fn nsg_trained(&self) -> bool {
        self.nsg.read().as_ref().is_some_and(|nsg| nsg.is_trained())
    }

    pub fn train_nsg(&self) -> Result<()> {
        self.train_nsg_internal()
    }

    fn train_nsg_internal(&self) -> Result<()> {
        let (ids, vectors, offsets) = self.load_all_vectors();
        if ids.is_empty() {
            return Ok(());
        }

        let index = super::nsg::training::train(
            &vectors,
            &ids,
            &offsets,
            self.dimensions,
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
