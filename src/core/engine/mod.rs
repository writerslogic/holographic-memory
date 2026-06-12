pub(crate) mod concepts;
pub(crate) mod knowledge;
pub(crate) mod query;
pub(crate) mod router;
pub(crate) mod shard;

use anyhow::{Context, Result};
use parking_lot::RwLock;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::config::HmsConfig;
use super::diffusion::{DiffusionConfig, DiffusionFactorizer};
use super::encoding::encode_text_internal;
use super::entangled::EntangledHVec;
use super::ivf::IVFIndex;
use super::storage::PersistentArena;
use super::text::TextProcessor;
use super::types::TextMetrics;

use shard::{ShardSet, ShardManager};

/// Lock ordering: ShardSet (read) -> Shard.vectors -> Shard.ivf -> Shard.nsg.
/// Arena lock is independent (managed internally by PersistentArena).
pub struct HmsCore {
    config: HmsConfig,
    pub(crate) arena: Arc<PersistentArena>,
    pub(crate) dimensions: usize,
    pub(crate) storage_path: PathBuf,
    shards: RwLock<ShardSet>,
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

        let shard_set = if config.shard.enabled && config.shard.shard_count > 1 {
            ShardSet::Multi(ShardManager::new(config.shard.shard_count, dim))
        } else {
            ShardSet::Single(Box::new(shard::Shard::new(dim)))
        };

        let core = Self {
            config: config.clone(),
            arena,
            dimensions: dim,
            storage_path: base_path,
            shards: RwLock::new(shard_set),
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
            let data = std::fs::read(&nsg_path)?;
            let nsg: super::nsg::NSGIndex = bincode::deserialize(&data)?;
            // Load NSG into the first (or only) shard
            let shards = self.shards.read();
            if let ShardSet::Single(ref shard) = *shards {
                *shard.nsg.write() = Some(nsg);
            }
        }

        let ivf_path = self.storage_path.join("ivf_index.bin");
        if ivf_path.exists() {
            let data = std::fs::read(&ivf_path)?;
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
        std::fs::write(self.storage_path.join("nsg_index.bin"), data)?;
        Ok(())
    }

    fn save_ivf(&self, ivf: &IVFIndex) -> Result<()> {
        let data = bincode::serialize(ivf)?;
        std::fs::write(self.storage_path.join("ivf_index.bin"), data)?;
        Ok(())
    }

    fn load_from_log(&self) -> Result<()> {
        let shards = self.shards.read();

        let mut offset = 0;
        while let Ok((payload, _version)) = self.arena.read_frame(offset) {
            let (id, vector) = Self::parse_log_payload(self.dimensions, &payload);
            if vector.dim == 0 {
                shards.remove(&id, self.dimensions)?;
            } else {
                shards.insert(id, vector, self.dimensions)?;
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
            shard.vector_count.store(
                reg.len() as u64,
                std::sync::atomic::Ordering::SeqCst,
            );
        });

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
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (id, EntangledHVec::from_deltas(&deltas, dimensions))
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

    pub fn delete(&self, id: &str) -> Result<bool> {
        // Persist tombstone first for crash-safety: if we crash after the
        // arena write but before the memory remove, load_from_log replays
        // the tombstone and correctly removes the vector.
        self.arena.write_slice(&Self::serialize_tombstone(id)?)?;
        let shards = self.shards.read();
        if !shards.remove(id, self.dimensions)? {
            return Ok(false);
        }
        Ok(true)
    }

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

        {
            let temp_arena = PersistentArena::new(&temp_dir)?;
            for (id, vector) in &snapshot {
                let entry = Self::serialize_log_entry(id, vector)?;
                temp_arena.write_slice(&entry)?;
            }
        }

        self.arena.replace_with_compacted(&temp_dir)?;

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

    pub fn memorize(&self, id: String, vector: EntangledHVec) -> Result<()> {
        let entry = Self::serialize_log_entry(&id, &vector)?;
        self.arena.write_slice(&entry)?;

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
                    vectors.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                }
                ShardSet::Multi(_) => return,
            }
        };

        let n_shards = (count as usize / cfg.target_shard_size).max(2);
        let mgr = ShardManager::new(n_shards, self.dimensions);
        for (id, vec) in snapshot {
            let target = mgr.shard_for(&id);
            // Insert into new shards cannot fail — fresh empty shards with no indices
            let _ = mgr.shards[target].insert(id, vec, self.dimensions);
        }

        *shards = ShardSet::Multi(mgr);
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
        self.shards.read().count()
    }

    pub fn ivf_trained(&self) -> bool {
        self.shards.read().ivf_trained()
    }

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

            let index = IVFIndex::train(
                &vectors,
                &ids,
                self.dimensions,
                &self.config.ivf,
            )?;
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

    pub fn nsg_trained(&self) -> bool {
        self.shards.read().nsg_trained()
    }

    pub fn train_nsg(&self) -> Result<()> {
        let shards = self.shards.read();
        shards.try_for_each_shard(|shard| {
            let (ids, vectors) = shard.load_all_vectors();
            if ids.is_empty() {
                return Ok(());
            }

            let index = super::nsg::training::train(
                &vectors,
                &ids,
                &self.config.nsg,
            )?;
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
