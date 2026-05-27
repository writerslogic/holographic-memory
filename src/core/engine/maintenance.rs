use super::HmsCore;
use crate::core::storage::PersistentArena;
use anyhow::Result;
use fxhash::FxHashMap;

impl HmsCore {
    /// Compact the arena by removing deleted entries and reclaiming space.
    /// This streams active entries to a temporary arena and swaps them.
    pub fn compact(&self) -> Result<()> {
        let registry = self.registry.read().clone();
        let vectors = self.vectors.read();

        let temp_dir_name = format!(
            "compact_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros()
        );
        let temp_dir_path = self.storage_path.join(temp_dir_name);

        let mut new_vectors = FxHashMap::default();

        {
            // Block scope ensures temp_arena is dropped and file handles are released before swap
            let temp_arena = PersistentArena::new(&temp_dir_path)?;

            for id in &registry {
                if let Some((vec, _)) = vectors.get(id) {
                    let id_bytes = id.as_bytes();
                    let deltas = vec.to_deltas();
                    
                    let mut entry = Vec::with_capacity(2 + id_bytes.len() + 4 + deltas.len() * 4);
                    entry.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
                    entry.extend_from_slice(id_bytes);
                    entry.extend_from_slice(&(deltas.len() as u32).to_le_bytes());
                    for &d in &deltas {
                        entry.extend_from_slice(&d.to_le_bytes());
                    }
                    
                    let new_offset = temp_arena.write_slice(&entry)?;
                    new_vectors.insert(id.clone(), (vec.clone(), new_offset));
                }
            }
        }

        // Atomically replace the current arena files with the compacted files
        self.arena.replace_with_compacted(&temp_dir_path)?;

        // Update the in-memory vectors map with new offsets
        {
            let mut vectors_guard = self.vectors.write();
            *vectors_guard = new_vectors;
        }

        // Serialize indices to disk
        if let Some(ref nsg) = *self.nsg.read() {
            self.save_nsg(nsg)?;
        }
        if let Some(ref ivf) = *self.ivf.read() {
            self.save_ivf(ivf)?;
        }

        self.rebuild_inverted_index()?;

        Ok(())
    }
}
