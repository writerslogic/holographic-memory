use super::HmsCore;
use super::{DELETED_IDS_TABLE, REGISTRY_TABLE};
use crate::core::storage::PersistentArena;
use anyhow::Result;
use redb::ReadableTable;
use std::sync::atomic::Ordering as AtomicOrdering;

impl HmsCore {
    /// Compact the arena by removing deleted entries and reclaiming space.
    /// This streams active entries to a temporary arena and swaps them,
    /// avoiding high memory usage for large databases.
    pub fn compact(&self) -> Result<()> {
        let registry = self.registry.read().clone();

        let parent_dir = self
            .arena
            .base_path()
            .parent()
            .unwrap_or(std::path::Path::new("."));
        let temp_dir_name = format!(
            "compact_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros()
        );
        let temp_dir_path = parent_dir.join(temp_dir_name);

        let mut new_registry: Vec<(String, usize)> = Vec::with_capacity(registry.len());

        {
            // Block scope ensures temp_arena is dropped and file handles are released before swap
            let temp_arena = PersistentArena::new(&temp_dir_path)?;
            let read_tx = self.db.begin_read()?;
            let tombstone_table = read_tx.open_table(DELETED_IDS_TABLE)?;

            for (id, offset) in registry {
                // Double check tombstone just in case
                if tombstone_table.get(id.as_str())?.is_none() {
                    let (payload, _version) = self.arena.read_frame(offset)?;
                    let new_offset = temp_arena.write_slice(&payload)?;
                    new_registry.push((id, new_offset));
                }
            }
        }

        // Atomically replace the current arena files with the compacted files
        self.arena.replace_with_compacted(&temp_dir_path)?;

        // Update the metadata registry
        let write_tx = self.db.begin_write()?;
        {
            let mut reg_table = write_tx.open_table(REGISTRY_TABLE)?;
            for (id, new_offset) in &new_registry {
                reg_table.insert(id.as_str(), *new_offset as u64)?;
            }

            // Clear tombstones after successful compaction
            let mut ts_table = write_tx.open_table(DELETED_IDS_TABLE)?;
            let keys: Vec<String> = ts_table
                .iter()?
                .filter_map(|r| r.ok())
                .map(|(k, _)| k.value().to_string())
                .collect();
            for k in keys {
                ts_table.remove(k.as_str())?;
            }
        }
        write_tx.commit()?;

        let mut reg_guard = self.registry.write();
        let mut ito_guard = self.id_to_offset.write();
        
        *reg_guard = new_registry;
        
        // Refresh id_to_offset cache
        ito_guard.clear();
        for (id, offset) in reg_guard.iter() {
            ito_guard.insert(id.clone(), *offset);
        }

        self.vector_count
            .store(reg_guard.len() as u64, AtomicOrdering::SeqCst);

        drop(reg_guard);
        drop(ito_guard);

        // Force re-train of indices to match new offsets
        self.train_nsg_internal()?;
        self.train_ivf_internal()?;
        self.rebuild_inverted_index()?;

        Ok(())
    }
}
