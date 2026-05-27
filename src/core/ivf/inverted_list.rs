use anyhow::Result;
use parking_lot::RwLock;
use std::sync::Arc;
use fxhash::FxHashMap;

const PQ_CODE_LEN: usize = 16;

/// Entry format: [id: String][pq_codes: 16][arena_offset: 8]
pub(crate) struct InvertedListEntry {
    pub id: String,
    pub pq_codes: [u8; PQ_CODE_LEN],
    pub arena_offset: usize,
}

pub(crate) struct InvertedLists {
    lists: Arc<RwLock<FxHashMap<u32, Vec<InvertedListEntry>>>>,
}

impl InvertedLists {
    pub fn new() -> Self {
        Self {
            lists: Arc::new(RwLock::new(FxHashMap::default())),
        }
    }

    /// Append an entry to the inverted list for `cluster_id`.
    pub fn append(
        &self,
        cluster_id: usize,
        id: &str,
        pq_codes: &[u8; PQ_CODE_LEN],
        arena_offset: usize,
    ) -> Result<()> {
        let mut lists = self.lists.write();
        let entry = InvertedListEntry {
            id: id.to_string(),
            pq_codes: *pq_codes,
            arena_offset,
        };
        lists.entry(cluster_id as u32).or_default().push(entry);
        Ok(())
    }

    /// Read all entries from the inverted list for `cluster_id`.
    pub fn read(&self, cluster_id: usize) -> Result<Vec<InvertedListEntry>> {
        let lists = self.lists.read();
        if let Some(list) = lists.get(&(cluster_id as u32)) {
            // Return a clone of the entries for safety, though it's less efficient than references.
            // For 1M scale, we might want to return references or use a different structure.
            let cloned = list.iter().map(|e| InvertedListEntry {
                id: e.id.clone(),
                pq_codes: e.pq_codes,
                arena_offset: e.arena_offset,
            }).collect();
            Ok(cloned)
        } else {
            Ok(Vec::new())
        }
    }

    /// Clear all inverted lists. Critical for correct re-training.
    pub fn clear_all(&self) -> Result<()> {
        self.lists.write().clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inverted_list_round_trip() {
        let lists = InvertedLists::new();

        let codes1 = [1u8; 16];
        let codes2 = [2u8; 16];

        lists.append(0, "vec_a", &codes1, 1000).unwrap();
        lists.append(0, "vec_b", &codes2, 2000).unwrap();
        lists.append(1, "vec_c", &codes1, 3000).unwrap();

        let entries0 = lists.read(0).unwrap();
        assert_eq!(entries0.len(), 2);
        assert_eq!(entries0[0].id, "vec_a");
        assert_eq!(entries0[0].pq_codes, codes1);
        assert_eq!(entries0[0].arena_offset, 1000);
        assert_eq!(entries0[1].id, "vec_b");
        assert_eq!(entries0[1].pq_codes, codes2);
        assert_eq!(entries0[1].arena_offset, 2000);

        let entries1 = lists.read(1).unwrap();
        assert_eq!(entries1.len(), 1);
        assert_eq!(entries1[0].id, "vec_c");

        let entries2 = lists.read(2).unwrap();
        assert!(entries2.is_empty());
    }
}
