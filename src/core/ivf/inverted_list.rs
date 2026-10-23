// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::core::hash::FastMap;
use crate::core::sync::RwLock;
use anyhow::{anyhow, Result};
use std::sync::Arc;

const PQ_CODE_LEN: usize = 16;

pub(crate) struct InvertedListEntry {
    pub id: String,
    pub pq_codes: [u8; PQ_CODE_LEN],
}

pub(crate) struct InvertedLists {
    lists: Arc<RwLock<FastMap<u32, Vec<InvertedListEntry>>>>,
}

impl InvertedLists {
    pub fn new() -> Self {
        Self {
            lists: Arc::new(RwLock::new(FastMap::default())),
        }
    }

    /// Append an entry to the inverted list for `cluster_id`.
    pub fn append(&self, cluster_id: usize, id: &str, pq_codes: &[u8; PQ_CODE_LEN]) -> Result<()> {
        let entry = InvertedListEntry {
            id: id.to_string(),
            pq_codes: *pq_codes,
        };
        self.lists
            .write()
            .entry(cluster_id as u32)
            .or_default()
            .push(entry);
        Ok(())
    }

    // Entries are append-only: freeze the prefix length, then release the read lock every
    // 256 entries. The callback must not reenter these lists.
    pub fn visit(&self, cluster_id: usize, mut visitor: impl FnMut(usize, &InvertedListEntry)) {
        let key = cluster_id as u32;
        let len = self.lists.read().get(&key).map_or(0, Vec::len);
        for start in (0..len).step_by(256) {
            let lists = self.lists.read();
            let entries = lists.get(&key).expect("append-only list disappeared");
            let end = start.saturating_add(256).min(len);
            for (offset, entry) in entries[start..end].iter().enumerate() {
                visitor(start + offset, entry);
            }
        }
    }

    pub fn id_at(&self, cluster_id: usize, index: usize) -> Result<String> {
        self.lists
            .read()
            .get(&(cluster_id as u32))
            .and_then(|entries| entries.get(index))
            .map(|entry| entry.id.clone())
            .ok_or_else(|| anyhow!("append-only inverted-list entry disappeared"))
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

        lists.append(0, "vec_a", &codes1).unwrap();
        lists.append(0, "vec_b", &codes2).unwrap();
        lists.append(1, "vec_c", &codes1).unwrap();

        let mut entries0 = Vec::new();
        lists.visit(0, |index, entry| entries0.push((index, entry.pq_codes)));
        assert_eq!(entries0, [(0, codes1), (1, codes2)]);
        assert_eq!(lists.id_at(0, 0).unwrap(), "vec_a");
        assert_eq!(lists.id_at(0, 1).unwrap(), "vec_b");
        let mut entries1 = Vec::new();
        lists.visit(1, |index, entry| entries1.push((index, entry.pq_codes)));
        assert_eq!(entries1, [(0, codes1)]);
        assert_eq!(lists.id_at(1, 0).unwrap(), "vec_c");
        lists.visit(2, |_, _| panic!("absent cluster must be empty"));
    }

    #[test]
    fn traversal_preserves_order_across_batch_boundaries() {
        for len in [255, 256, 257] {
            let lists = InvertedLists::new();
            for index in 0..len {
                lists
                    .append(0, &index.to_string(), &[index as u8; 16])
                    .unwrap();
            }
            let mut seen = Vec::new();
            lists.visit(0, |index, entry| {
                assert_eq!(entry.pq_codes, [index as u8; 16]);
                seen.push(index);
            });
            assert_eq!(seen, (0..len).collect::<Vec<_>>());
            for index in seen {
                assert_eq!(lists.id_at(0, index).unwrap(), index.to_string());
            }
        }
    }
}
