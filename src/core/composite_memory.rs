// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use super::entangled::EntangledHVec;
use super::indexed_memory::IndexedMemory;

pub struct CompositeMemory {
    inner: IndexedMemory,
}

impl CompositeMemory {
    pub fn new(dim: usize, idf_clip: f64) -> Self {
        Self {
            inner: IndexedMemory::new(dim, idf_clip),
        }
    }

    pub fn insert(&self, id: String, composite: EntangledHVec) -> u32 {
        self.inner.insert(id, composite)
    }

    #[allow(dead_code)]
    pub fn get(&self, id: &str) -> Option<EntangledHVec> {
        self.inner.get(id)
    }

    pub fn get_by_idx(&self, idx: u32) -> Option<(String, EntangledHVec)> {
        self.inner.get_by_idx(idx)
    }

    pub fn overlap_scan(&self, query: &EntangledHVec) -> Vec<(u32, f32)> {
        self.inner.overlap_scan(query)
    }

    #[allow(dead_code)]
    pub fn delete(&self, id: &str) -> bool {
        self.inner.delete(id)
    }

    pub fn count(&self) -> usize {
        self.inner.count()
    }

    pub fn rebuild_indices(&self) {
        self.inner.rebuild_indices();
    }

    pub fn inner(&self) -> &IndexedMemory {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composite_insert_scan() {
        let mem = CompositeMemory::new(16384, 3.0);
        let v1 = EntangledHVec::new_deterministic(16384, 100);
        let v2 = EntangledHVec::new_deterministic(16384, 200);
        mem.insert("c1".to_string(), v1.clone());
        mem.insert("c2".to_string(), v2);

        let results = mem.overlap_scan(&v1);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0);
    }
}
