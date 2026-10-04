// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use super::entangled::EntangledHVec;
use super::indexed_memory::{hopfield_cleanup, CleanupResult, IndexedMemory};

pub struct AtomMemory {
    inner: IndexedMemory,
}

impl AtomMemory {
    pub fn new(dim: usize, idf_clip: f64) -> Self {
        Self {
            inner: IndexedMemory::new(dim, idf_clip),
        }
    }

    pub fn get_or_insert(&self, atom_str: &str) -> (u32, EntangledHVec) {
        if let Some(idx) = self.inner.idx_for(atom_str) {
            if let Some((_, vec)) = self.inner.get_by_idx(idx) {
                return (idx, vec);
            }
        }
        let vec = super::encoding::encode_text_internal(atom_str, self.inner.dim());
        let idx = self.inner.insert(atom_str.to_string(), vec.clone());
        (idx, vec)
    }

    pub fn insert_with_vec(&self, id: &str, vec: &EntangledHVec) -> u32 {
        self.inner.insert(id.to_string(), vec.clone())
    }

    pub fn get(&self, id: &str) -> Option<EntangledHVec> {
        self.inner.get(id)
    }

    #[allow(dead_code)]
    pub fn get_by_idx(&self, idx: u32) -> Option<(String, EntangledHVec)> {
        self.inner.get_by_idx(idx)
    }

    pub fn cleanup(
        &self,
        noisy: &EntangledHVec,
        beta: f64,
        k: usize,
        max_iter: usize,
    ) -> CleanupResult {
        hopfield_cleanup(noisy, &self.inner, beta, k, max_iter)
    }

    pub fn delete(&self, id: &str) -> bool {
        self.inner.delete(id)
    }

    pub fn count(&self) -> usize {
        self.inner.count()
    }

    pub fn rebuild_indices(&self) {
        self.inner.rebuild_indices();
    }

    pub fn load_atom(&self, id: String, vec: EntangledHVec) {
        self.inner.insert(id, vec);
    }

    pub fn inner(&self) -> &IndexedMemory {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atom_get_or_insert() {
        let mem = AtomMemory::new(16384, 3.0);
        let (idx1, v1) = mem.get_or_insert("cat");
        let (idx2, v2) = mem.get_or_insert("cat");
        assert_eq!(idx1, idx2);
        assert!((v1.similarity(&v2) - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_atom_cleanup() {
        let mem = AtomMemory::new(16384, 3.0);
        for i in 0..50u64 {
            mem.get_or_insert(&format!("atom_{}", i));
        }
        let (_, original) = mem.get_or_insert("atom_25");
        let result = mem.cleanup(&original, 24.0, 64, 3);
        assert!(result.found);
        assert_eq!(result.id, "atom_25");
    }
}
