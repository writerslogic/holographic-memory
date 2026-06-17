// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::entangled::EntangledHVec;
use super::indexed_memory::{hopfield_cleanup, CleanupResult, IndexedMemory};

const ATOM_MAGIC: u8 = 0xFD;

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
        let seed = fxhash::hash64(atom_str.as_bytes());
        let vec = EntangledHVec::new_deterministic(self.inner.dim(), seed);
        let idx = self.inner.insert(atom_str.to_string(), vec.clone());
        (idx, vec)
    }

    pub fn get(&self, id: &str) -> Option<EntangledHVec> {
        self.inner.get(id)
    }

    pub fn get_by_idx(&self, idx: u32) -> Option<(String, EntangledHVec)> {
        self.inner.get_by_idx(idx)
    }

    pub fn cleanup(&self, noisy: &EntangledHVec, beta: f64, k: usize, max_iter: usize) -> CleanupResult {
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

    pub fn serialize_atom(id: &str, vec: &EntangledHVec) -> Vec<u8> {
        let id_bytes = id.as_bytes();
        let deltas = vec.to_deltas();
        let mut buf = Vec::with_capacity(1 + 2 + id_bytes.len() + 4 + deltas.len() * 4);
        buf.push(ATOM_MAGIC);
        buf.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(id_bytes);
        buf.extend_from_slice(&(deltas.len() as u32).to_le_bytes());
        for &d in &deltas {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf
    }

    pub fn deserialize_atom(data: &[u8], dim: usize) -> Option<(String, EntangledHVec)> {
        if data.is_empty() || data[0] != ATOM_MAGIC {
            return None;
        }
        let mut pos = 1;
        let id_len = u16::from_le_bytes(data.get(pos..pos + 2)?.try_into().ok()?) as usize;
        pos += 2;
        let id = std::str::from_utf8(data.get(pos..pos + id_len)?).ok()?.to_string();
        pos += id_len;
        let delta_count = u32::from_le_bytes(data.get(pos..pos + 4)?.try_into().ok()?) as usize;
        pos += 4;
        let deltas: Vec<u32> = (0..delta_count)
            .filter_map(|i| {
                let start = pos + i * 4;
                Some(u32::from_le_bytes(data.get(start..start + 4)?.try_into().ok()?))
            })
            .collect();
        let vec = EntangledHVec::from_deltas(&deltas, dim);
        Some((id, vec))
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

    #[test]
    fn test_atom_serialize_roundtrip() {
        let id = "test_atom";
        let vec = EntangledHVec::new_deterministic(16384, 42);
        let data = AtomMemory::serialize_atom(id, &vec);
        let (parsed_id, parsed_vec) = AtomMemory::deserialize_atom(&data, 16384).unwrap();
        assert_eq!(parsed_id, id);
        assert!((parsed_vec.similarity(&vec) - 1.0).abs() < 0.0001);
    }
}
