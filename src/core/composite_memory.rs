// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::entangled::EntangledHVec;
use super::indexed_memory::IndexedMemory;
use super::wire;

const COMPOSITE_MAGIC: u8 = wire::magic::COMPOSITE;

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

    pub fn load_composite(&self, id: String, vec: EntangledHVec) {
        self.inner.insert(id, vec);
    }

    pub fn inner(&self) -> &IndexedMemory {
        &self.inner
    }

    pub fn serialize_composite(id: &str, vec: &EntangledHVec) -> Vec<u8> {
        let deltas = vec.to_deltas();
        let mut buf = Vec::with_capacity(1 + 2 + id.len() + 4 + deltas.len() * 4);
        buf.push(COMPOSITE_MAGIC);
        wire::write_lp_str(&mut buf, id);
        wire::write_deltas(&mut buf, &deltas);
        buf
    }

    pub fn deserialize_composite(data: &[u8], dim: usize) -> Option<(String, EntangledHVec)> {
        if data.is_empty() || data[0] != COMPOSITE_MAGIC {
            return None;
        }
        let (id, pos) = wire::read_lp_str(data, 1)?;
        let (deltas, _) = wire::read_deltas(data, pos)?;
        let vec = EntangledHVec::from_deltas(&deltas, dim);
        Some((id, vec))
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

    #[test]
    fn test_composite_serialize_roundtrip() {
        let vec = EntangledHVec::new_deterministic(16384, 42);
        let data = CompositeMemory::serialize_composite("triple_1", &vec);
        let (id, parsed) = CompositeMemory::deserialize_composite(&data, 16384).unwrap();
        assert_eq!(id, "triple_1");
        assert!((parsed.similarity(&vec) - 1.0).abs() < 0.0001);
    }
}
