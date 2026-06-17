// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use fxhash::FxHashMap;
use parking_lot::RwLock;

use super::entangled::EntangledHVec;
use super::idf::IdfWeights;
use super::posting::PostingShard;
use super::tombstone::TombstoneMap;

pub struct IndexedMemory {
    vectors: RwLock<Vec<(String, EntangledHVec)>>,
    id_to_idx: RwLock<FxHashMap<String, usize>>,
    postings: RwLock<PostingShard>,
    idf: RwLock<IdfWeights>,
    tombstones: RwLock<TombstoneMap>,
    dim: usize,
}

pub struct CleanupResult {
    pub found: bool,
    pub id: String,
    pub vector: EntangledHVec,
    pub confidence: f64,
    pub iterations: usize,
}

impl IndexedMemory {
    pub fn new(dim: usize, idf_clip_factor: f64) -> Self {
        Self {
            vectors: RwLock::new(Vec::new()),
            id_to_idx: RwLock::new(FxHashMap::default()),
            postings: RwLock::new(PostingShard::new(dim)),
            idf: RwLock::new(IdfWeights::new(dim, idf_clip_factor)),
            tombstones: RwLock::new(TombstoneMap::new()),
            dim,
        }
    }

    pub fn insert(&self, id: String, vec: EntangledHVec) -> u32 {
        let mut vecs = self.vectors.write();
        let idx = vecs.len() as u32;
        self.postings.write().insert(idx, vec.indices());
        self.idf
            .write()
            .update_insert(vec.indices(), &self.postings.read());
        self.id_to_idx.write().insert(id.clone(), idx as usize);
        vecs.push((id, vec));
        idx
    }

    pub fn get(&self, id: &str) -> Option<EntangledHVec> {
        let map = self.id_to_idx.read();
        let idx = *map.get(id)?;
        let vecs = self.vectors.read();
        if self.tombstones.read().is_deleted(idx as u32) {
            return None;
        }
        Some(vecs[idx].1.clone())
    }

    pub fn idx_for(&self, id: &str) -> Option<u32> {
        self.id_to_idx.read().get(id).map(|&i| i as u32)
    }

    pub fn get_by_idx(&self, idx: u32) -> Option<(String, EntangledHVec)> {
        let vecs = self.vectors.read();
        if (idx as usize) >= vecs.len() || self.tombstones.read().is_deleted(idx) {
            return None;
        }
        let entry = &vecs[idx as usize];
        Some((entry.0.clone(), entry.1.clone()))
    }

    pub fn delete(&self, id: &str) -> bool {
        let map = self.id_to_idx.read();
        if let Some(&idx) = map.get(id) {
            self.tombstones.write().mark_deleted(idx as u32);
            true
        } else {
            false
        }
    }

    pub fn overlap_scan(&self, query: &EntangledHVec) -> Vec<(u32, f32)> {
        let postings = self.postings.read();
        let tombstones = self.tombstones.read();
        let idf = self.idf.read();
        let weights = idf.weights_for(query.indices());
        postings.weighted_overlap(query.indices(), &weights, &tombstones)
    }

    pub fn count(&self) -> usize {
        let vecs = self.vectors.read();
        let tombstones = self.tombstones.read();
        vecs.len() - tombstones.count()
    }

    pub fn rebuild_indices(&self) {
        let vecs = self.vectors.read();
        let entries: Vec<(u32, &[u32])> = vecs
            .iter()
            .enumerate()
            .map(|(i, (_, v))| (i as u32, v.indices()))
            .collect();
        self.postings.write().rebuild(&entries);
        self.idf
            .write()
            .recompute(&self.postings.read(), vecs.len() as u32);
    }

    pub fn all_vectors(&self) -> Vec<(u32, String, EntangledHVec)> {
        let vecs = self.vectors.read();
        let tombstones = self.tombstones.read();
        vecs.iter()
            .enumerate()
            .filter(|(i, _)| !tombstones.is_deleted(*i as u32))
            .map(|(i, (id, v))| (i as u32, id.clone(), v.clone()))
            .collect()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn vectors_read(&self) -> parking_lot::RwLockReadGuard<'_, Vec<(String, EntangledHVec)>> {
        self.vectors.read()
    }
}

pub fn sparse_softmax(overlaps: &[(u32, f32)], beta: f64) -> Vec<(u32, f64)> {
    if overlaps.is_empty() {
        return Vec::new();
    }
    let max_val = overlaps
        .iter()
        .map(|o| o.1)
        .fold(f32::NEG_INFINITY, f32::max) as f64;
    let exps: Vec<(u32, f64)> = overlaps
        .iter()
        .map(|&(id, score)| (id, (beta * (score as f64 - max_val)).exp()))
        .collect();
    let sum: f64 = exps.iter().map(|e| e.1).sum();
    if sum == 0.0 {
        return exps;
    }
    exps.into_iter().map(|(id, e)| (id, e / sum)).collect()
}

pub fn hopfield_cleanup(
    probe: &EntangledHVec,
    memory: &IndexedMemory,
    beta: f64,
    k: usize,
    max_iter: usize,
) -> CleanupResult {
    let empty = CleanupResult {
        found: false,
        id: String::new(),
        vector: EntangledHVec::from_indices(vec![], probe.dim),
        confidence: 0.0,
        iterations: 0,
    };

    let mut current = probe.clone();

    for iter in 0..max_iter {
        let overlaps = memory.overlap_scan(&current);
        if overlaps.is_empty() {
            return empty;
        }

        let attention = sparse_softmax(&overlaps, beta);

        let vecs = memory.vectors_read();
        let mut dim_scores: FxHashMap<u32, f64> = FxHashMap::default();
        for &(vec_id, weight) in &attention {
            if weight < 1e-10 {
                continue;
            }
            if let Some((_, v)) = vecs.get(vec_id as usize) {
                for &idx in v.indices() {
                    *dim_scores.entry(idx).or_insert(0.0) += weight;
                }
            }
        }
        drop(vecs);

        let mut scored: Vec<(u32, f64)> = dim_scores.into_iter().collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        let mut new_indices: Vec<u32> = scored.into_iter().map(|(idx, _)| idx).collect();
        new_indices.sort_unstable();

        let new_vec = EntangledHVec::from_indices(new_indices, current.dim);
        let sim = current.similarity(&new_vec);
        current = new_vec;

        if sim > 0.999 {
            let best = &attention[0];
            if let Some((id, vec)) = memory.get_by_idx(best.0) {
                return CleanupResult {
                    found: true,
                    id,
                    vector: vec,
                    confidence: best.1,
                    iterations: iter + 1,
                };
            }
            return empty;
        }
    }

    let overlaps = memory.overlap_scan(&current);
    if overlaps.is_empty() {
        return empty;
    }
    let attention = sparse_softmax(&overlaps, beta);
    let best = &attention[0];
    if let Some((id, vec)) = memory.get_by_idx(best.0) {
        CleanupResult {
            found: true,
            id,
            vector: vec,
            confidence: best.1,
            iterations: max_iter,
        }
    } else {
        empty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexed_memory_insert_get() {
        let mem = IndexedMemory::new(16384, 3.0);
        let v = EntangledHVec::new_deterministic(16384, 42);
        let idx = mem.insert("test".to_string(), v.clone());
        assert_eq!(idx, 0);
        let retrieved = mem.get("test").unwrap();
        assert!((retrieved.similarity(&v) - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_indexed_memory_delete() {
        let mem = IndexedMemory::new(16384, 3.0);
        let v = EntangledHVec::new_deterministic(16384, 42);
        mem.insert("test".to_string(), v);
        assert_eq!(mem.count(), 1);
        assert!(mem.delete("test"));
        assert_eq!(mem.count(), 0);
        assert!(mem.get("test").is_none());
    }

    #[test]
    fn test_overlap_scan() {
        let mem = IndexedMemory::new(16384, 3.0);
        let v1 = EntangledHVec::new_deterministic(16384, 1);
        let v2 = EntangledHVec::new_deterministic(16384, 2);
        mem.insert("a".to_string(), v1.clone());
        mem.insert("b".to_string(), v2);

        let results = mem.overlap_scan(&v1);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0); // v1 should be top match
    }

    #[test]
    fn test_hopfield_cleanup_exact() {
        let mem = IndexedMemory::new(16384, 3.0);
        for i in 0..100u64 {
            let v = EntangledHVec::new_deterministic(16384, i);
            mem.insert(format!("atom_{}", i), v);
        }

        let query = EntangledHVec::new_deterministic(16384, 42);
        let result = hopfield_cleanup(&query, &mem, 24.0, 64, 3);
        assert!(result.found);
        assert_eq!(result.id, "atom_42");
        assert!(result.confidence > 0.9);
    }

    #[test]
    fn test_hopfield_cleanup_noisy() {
        let mem = IndexedMemory::new(16384, 3.0);
        for i in 0..100u64 {
            let v = EntangledHVec::new_deterministic(16384, i);
            mem.insert(format!("atom_{}", i), v);
        }

        let original = EntangledHVec::new_deterministic(16384, 42);
        let indices = original.indices().to_vec();
        let mut noisy = indices.clone();
        // Flip 25% of active indices (16 of 64)
        let mut rng = rand::thread_rng();
        use rand::Rng;
        for item in noisy.iter_mut().take(16) {
            *item = rng.gen_range(0..16384u32);
        }
        noisy.sort_unstable();
        noisy.dedup();
        let noisy_vec = EntangledHVec::from_indices(noisy, 16384);

        let result = hopfield_cleanup(&noisy_vec, &mem, 24.0, 64, 3);
        assert!(result.found, "Should recover from 25% noise");
        assert_eq!(result.id, "atom_42");
    }

    #[test]
    fn test_sparse_softmax() {
        let overlaps = vec![(0, 10.0f32), (1, 5.0), (2, 1.0)];
        let result = sparse_softmax(&overlaps, 1.0);
        assert!(result[0].1 > result[1].1);
        assert!(result[1].1 > result[2].1);
        let sum: f64 = result.iter().map(|r| r.1).sum();
        assert!((sum - 1.0).abs() < 0.001);
    }
}
