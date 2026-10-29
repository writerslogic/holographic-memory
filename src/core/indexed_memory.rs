// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::core::hash::FastMap;
use crate::core::sync::RwLock;

use super::entangled::EntangledHVec;
use super::idf::IdfWeights;
use super::posting::PostingShard;
use super::query_limits::{QueryBudget, QueryLimitExceeded};
use super::tombstone::TombstoneMap;

pub struct IndexedMemory {
    vectors: RwLock<Vec<(String, EntangledHVec)>>,
    id_to_idx: RwLock<FastMap<String, usize>>,
    postings: RwLock<PostingShard>,
    idf: RwLock<IdfWeights>,
    tombstones: RwLock<TombstoneMap>,
    dim: usize,
}

pub struct CleanupResult {
    pub found: bool,
    pub id: String,
    pub confidence: f64,
}

impl IndexedMemory {
    pub fn new(dim: usize, idf_clip_factor: f64) -> Self {
        Self {
            vectors: RwLock::new(Vec::new()),
            id_to_idx: RwLock::new(FastMap::default()),
            postings: RwLock::new(PostingShard::new(dim)),
            idf: RwLock::new(IdfWeights::new(dim, idf_clip_factor)),
            tombstones: RwLock::new(TombstoneMap::new()),
            dim,
        }
    }

    pub fn insert(&self, id: String, vec: EntangledHVec) -> u32 {
        let mut vectors = self.vectors.write();
        let mut ids = self.id_to_idx.write();
        let mut postings = self.postings.write();
        let mut tombstones = self.tombstones.write();
        let mut idf = self.idf.write();
        let index = if let Some(&index) = ids.get(&id) {
            if !tombstones.is_deleted(index as u32) {
                postings.remove(index as u32, vectors[index].1.indices());
                idf.update_remove(vectors[index].1.indices(), &postings);
            }
            tombstones.restore(index as u32);
            vectors[index].1 = vec;
            index as u32
        } else {
            let index = vectors.len();
            ids.insert(id.clone(), index);
            vectors.push((id, vec));
            index as u32
        };
        postings.insert(index, vectors[index as usize].1.indices());
        idf.update_insert(vectors[index as usize].1.indices(), &postings);
        index
    }

    pub fn get(&self, id: &str) -> Option<EntangledHVec> {
        self.get_with_idx(id).map(|(_, vector)| vector)
    }

    pub(crate) fn get_with_idx(&self, id: &str) -> Option<(u32, EntangledHVec)> {
        let idx = *self.id_to_idx.read().get(id)?;
        let vecs = self.vectors.read();
        if self.tombstones.read().is_deleted(idx as u32) {
            return None;
        }
        Some((idx as u32, vecs[idx].1.clone()))
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
        let vectors = self.vectors.read();
        let ids = self.id_to_idx.read();
        let Some(&index) = ids.get(id) else {
            return false;
        };
        let mut postings = self.postings.write();
        let mut tombstones = self.tombstones.write();
        if tombstones.is_deleted(index as u32) {
            return false;
        }
        postings.remove(index as u32, vectors[index].1.indices());
        tombstones.mark_deleted(index as u32);
        self.idf
            .write()
            .update_remove(vectors[index].1.indices(), &postings);
        true
    }

    pub fn overlap_scan(&self, query: &EntangledHVec) -> Vec<(u32, f32)> {
        let postings = self.postings.read();
        let tombstones = self.tombstones.read();
        let idf = self.idf.read();
        let weights = idf.weights_for(query.indices());
        postings.weighted_overlap(query.indices(), &weights, &tombstones)
    }

    pub(crate) fn overlapping_slots(
        &self,
        query: &EntangledHVec,
        after: u32,
        through: u32,
    ) -> Vec<u32> {
        let postings = self.postings.read();
        let tombstones = self.tombstones.read();
        let mut candidates = crate::core::hash::FastSet::default();
        for &dimension in query.indices() {
            let list = postings.get_list(dimension);
            let start = list.partition_point(|&id| id <= after);
            let end = list.partition_point(|&id| id <= through);
            for &id in &list[start..end] {
                if !tombstones.is_deleted(id) {
                    candidates.insert(id);
                }
            }
        }
        candidates.into_iter().collect()
    }

    pub fn count(&self) -> usize {
        let vecs = self.vectors.read();
        let tombstones = self.tombstones.read();
        vecs.len() - tombstones.count()
    }

    pub fn rebuild_indices(&self) {
        let vecs = self.vectors.read();
        let mut postings = self.postings.write();
        let tombstones = self.tombstones.read();
        let entries: Vec<(u32, &[u32])> = vecs
            .iter()
            .enumerate()
            .filter(|(i, _)| !tombstones.is_deleted(*i as u32))
            .map(|(i, (_, v))| (i as u32, v.indices()))
            .collect();
        postings.rebuild(&entries);
        self.idf.write().recompute(
            &postings,
            u32::try_from(entries.len()).expect("indexed memory exceeds u32 live slots"),
        );
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

    pub(crate) fn bounded_snapshot(
        &self,
        budget: &mut QueryBudget,
    ) -> Result<Vec<(String, EntangledHVec)>, QueryLimitExceeded> {
        let vectors = self.vectors.read();
        let tombstones = self.tombstones.read();
        let mut snapshot = Vec::new();
        for (slot, (id, vector)) in vectors.iter().enumerate() {
            budget.work(1)?;
            if !tombstones.is_deleted(slot as u32) {
                budget.reserve(
                    id.len()
                        .saturating_mul(4)
                        .saturating_add(vector.indices().len().saturating_mul(64))
                        .saturating_add(256),
                )?;
                snapshot.push((id.clone(), vector.clone()));
            }
        }
        budget.work(
            snapshot
                .len()
                .saturating_mul(snapshot.len().max(1).ilog2() as usize + 1),
        )?;
        snapshot.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        Ok(snapshot)
    }

    pub(crate) fn vector_page(
        &self,
        cursor: &mut usize,
        limit: usize,
    ) -> Vec<(u32, String, EntangledHVec)> {
        let vectors = self.vectors.read();
        let tombstones = self.tombstones.read();
        let mut page = Vec::new();
        while *cursor < vectors.len() && page.len() < limit {
            let i = *cursor;
            *cursor += 1;
            if !tombstones.is_deleted(i as u32) {
                let (id, vector) = &vectors[i];
                page.push((i as u32, id.clone(), vector.clone()));
            }
        }
        page
    }

    pub(crate) fn id_page(&self, cursor: &mut usize, limit: usize) -> Vec<String> {
        let vectors = self.vectors.read();
        let tombstones = self.tombstones.read();
        let mut page = Vec::new();
        while *cursor < vectors.len() && page.len() < limit {
            let i = *cursor;
            *cursor += 1;
            if !tombstones.is_deleted(i as u32) {
                page.push(vectors[i].0.clone());
            }
        }
        page
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn vectors_read(
        &self,
    ) -> crate::core::sync::RwLockReadGuard<'_, Vec<(String, EntangledHVec)>> {
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
        confidence: 0.0,
    };

    let mut current = probe.clone();

    for _ in 0..max_iter {
        let overlaps = memory.overlap_scan(&current);
        if overlaps.is_empty() {
            return empty;
        }

        let attention = sparse_softmax(&overlaps, beta);

        let vecs = memory.vectors_read();
        let mut dim_scores: FastMap<u32, f64> = FastMap::default();
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
            if let Some((id, _)) = memory.get_by_idx(best.0) {
                return CleanupResult {
                    found: true,
                    id,
                    confidence: best.1,
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
    if let Some((id, _)) = memory.get_by_idx(best.0) {
        CleanupResult {
            found: true,
            id,
            confidence: best.1,
        }
    } else {
        empty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebuild_uses_only_live_vectors_and_never_restores_deleted_coordinates() {
        let memory = IndexedMemory::new(256, 3.0);
        let old = EntangledHVec::from_indices(vec![1, 2, 3], 256);
        let live_a = EntangledHVec::from_indices(vec![1, 2], 256);
        let live_b = EntangledHVec::from_indices(vec![2, 4], 256);
        memory.insert("ghost".into(), old.clone());
        memory.insert("a".into(), live_a.clone());
        memory.insert("b".into(), live_b.clone());
        assert!(memory.delete("ghost"));
        memory.rebuild_indices();
        let fresh = IndexedMemory::new(256, 3.0);
        fresh.insert("a".into(), live_a);
        fresh.insert("b".into(), live_b);
        fresh.rebuild_indices();
        let fingerprint = |memory: &IndexedMemory, query: &EntangledHVec| {
            let mut hits: Vec<_> = memory
                .overlap_scan(query)
                .into_iter()
                .map(|(slot, score)| (memory.get_by_idx(slot).unwrap().0, score.to_bits()))
                .collect();
            hits.sort_unstable();
            hits
        };
        let query = EntangledHVec::from_indices(vec![1, 2, 3, 4], 256);
        assert_eq!(fingerprint(&memory, &query), fingerprint(&fresh, &query));
        memory.insert("ghost".into(), EntangledHVec::from_indices(vec![5], 256));
        assert!(memory
            .overlap_scan(&EntangledHVec::from_indices(vec![3], 256))
            .is_empty());
        assert_eq!(
            memory
                .overlap_scan(&EntangledHVec::from_indices(vec![5], 256))
                .len(),
            1
        );
        memory.delete("a");
        memory.delete("b");
        memory.delete("ghost");
        memory.rebuild_indices();
        assert!(memory.overlap_scan(&query).is_empty());
        memory.insert("ghost".into(), old);
        assert_eq!(memory.overlap_scan(&query).len(), 1);
    }

    #[test]
    fn deleted_coordinates_do_not_match_a_reused_id_after_rebuild() {
        let memory = IndexedMemory::new(256, 3.0);
        memory.insert("fact".into(), EntangledHVec::from_indices(vec![1, 2], 256));
        memory.delete("fact");
        memory.rebuild_indices();
        memory.insert("fact".into(), EntangledHVec::from_indices(vec![3, 4], 256));
        assert!(memory
            .overlap_scan(&EntangledHVec::from_indices(vec![1], 256))
            .is_empty());
    }

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

    fn hundred_atoms() -> IndexedMemory {
        let mem = IndexedMemory::new(16384, 3.0);
        for i in 0..100u64 {
            mem.insert(
                format!("atom_{i}"),
                EntangledHVec::new_deterministic(16384, i),
            );
        }
        mem
    }

    #[test]
    fn test_hopfield_cleanup_exact() {
        let mem = hundred_atoms();
        let query = EntangledHVec::new_deterministic(16384, 42);
        let result = hopfield_cleanup(&query, &mem, 24.0, 64, 3);
        assert!(result.found);
        assert_eq!(result.id, "atom_42");
        assert!(result.confidence > 0.9);
    }

    #[test]
    fn test_hopfield_cleanup_noisy() {
        let mem = hundred_atoms();
        let mut noisy = EntangledHVec::new_deterministic(16384, 42)
            .indices()
            .to_vec();
        // Replace 25% of the active indices (16 of 64) with seeded noise.
        for (i, item) in noisy.iter_mut().take(16).enumerate() {
            *item = (crate::core::entangled::hash_u64(7, i as u64) % 16384) as u32;
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
