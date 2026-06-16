// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use anyhow::Result;
use fxhash::{FxHashMap, FxHasher};
use parking_lot::RwLock;
use rayon::prelude::*;
use std::hash::Hasher;

use crate::core::entangled::EntangledHVec;
use crate::core::index::inverted::{Accumulator, SparseInvertedIndex};
use crate::core::ivf::IVFIndex;
use crate::core::nsg::NSGIndex;
use crate::core::types::RetrievalResult;

use super::router;

/// Per-shard state: vectors, registry, and indices.
/// Lock ordering within a shard: vectors -> ivf -> nsg.
pub(crate) struct Shard {
    pub vectors: RwLock<FxHashMap<String, EntangledHVec>>,
    pub registry: RwLock<Vec<String>>,
    pub inverted: RwLock<SparseInvertedIndex>,
    pub accumulator: parking_lot::Mutex<Accumulator>,
    pub nsg: RwLock<Option<NSGIndex>>,
    pub ivf: RwLock<Option<IVFIndex>>,
    pub vector_count: AtomicU64,
}

impl Shard {
    pub fn new(dimensions: usize) -> Self {
        let m = (dimensions / 256).max(1);
        Self {
            vectors: RwLock::new(FxHashMap::default()),
            registry: RwLock::new(Vec::new()),
            inverted: RwLock::new(SparseInvertedIndex::new(dimensions, m)),
            accumulator: parking_lot::Mutex::new(Accumulator::new(1024)),
            nsg: RwLock::new(None),
            ivf: RwLock::new(None),
            vector_count: AtomicU64::new(0),
        }
    }

    pub fn insert(&self, id: String, vector: EntangledHVec, dimensions: usize) -> Result<()> {
        // Phase 1: update vectors and registry (holds vectors -> registry locks)
        let is_replacement = {
            let mut vectors = self.vectors.write();
            let mut reg = self.registry.write();

            let is_replacement = vectors.contains_key(&id);
            vectors.insert(id.clone(), vector.clone());

            if !is_replacement {
                reg.push(id.clone());
                let count = reg.len() as u64;
                self.vector_count.store(count, AtomicOrdering::SeqCst);

                let mut inv = self.inverted.write();
                inv.add_doc((count - 1) as u32, &vector.indices);
            }
            is_replacement
        }; // vectors + registry locks released

        if is_replacement {
            self.rebuild_inverted_index(dimensions)?;
        }

        // Phase 2: update indices (acquires ivf -> nsg in order)
        if let Some(ref mut ivf) = *self.ivf.write() {
            ivf.insert(&id, &vector)?;
        }
        if let Some(ref mut nsg) = *self.nsg.write() {
            nsg.insert(&id, &vector)?;
        }
        Ok(())
    }

    pub fn remove(&self, id: &str, dimensions: usize) -> Result<bool> {
        let mut vectors = self.vectors.write();
        let mut reg = self.registry.write();

        if vectors.remove(id).is_none() {
            return Ok(false);
        }

        reg.retain(|r| r != id);
        self.vector_count
            .store(reg.len() as u64, AtomicOrdering::SeqCst);

        drop(reg);
        drop(vectors);

        self.rebuild_inverted_index(dimensions)?;
        Ok(true)
    }

    pub fn rebuild_inverted_index(&self, dimensions: usize) -> Result<()> {
        let vectors = self.vectors.read();
        let registry = self.registry.read();
        let mut inv = self.inverted.write();

        *inv = SparseInvertedIndex::new(dimensions, (dimensions / 256).max(1));

        for (i, id) in registry.iter().enumerate() {
            if let Some(vec) = vectors.get(id) {
                inv.add_doc(i as u32, &vec.indices);
            }
        }
        inv.finalize();

        let mut acc = self.accumulator.lock();
        *acc = Accumulator::new(registry.len().max(1024));

        Ok(())
    }

    pub fn load_all_vectors(&self) -> (Vec<String>, Vec<EntangledHVec>) {
        let vectors = self.vectors.read();
        let registry = self.registry.read();
        let mut ids = Vec::with_capacity(registry.len());
        let mut out_vectors = Vec::with_capacity(registry.len());

        for id in registry.iter() {
            if let Some(vec) = vectors.get(id) {
                ids.push(id.clone());
                out_vectors.push(vec.clone());
            }
        }

        (ids, out_vectors)
    }

    pub fn query(&self, query_vec: &EntangledHVec, k: u32, dimensions: usize) -> Vec<RetrievalResult> {
        let n = self.vector_count.load(AtomicOrdering::SeqCst) as usize;

        let planner = router::QueryPlanner::new(
            self.nsg_trained(),
            n > 0,
            self.ivf_trained(),
            n,
            dimensions,
        );
        let plan = planner.plan(query_vec, k);

        let vectors = self.vectors.read();

        match plan.route {
            router::IndexRoute::NSG => {
                if let Some(ref nsg) = *self.nsg.read() {
                    let results: Vec<RetrievalResult> = nsg
                        .query(query_vec, k as usize, plan.ef_search)
                        .into_iter()
                        .filter(|r| vectors.contains_key(&r.id))
                        .collect();
                    if !results.is_empty() {
                        return results;
                    }
                }
            }
            router::IndexRoute::Inverted => {
                let mut acc = self.accumulator.lock();
                let results = self
                    .inverted
                    .read()
                    .query(&query_vec.indices, k as usize, &mut acc);

                let reg = self.registry.read();
                let mapped: Vec<RetrievalResult> = results
                    .into_iter()
                    .filter_map(|r| {
                        let doc_id: u32 = r.id.parse().ok()?;
                        let id_str = reg.get(doc_id as usize)?;
                        if vectors.contains_key(id_str) {
                            Some(RetrievalResult {
                                id: id_str.clone(),
                                similarity: r.similarity,
                            })
                        } else {
                            None
                        }
                    })
                    .collect();

                if !mapped.is_empty() {
                    return mapped;
                }
            }
            router::IndexRoute::IVF => {
                if let Some(ref ivf) = *self.ivf.read() {
                    if let Ok(candidates) = ivf.query(query_vec, k as usize, plan.n_probe) {
                        let reranked = Self::rerank_by_exact_similarity(&vectors, query_vec, &candidates);
                        if !reranked.is_empty() {
                            return reranked;
                        }
                    }
                }
            }
            router::IndexRoute::BruteForce => {
                return Self::brute_force_scan(&vectors, query_vec, k as usize);
            }
        }

        Self::brute_force_scan(&vectors, query_vec, k as usize)
    }

    fn rerank_by_exact_similarity(
        vectors: &FxHashMap<String, EntangledHVec>,
        query_vec: &EntangledHVec,
        candidates: &[RetrievalResult],
    ) -> Vec<RetrievalResult> {
        let mut results: Vec<RetrievalResult> = candidates
            .iter()
            .filter_map(|c| {
                let vec = vectors.get(&c.id)?;
                Some(RetrievalResult {
                    id: c.id.clone(),
                    similarity: query_vec.similarity(vec),
                })
            })
            .collect();

        results.sort_unstable_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    fn brute_force_scan(
        vectors: &FxHashMap<String, EntangledHVec>,
        query_vec: &EntangledHVec,
        k: usize,
    ) -> Vec<RetrievalResult> {
        let mut heap: BinaryHeap<RetrievalResult> = BinaryHeap::with_capacity(k + 1);

        for (id, vec) in vectors.iter() {
            let sim = query_vec.similarity(vec);

            if heap.len() < k {
                heap.push(RetrievalResult {
                    similarity: sim,
                    id: id.clone(),
                });
            } else if let Some(top) = heap.peek() {
                if sim > top.similarity {
                    heap.pop();
                    heap.push(RetrievalResult {
                        similarity: sim,
                        id: id.clone(),
                    });
                }
            }
        }

        let mut results = heap.into_sorted_vec();
        results.reverse();
        results
    }

    pub fn nsg_trained(&self) -> bool {
        self.nsg.read().as_ref().is_some_and(|nsg| nsg.is_trained())
    }

    pub fn ivf_trained(&self) -> bool {
        self.ivf.read().as_ref().is_some_and(|ivf| ivf.is_trained())
    }

    pub fn count(&self) -> u64 {
        self.vector_count.load(AtomicOrdering::SeqCst)
    }
}

/// Multi-shard coordinator. Distributes vectors by consistent hash of ID.
pub(crate) struct ShardManager {
    pub shards: Vec<Shard>,
}

impl ShardManager {
    pub fn new(shard_count: usize, dimensions: usize) -> Self {
        assert!(shard_count >= 2, "ShardManager requires at least 2 shards");
        let shards = (0..shard_count).map(|_| Shard::new(dimensions)).collect();
        Self { shards }
    }

    pub fn shard_for(&self, id: &str) -> usize {
        let mut hasher = FxHasher::default();
        hasher.write(id.as_bytes());
        (hasher.finish() as usize) % self.shards.len()
    }

    pub fn shard(&self, id: &str) -> &Shard {
        &self.shards[self.shard_for(id)]
    }

    pub fn query(&self, query_vec: &EntangledHVec, k: u32, dimensions: usize) -> Vec<RetrievalResult> {
        let per_shard: Vec<Vec<RetrievalResult>> = self
            .shards
            .par_iter()
            .map(|shard| shard.query(query_vec, k, dimensions))
            .collect();

        let mut heap: BinaryHeap<RetrievalResult> = BinaryHeap::with_capacity(k as usize + 1);
        for results in per_shard {
            for r in results {
                heap.push(r);
                if heap.len() > k as usize {
                    heap.pop();
                }
            }
        }
        let mut merged = heap.into_sorted_vec();
        merged.reverse();
        merged
    }

    pub fn total_count(&self) -> u64 {
        self.shards.iter().map(|s| s.count()).sum()
    }

    pub fn nsg_trained(&self) -> bool {
        self.shards.iter().all(|s| s.nsg_trained())
    }

    pub fn ivf_trained(&self) -> bool {
        self.shards.iter().all(|s| s.ivf_trained())
    }
}

/// Wrapper that transparently handles single vs multi-shard operation.
pub(crate) enum ShardSet {
    Single(Box<Shard>),
    Multi(ShardManager),
}

impl ShardSet {
    pub fn insert(&self, id: String, vector: EntangledHVec, dimensions: usize) -> Result<()> {
        match self {
            ShardSet::Single(shard) => shard.insert(id, vector, dimensions),
            ShardSet::Multi(mgr) => mgr.shard(&id).insert(id, vector, dimensions),
        }
    }

    pub fn remove(&self, id: &str, dimensions: usize) -> Result<bool> {
        match self {
            ShardSet::Single(shard) => shard.remove(id, dimensions),
            ShardSet::Multi(mgr) => mgr.shard(id).remove(id, dimensions),
        }
    }

    pub fn query(&self, query_vec: &EntangledHVec, k: u32, dimensions: usize) -> Vec<RetrievalResult> {
        match self {
            ShardSet::Single(shard) => shard.query(query_vec, k, dimensions),
            ShardSet::Multi(mgr) => mgr.query(query_vec, k, dimensions),
        }
    }

    pub fn count(&self) -> u64 {
        match self {
            ShardSet::Single(shard) => shard.count(),
            ShardSet::Multi(mgr) => mgr.total_count(),
        }
    }

    pub fn nsg_trained(&self) -> bool {
        match self {
            ShardSet::Single(shard) => shard.nsg_trained(),
            ShardSet::Multi(mgr) => mgr.nsg_trained(),
        }
    }

    pub fn ivf_trained(&self) -> bool {
        match self {
            ShardSet::Single(shard) => shard.ivf_trained(),
            ShardSet::Multi(mgr) => mgr.ivf_trained(),
        }
    }

    pub fn for_each_shard<F: FnMut(&Shard)>(&self, mut f: F) {
        match self {
            ShardSet::Single(shard) => f(shard),
            ShardSet::Multi(mgr) => {
                for shard in &mgr.shards {
                    f(shard);
                }
            }
        }
    }

    pub fn try_for_each_shard<F: FnMut(&Shard) -> Result<()>>(&self, mut f: F) -> Result<()> {
        match self {
            ShardSet::Single(shard) => f(shard),
            ShardSet::Multi(mgr) => {
                for shard in &mgr.shards {
                    f(shard)?;
                }
                Ok(())
            }
        }
    }
}
