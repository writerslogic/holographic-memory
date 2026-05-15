use rayon::prelude::*;
use std::cmp::Ordering;

use crate::core::config::ShardConfig;
use crate::core::entangled::EntangledHVec;
use crate::core::nsg::NSGIndex;
use crate::core::types::RetrievalResult;

pub trait Shard: Send + Sync {
    fn query(&self, query: &EntangledHVec, k: usize, ef_search: usize) -> Vec<RetrievalResult>;
    fn is_trained(&self) -> bool;
    fn vector_count(&self) -> usize;
}

impl Shard for NSGIndex {
    fn query(&self, query: &EntangledHVec, k: usize, ef_search: usize) -> Vec<RetrievalResult> {
        self.query(query, k, ef_search)
    }

    fn is_trained(&self) -> bool {
        self.is_trained()
    }

    fn vector_count(&self) -> usize {
        self.vectors.len()
    }
}

/// Multi-shard query coordinator. Currently used for test/development only;
/// production workloads use a single HmsCore instance. Shards are not
/// auto-populated — callers must add them explicitly via `add_shard`.
#[allow(dead_code)]
pub(crate) struct ShardManager {
    shards: Vec<Box<dyn Shard>>,
    #[allow(dead_code)]
    config: ShardConfig,
}

#[allow(dead_code)]
impl ShardManager {
    pub fn new(config: ShardConfig) -> Self {
        Self {
            shards: Vec::new(),
            config,
        }
    }

    pub fn add_shard(&mut self, shard: Box<dyn Shard>) {
        self.shards.push(shard);
    }

    pub fn is_trained(&self) -> bool {
        !self.shards.is_empty() && self.shards.iter().all(|s| s.is_trained())
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub fn total_vectors(&self) -> usize {
        self.shards.iter().map(|s| s.vector_count()).sum()
    }

    pub fn query(&self, query: &EntangledHVec, k: usize, ef_search: usize) -> Vec<RetrievalResult> {
        if self.shards.is_empty() {
            return vec![];
        }

        let mut all_results: Vec<RetrievalResult> = self
            .shards
            .par_iter()
            .flat_map(|shard| shard.query(query, k, ef_search))
            .collect();

        all_results.sort_unstable_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(Ordering::Equal)
        });

        // Deduplicate by ID, keeping highest similarity (already sorted desc)
        let mut seen = std::collections::HashSet::new();
        all_results.retain(|r| seen.insert(r.id.clone()));

        all_results.truncate(k);
        all_results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::NSGConfig;
    use crate::core::nsg::training;

    fn make_shard(seed_start: u64, n: usize) -> Box<dyn Shard> {
        let vectors: Vec<EntangledHVec> = (0..n)
            .map(|i| EntangledHVec::new_deterministic(1000, seed_start + i as u64))
            .collect();
        let ids: Vec<String> = (0..n).map(|i| format!("s{}_v{}", seed_start, i)).collect();
        let offsets: Vec<usize> = (0..n).map(|i| i * 100).collect();
        let config = NSGConfig {
            max_degree: 8,
            ef_search: 32,
            ef_construction: 16,
            auto_threshold: 0,
            seed: 42,
        };
        let index = training::train(&vectors, &ids, &offsets, 1000, &config).unwrap();
        Box::new(index)
    }

    #[test]
    fn parallel_query() {
        let mut mgr = ShardManager::new(ShardConfig::default());
        mgr.add_shard(make_shard(0, 20));
        mgr.add_shard(make_shard(100, 20));
        mgr.add_shard(make_shard(200, 20));

        assert!(mgr.is_trained());
        assert_eq!(mgr.shard_count(), 3);

        let query = EntangledHVec::new_deterministic(1000, 0);
        let results = mgr.query(&query, 5, 32);
        assert!(!results.is_empty(), "Should return results from shards");
    }

    #[test]
    fn merge_topk() {
        let mut mgr = ShardManager::new(ShardConfig::default());
        mgr.add_shard(make_shard(0, 30));
        mgr.add_shard(make_shard(100, 30));

        let query = EntangledHVec::new_deterministic(1000, 0);
        let results = mgr.query(&query, 3, 32);
        assert!(results.len() <= 3, "Should respect top-k limit");
        for w in results.windows(2) {
            assert!(w[0].similarity >= w[1].similarity, "Should be sorted desc");
        }
    }

    #[test]
    fn empty_shard_manager() {
        let mgr = ShardManager::new(ShardConfig::default());
        assert!(!mgr.is_trained());
        let query = EntangledHVec::new_deterministic(1000, 0);
        let results = mgr.query(&query, 5, 32);
        assert!(results.is_empty(), "Empty manager should return empty");
    }
}
