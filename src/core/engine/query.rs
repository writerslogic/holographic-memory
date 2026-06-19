// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::HmsCore;
use crate::core::entangled::EntangledHVec;
use crate::core::hopfield;
use crate::core::types::RetrievalResult;
use rayon::prelude::*;

impl HmsCore {
    /// Query the memory system for the k most similar vectors.
    pub fn query(&self, query_vec: &EntangledHVec, k: u32) -> Vec<RetrievalResult> {
        self.shards.read().query(query_vec, k, self.dimensions)
    }

    /// Energy-based associative retrieval using Hopfield-Fenchel-Young dynamics.
    ///
    /// Unlike `query` (which returns a fixed top-k by similarity), this uses
    /// sparse entmax attention to naturally determine how many results are
    /// relevant. Returns at most `max_results` patterns with non-zero
    /// Hopfield attention weight.
    pub fn query_hopfield(
        &self,
        query_vec: &EntangledHVec,
        max_results: u32,
    ) -> Vec<RetrievalResult> {
        let shards = self.shards.read();
        let patterns = shards.collect_all_patterns();
        let config = &self.config.hopfield;

        if config.max_iter > 1 {
            hopfield::hopfield_query_iterative(query_vec, &patterns, config, max_results as usize)
        } else {
            hopfield::hopfield_query(query_vec, &patterns, config, max_results as usize)
        }
    }

    /// Process multiple queries in parallel using rayon.
    pub fn query_batch(&self, queries: &[EntangledHVec], k: u32) -> Vec<Vec<RetrievalResult>> {
        queries.par_iter().map(|q| self.query(q, k)).collect()
    }

    /// Analyze components of a vector by finding its nearest neighbors.
    /// Filters by similarity threshold from QueryConfig (default 0.05).
    pub fn analyze_components(&self, vector: &EntangledHVec) -> Vec<RetrievalResult> {
        let cfg = &self.config.query;
        let neighbors = self.query(vector, cfg.component_max_neighbors);
        neighbors
            .into_iter()
            .filter(|r| r.similarity > cfg.component_similarity_threshold)
            .collect()
    }
}
