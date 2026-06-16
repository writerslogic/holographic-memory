// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use rayon::prelude::*;
use super::HmsCore;
use crate::core::entangled::EntangledHVec;
use crate::core::types::RetrievalResult;

impl HmsCore {
    /// Query the memory system for the k most similar vectors.
    pub fn query(&self, query_vec: &EntangledHVec, k: u32) -> Vec<RetrievalResult> {
        self.shards.read().query(query_vec, k, self.dimensions)
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
