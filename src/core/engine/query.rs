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
    pub fn analyze_components(&self, vector: &EntangledHVec) -> Vec<RetrievalResult> {
        let neighbors = self.query(vector, 20);
        neighbors
            .into_iter()
            .filter(|r| r.similarity > 0.05)
            .collect()
    }
}
