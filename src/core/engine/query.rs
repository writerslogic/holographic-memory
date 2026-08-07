// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::HmsCore;
use crate::core::entangled::EntangledHVec;
use crate::core::hopfield;
use crate::core::types::{IndexStatus, QueryExplanation, RetrievalResult, StorageHealth};
use rayon::prelude::*;

impl HmsCore {
    /// Query the memory system for the k most similar vectors.
    pub fn query(&self, query_vec: &EntangledHVec, k: u32) -> Vec<RetrievalResult> {
        self.shards.read().query(query_vec, k, self.dimensions)
    }

    pub fn explain_query(&self, query_vec: &EntangledHVec, k: u32) -> QueryExplanation {
        self.shards
            .read()
            .explain_query(query_vec, k, self.dimensions)
    }

    pub fn index_status(&self) -> IndexStatus {
        let count = self.vector_count().min(u32::MAX as u64) as u32;
        let nsg = self.nsg_trained();
        let ivf = self.ivf_trained();
        let nsg_recommended = !nsg && count as usize >= self.config.nsg.auto_threshold;
        let ivf_recommended =
            self.config.ivf.enabled && !ivf && count as usize >= self.config.ivf.auto_threshold;
        let recommendation = if nsg_recommended && ivf_recommended {
            "train NSG and IVF indices"
        } else if nsg_recommended {
            "train the NSG index"
        } else if ivf_recommended {
            "train the IVF index"
        } else {
            "indices are current for the configured thresholds"
        };
        IndexStatus {
            vector_count: count,
            nsg_trained: nsg,
            ivf_trained: ivf,
            nsg_training_recommended: nsg_recommended,
            ivf_training_recommended: ivf_recommended,
            recommendation: recommendation.to_string(),
        }
    }

    pub fn storage_health(&self) -> StorageHealth {
        let stats = self.arena.stats();
        StorageHealth {
            format_version: stats.format_version,
            segment_count: stats.segment_count.min(u32::MAX as usize) as u32,
            used_bytes: stats.used_bytes as f64,
            capacity_bytes: stats.capacity_bytes as f64,
        }
    }

    /// Train any index whose configured threshold has been reached, then return
    /// the resulting lifecycle state. Calls are idempotent when indices are current.
    pub fn maintain_indices(&self) -> anyhow::Result<IndexStatus> {
        let before = self.index_status();
        if before.ivf_training_recommended {
            self.train_ivf()?;
        }
        if before.nsg_training_recommended {
            self.train_nsg()?;
        }
        Ok(self.index_status())
    }

    /// Flush persistent vector data to durable storage.
    pub fn flush(&self) -> anyhow::Result<()> {
        self.arena.flush()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_report_route_lifecycle_and_storage() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let hms = HmsCore::new(
            1024,
            Some(directory.path().to_string_lossy().into_owned()),
            None,
        )?;
        let query = EntangledHVec::new_deterministic(1024, 1);
        let explanation = hms.explain_query(&query, 5);
        assert_eq!(explanation.route, "brute_force");
        assert_eq!(hms.index_status().vector_count, 0);
        assert_eq!(hms.storage_health().format_version, 1);
        hms.flush()?;
        Ok(())
    }
}
