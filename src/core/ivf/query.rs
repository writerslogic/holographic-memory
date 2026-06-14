use std::cmp::Ordering;
use std::collections::BinaryHeap;

use anyhow::Result;

use super::IVFIndex;
use crate::core::entangled::EntangledHVec;
use crate::core::ivf::pq::PQEncoder;
use crate::core::types::RetrievalResult;

struct ScoredEntry {
    id: String,
    distance: u32,
}

impl Eq for ScoredEntry {}
impl PartialEq for ScoredEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Ord for ScoredEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.cmp(&other.distance)
    }
}
impl PartialOrd for ScoredEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl IVFIndex {
    /// IVF query: coarse search → n_probe clusters → ADC scan → top-k
    pub fn query(
        &self,
        query_vec: &EntangledHVec,
        k: usize,
        n_probe: usize,
    ) -> Result<Vec<RetrievalResult>> {
        if !self.trained {
            return Ok(Vec::new());
        }

        let projected = self.projector.project(query_vec);
        let probe_clusters = self.kmeans.top_clusters(&projected, n_probe);
        let adc_table = self.pq.build_adc_table(query_vec);

        let mut heap: BinaryHeap<ScoredEntry> = BinaryHeap::new();

        let lists = self
            .lists
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Inverted lists database not connected"))?;

        for cluster_id in probe_clusters {
            let entries = lists.read(cluster_id)?;
            for entry in entries {
                let dist = PQEncoder::adc_distance(&adc_table, &entry.pq_codes);
                heap.push(ScoredEntry {
                    id: entry.id,
                    distance: dist,
                });
                if heap.len() > k {
                    heap.pop();
                }
            }
        }

        // PQ distance = sum of symmetric-diff counts across 16 subvectors.
        // Max possible ≈ 2 * active_count where active_count = dim/256.
        let active_count = (self.dim / 256).max(1) as f64;
        let max_pq_dist = 2.0 * active_count;

        // into_sorted_vec() returns ascending distance = best (highest similarity) first
        let results: Vec<RetrievalResult> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|e| RetrievalResult {
                id: e.id,
                similarity: (1.0 - (e.distance as f64 / max_pq_dist)).clamp(0.0, 1.0),
            })
            .collect();

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::config::IVFConfig;
    use crate::core::entangled::EntangledHVec;
    use crate::core::ivf::IVFIndex;
    #[test]
    fn test_top1_accuracy() {
        let dim = 10000;
        let n = 200usize;
        let vectors: Vec<EntangledHVec> = (0..n)
            .map(|s| EntangledHVec::new_deterministic(dim, s as u64))
            .collect();
        let ids: Vec<String> = (0..n).map(|i| format!("vec_{}", i)).collect();

        let config = IVFConfig {
            enabled: true,
            n_clusters: 8,
            n_landmarks: 64,
            d_reduced: 16,
            n_probe: 8,
            auto_threshold: 0,
        };

        let index = IVFIndex::train(&vectors, &ids, dim, &config).unwrap();

        let results = index.query(&vectors[0], 10, config.n_probe).unwrap();
        assert!(!results.is_empty(), "Should return results");
        let found = results.iter().any(|r| r.id == "vec_0");
        assert!(
            found,
            "vec_0 should appear in top-10 results, got: {:?}",
            results.iter().map(|r| &r.id).collect::<Vec<_>>()
        );
        assert!(
            results[0].similarity > 0.5,
            "Top result similarity should be > 0.5, got {}",
            results[0].similarity
        );
    }
}
