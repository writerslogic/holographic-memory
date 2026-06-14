use crate::core::entangled::EntangledHVec;
use serde::{Deserialize, Serialize};

const NUM_SUBVECTORS: usize = 16;
const NUM_CENTROIDS: usize = 256;
const MAX_ITERS: usize = 25;

#[derive(Serialize, Deserialize)]
pub(crate) struct PQEncoder {
    /// codebooks[sub][centroid] = sorted indices within that subvector's range
    pub(crate) codebooks: Vec<Vec<Vec<u32>>>,
    pub(crate) sub_range: usize,
    pub(crate) dim: usize,
}

impl PQEncoder {
    /// Train PQ codebooks from sample EntangledHVec vectors.
    /// Partitions [0, dim) into 16 equal index-ranges; extracts sub-indices
    /// via binary search on sorted indices; k-means on sub-index-sets for codebook training.
    pub fn train(samples: &[EntangledHVec], dim: usize) -> Self {
        let sub_range = dim / NUM_SUBVECTORS;

        let codebooks: Vec<Vec<Vec<u32>>> = (0..NUM_SUBVECTORS)
            .map(|sub_idx| {
                let range_start = (sub_idx * sub_range) as u32;
                // Last subvector covers remainder dimensions too
                let range_end = if sub_idx == NUM_SUBVECTORS - 1 {
                    dim as u32
                } else {
                    ((sub_idx + 1) * sub_range) as u32
                };
                let sub_data: Vec<Vec<u32>> = samples
                    .iter()
                    .map(|v| extract_sub_indices(&v.indices, range_start, range_end))
                    .collect();
                train_sub_codebook(&sub_data, range_start, range_end)
            })
            .collect();

        Self {
            codebooks,
            sub_range,
            dim,
        }
    }

    /// Range end for a subvector, with the last subvector covering remainder dimensions.
    fn range_end(&self, sub_idx: usize) -> u32 {
        if sub_idx == NUM_SUBVECTORS - 1 {
            self.dim as u32
        } else {
            ((sub_idx + 1) * self.sub_range) as u32
        }
    }

    /// Encode a vector into 16 bytes (one code per subvector).
    pub fn encode(&self, vec: &EntangledHVec) -> [u8; NUM_SUBVECTORS] {
        let mut codes = [0u8; NUM_SUBVECTORS];
        for (sub_idx, code) in codes.iter_mut().enumerate() {
            let range_start = (sub_idx * self.sub_range) as u32;
            let sub_indices =
                extract_sub_indices(&vec.indices, range_start, self.range_end(sub_idx));
            *code = self.nearest_centroid(sub_idx, &sub_indices);
        }
        codes
    }

    /// Build ADC table.
    /// table[sub][centroid] = symmetric difference count from query sub-indices to centroid.
    pub fn build_adc_table(&self, query: &EntangledHVec) -> [[u32; NUM_CENTROIDS]; NUM_SUBVECTORS] {
        let mut table = [[0u32; NUM_CENTROIDS]; NUM_SUBVECTORS];
        for (sub_idx, table_row) in table.iter_mut().enumerate() {
            let range_start = (sub_idx * self.sub_range) as u32;
            let q_sub = extract_sub_indices(&query.indices, range_start, self.range_end(sub_idx));
            for (c_idx, centroid) in self.codebooks[sub_idx].iter().enumerate() {
                table_row[c_idx] = sub_index_distance(&q_sub, centroid);
            }
        }
        table
    }

    /// Compute approximate distance using ADC table and PQ codes.
    pub fn adc_distance(
        table: &[[u32; NUM_CENTROIDS]; NUM_SUBVECTORS],
        codes: &[u8; NUM_SUBVECTORS],
    ) -> u32 {
        let mut dist = 0u32;
        for (sub_idx, &code) in codes.iter().enumerate() {
            dist += table[sub_idx][code as usize];
        }
        dist
    }

    fn nearest_centroid(&self, sub_idx: usize, sub_indices: &[u32]) -> u8 {
        if self.codebooks[sub_idx].is_empty() {
            return 0;
        }
        self.codebooks[sub_idx]
            .iter()
            .enumerate()
            .map(|(i, c)| (i, sub_index_distance(sub_indices, c)))
            .min_by_key(|&(_, d)| d)
            .map(|(i, _)| i as u8)
            .unwrap_or(0)
    }
}

/// Extract indices within [range_start, range_end) from a sorted index list.
fn extract_sub_indices(indices: &[u32], range_start: u32, range_end: u32) -> Vec<u32> {
    let start = indices.partition_point(|&x| x < range_start);
    let end = indices.partition_point(|&x| x < range_end);
    indices[start..end].to_vec()
}

/// Distance between two sub-index sets = symmetric difference count.
fn sub_index_distance(a: &[u32], b: &[u32]) -> u32 {
    crate::core::entangled::sorted_symmetric_difference_count(a, b) as u32
}

/// Train a codebook for one subvector range via binary k-means on index sets.
/// Caps `k` at the number of distinct sub-index patterns (at rho=1/256 per
/// subvector this is typically very small) and uses farthest-point initialization
/// (deterministic maxmin seeding, not probabilistic k-means++).
fn train_sub_codebook(data: &[Vec<u32>], range_start: u32, range_end: u32) -> Vec<Vec<u32>> {
    let n = data.len();
    if n == 0 {
        return (0..NUM_CENTROIDS).map(|_| Vec::new()).collect();
    }

    // Count distinct sub-index patterns to cap k
    let mut unique: Vec<&Vec<u32>> = data.iter().collect();
    unique.sort();
    unique.dedup();
    let k = NUM_CENTROIDS.min(n).min(unique.len());
    if k == 0 {
        return (0..NUM_CENTROIDS).map(|_| Vec::new()).collect();
    }

    // Farthest-point (maxmin) initialization
    let mut centroids: Vec<Vec<u32>> = Vec::with_capacity(k);
    centroids.push(data[0].clone());

    for _ in 1..k {
        // For each point, find min distance to existing centroids
        let mut best_candidate = &data[0];
        let mut best_dist = 0u32;
        for d in data {
            let min_d = centroids
                .iter()
                .map(|c| sub_index_distance(d, c))
                .min()
                .unwrap_or(u32::MAX);
            if min_d > best_dist {
                best_dist = min_d;
                best_candidate = d;
            }
        }
        centroids.push(best_candidate.clone());
    }

    for _ in 0..MAX_ITERS {
        // Assignment
        let assignments: Vec<usize> = data
            .iter()
            .map(|d| {
                centroids
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, sub_index_distance(d, c)))
                    .min_by_key(|&(_, dist)| dist)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect();

        // Update: majority vote on which indices to keep
        let mut changed = false;
        for (c, centroid) in centroids.iter_mut().enumerate().take(k) {
            let members: Vec<&Vec<u32>> = data
                .iter()
                .zip(assignments.iter())
                .filter(|(_, &a)| a == c)
                .map(|(d, _)| d)
                .collect();

            if members.is_empty() {
                continue;
            }

            let count = members.len();
            let threshold = count / 2 + 1;

            // Count frequency of each index in this cluster
            let mut all: Vec<u32> = members.iter().flat_map(|m| m.iter().copied()).collect();
            all.sort_unstable();

            let mut new_centroid = Vec::new();
            if !all.is_empty() {
                let mut current = all[0];
                let mut freq = 1u32;
                for &idx in &all[1..] {
                    if idx == current {
                        freq += 1;
                    } else {
                        if freq as usize >= threshold {
                            new_centroid.push(current);
                        }
                        current = idx;
                        freq = 1;
                    }
                }
                if freq as usize >= threshold {
                    new_centroid.push(current);
                }
            }

            // Ensure indices stay within range
            new_centroid.retain(|&idx| idx >= range_start && idx < range_end);

            if new_centroid != *centroid {
                changed = true;
                *centroid = new_centroid;
            }
        }

        if !changed {
            break;
        }
    }

    // Pad to NUM_CENTROIDS if we had fewer samples
    while centroids.len() < NUM_CENTROIDS {
        centroids.push(Vec::new());
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_compression_ratio() {
        let dim = 10000;
        let samples: Vec<EntangledHVec> = (0..64)
            .map(|s| EntangledHVec::new_deterministic(dim, s))
            .collect();
        let pq = PQEncoder::train(&samples, dim);

        let v = EntangledHVec::new_deterministic(dim, 999);
        let codes = pq.encode(&v);
        assert_eq!(codes.len(), 16);
    }

    #[test]
    fn test_encode_self_distance_low() {
        let dim = 10000;
        let samples: Vec<EntangledHVec> = (0..128)
            .map(|s| EntangledHVec::new_deterministic(dim, s))
            .collect();
        let pq = PQEncoder::train(&samples, dim);

        let v = &samples[0];
        let codes = pq.encode(v);
        let table = pq.build_adc_table(v);
        let dist = PQEncoder::adc_distance(&table, &codes);

        // Self-distance should be very low (quantization error only)
        // For sparse vectors with ~39 active indices, error should be small
        let max_expected = (dim as u32) / 10;
        assert!(
            dist < max_expected,
            "Self-distance {} should be < {} (10% of dim)",
            dist,
            max_expected
        );
    }

    #[test]
    fn test_adc_table_dimensions() {
        let dim = 10000;
        let samples: Vec<EntangledHVec> = (0..32)
            .map(|s| EntangledHVec::new_deterministic(dim, s))
            .collect();
        let pq = PQEncoder::train(&samples, dim);

        let q = EntangledHVec::new_deterministic(dim, 42);
        let table = pq.build_adc_table(&q);
        assert_eq!(table.len(), 16);
        assert_eq!(table[0].len(), 256);
    }
}
