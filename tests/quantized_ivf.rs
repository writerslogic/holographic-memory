// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "experimental")]

use holographic_memory::core::qgraph::ivf::{IvfSearchParams, IvfSearcher, QuantizedIvf};
use rand::{rngs::StdRng, RngExt, SeedableRng};

fn hit_bits(hits: &[(u32, f32)]) -> Vec<(u32, u32)> {
    hits.iter()
        .map(|&(id, distance)| (id, distance.to_bits()))
        .collect()
}

fn random_fixture(dim: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<u32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers = (0..4)
        .flat_map(|_| unit((0..dim).map(|_| rng.random_range(-1.0..1.0)).collect()))
        .collect();
    let rows = (0..193)
        .flat_map(|row| {
            if row % 17 == 0 {
                vec![0.0; dim]
            } else {
                unit((0..dim).map(|_| rng.random_range(-1.0..1.0)).collect())
            }
        })
        .collect();
    (
        rows,
        centers,
        (0..193).map(|row| (row % 3) as u32).collect(),
    )
}

fn unit(mut row: Vec<f32>) -> Vec<f32> {
    let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut row {
        *x /= norm;
    }
    row
}

fn fixture(dim: usize) -> (Vec<f32>, Vec<f32>, Vec<u32>) {
    let centers: Vec<f32> = (0..3)
        .flat_map(|c| {
            unit(
                (0..dim)
                    .map(|i| ((i * 5 + c * 19 + 1) as f32).sin())
                    .collect(),
            )
        })
        .collect();
    let rows = (0..99)
        .flat_map(|r| {
            unit(
                (0..dim)
                    .map(|i| ((i * 7 + r * 13 + 2) as f32).cos())
                    .collect(),
            )
        })
        .collect();
    let assignments = (0..99).map(|i| (i % 2) as u32).collect();
    (rows, centers, assignments)
}

#[test]
fn all_lists_retain_original_ids_across_padding_and_residual_widths() {
    for dim in [31, 32, 33] {
        let (rows, centers, assignments) = fixture(dim);
        for bits in [4, 8] {
            for residual_bits in [0, 4, 5, 8] {
                let index =
                    QuantizedIvf::build(&rows, dim, &centers, &assignments, bits, residual_bits)
                        .unwrap();
                let bytes = index.to_bytes();
                assert_eq!(bytes.len(), index.serialized_bytes());
                let restored = QuantizedIvf::from_bytes(&bytes).unwrap();
                assert_eq!(bytes, restored.to_bytes());
                let mut search = IvfSearcher::new(&index);
                let mut loaded = IvfSearcher::new(&restored);
                for id in [0, 48, 98] {
                    let query = &rows[id * dim..(id + 1) * dim];
                    let params = IvfSearchParams {
                        nprobe: 3,
                        rerank: 99,
                    };
                    let hits = search.search(query, 99, params).unwrap();
                    assert_eq!(hits.len(), 99);
                    assert_eq!(search.scanned(), 99);
                    assert_eq!(hits, loaded.search(query, 99, params).unwrap());
                    let mut ids = hits.iter().map(|h| h.0).collect::<Vec<_>>();
                    ids.sort_unstable();
                    assert_eq!(ids, (0..99).collect::<Vec<_>>());
                    for &(row_id, distance) in &hits {
                        let row = &rows[row_id as usize * dim..(row_id as usize + 1) * dim];
                        let exact = -query.iter().zip(row).map(|(q, x)| q * x).sum::<f32>();
                        let cluster = assignments[row_id as usize] as usize;
                        let center = &centers[cluster * dim..(cluster + 1) * dim];
                        let max_relative = row
                            .iter()
                            .zip(center)
                            .map(|(x, c)| (x - c).abs())
                            .fold(0.0f32, f32::max);
                        let coarse_step = max_relative / if bits == 4 { 7.0 } else { 127.0 };
                        let max_error = match residual_bits {
                            0 => coarse_step / 2.0,
                            4 => coarse_step / 28.0,
                            5 => coarse_step / 60.0,
                            _ => coarse_step / 508.0,
                        };
                        let tolerance =
                            query.iter().map(|q| q.abs()).sum::<f32>() * max_error + 0.00001;
                        assert!(
                            (distance - exact).abs() < tolerance,
                            "{bits}/{residual_bits} dim={dim} id={row_id}: {distance} != {exact}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn centroid_scores_empty_lists_and_equal_vector_ties_are_consistent() {
    let rows = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let centers = [1.0, 0.0, -1.0, 0.0, 0.0, 1.0];
    let index = QuantizedIvf::build(&rows, 2, &centers, &[0, 1, 0, 0], 4, 4).unwrap();
    let mut search = IvfSearcher::new(&index);
    for rerank in [0, 1, 4] {
        let hits = search
            .search(&[1.0, 0.0], 4, IvfSearchParams { nprobe: 3, rerank })
            .unwrap();
        assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), [0, 2, 3, 1]);
    }
    assert!(search
        .search(
            &[0.0, 1.0],
            1,
            IvfSearchParams {
                nprobe: 1,
                rerank: 0
            }
        )
        .unwrap()
        .is_empty());
    assert_eq!(search.scanned(), 0);
    let extreme = search
        .search(
            &[f32::MAX, 0.0],
            4,
            IvfSearchParams {
                nprobe: 3,
                rerank: 4,
            },
        )
        .unwrap();
    let tiny = search
        .search(
            &[f32::from_bits(1), 0.0],
            4,
            IvfSearchParams {
                nprobe: 3,
                rerank: 4,
            },
        )
        .unwrap();
    assert_eq!(extreme, tiny);
}

#[test]
fn zero_database_rows_are_retained_and_zero_centroids_are_rejected() {
    for bits in [4, 8] {
        for residual_bits in [0, 4, 5, 8] {
            let rows = [0.0, 0.0, -1.0, 0.0, 1.0, 0.0];
            let index = QuantizedIvf::build(&rows, 2, &[1.0, 0.0], &[0, 0, 0], bits, residual_bits)
                .unwrap();
            let loaded = QuantizedIvf::from_bytes(&index.to_bytes()).unwrap();
            let hits = IvfSearcher::new(&loaded)
                .search(
                    &[1.0, 0.0],
                    3,
                    IvfSearchParams {
                        nprobe: 1,
                        rerank: 3,
                    },
                )
                .unwrap();
            assert_eq!(hits, [(2, -1.0), (0, 0.0), (1, 1.0)]);
        }
    }
    assert!(QuantizedIvf::build(&[1.0, 0.0], 2, &[0.0, 0.0], &[0], 8, 0).is_err());
}

#[test]
fn packed_five_bit_residual_minimum_is_rejected_on_reload() {
    let (rows, centers, assignments) = fixture(32);
    let index = QuantizedIvf::build(&rows, 32, &centers, &assignments, 4, 5).unwrap();
    let mut bytes = index.to_bytes();
    let residual = 24 + centers.len() * 4 + 4 * 4 + 99 * 12 + 99 * 16;
    bytes[residual] = (bytes[residual] & !31) | 16;
    repaired_crc(&mut bytes);
    assert!(QuantizedIvf::from_bytes(&bytes).is_err());
}

#[test]
fn malformed_shapes_values_and_search_limits_are_rejected() {
    let (rows, centers, assignments) = fixture(32);
    assert!(QuantizedIvf::build(&rows, 0, &centers, &assignments, 8, 0).is_err());
    for dim in [4095, 4096, 4097] {
        let row = unit(vec![1.0; dim]);
        assert_eq!(
            QuantizedIvf::build(&row, dim, &row, &[0], 8, 0).is_ok(),
            dim <= 4096
        );
    }
    assert!(
        QuantizedIvf::build(&rows[..rows.len() - 1], 32, &centers, &assignments, 8, 0).is_err()
    );
    assert!(QuantizedIvf::build(&rows, 32, &centers, &assignments[..98], 8, 0).is_err());
    for value in [2, 3, 4] {
        let mut assignments = assignments.clone();
        assignments[0] = value;
        assert_eq!(
            QuantizedIvf::build(&rows, 32, &centers, &assignments, 8, 0).is_ok(),
            value < 3
        );
    }
    for value in [f32::NAN, f32::INFINITY, 2.0] {
        let mut bad = rows.clone();
        bad[0] = value;
        assert!(QuantizedIvf::build(&bad, 32, &centers, &assignments, 8, 0).is_err());
    }
    assert!(QuantizedIvf::build(&rows, 32, &centers, &assignments, 7, 0).is_err());
    assert!(QuantizedIvf::build(&rows, 32, &centers, &assignments, 8, 7).is_err());
    let index = QuantizedIvf::build(&rows, 32, &centers, &assignments, 8, 4).unwrap();
    let mut search = IvfSearcher::new(&index);
    for nprobe in [0, 1, 3, 4] {
        assert_eq!(
            search
                .search(&rows[..32], 1, IvfSearchParams { nprobe, rerank: 0 })
                .is_ok(),
            (1..=3).contains(&nprobe)
        );
    }
    for limit in [98, 99, 100] {
        assert_eq!(
            search
                .search(
                    &rows[..32],
                    limit,
                    IvfSearchParams {
                        nprobe: 3,
                        rerank: 0
                    }
                )
                .is_ok(),
            limit <= 99
        );
        assert_eq!(
            search
                .search(
                    &rows[..32],
                    1,
                    IvfSearchParams {
                        nprobe: 3,
                        rerank: limit
                    }
                )
                .is_ok(),
            limit <= 99
        );
    }
    for query in [vec![0.0; 32], vec![f32::NAN; 32], vec![1.0; 31]] {
        assert!(search
            .search(
                &query,
                1,
                IvfSearchParams {
                    nprobe: 3,
                    rerank: 0
                }
            )
            .is_err());
    }
}

fn repaired_crc(bytes: &mut [u8]) {
    let end = bytes.len() - 4;
    let crc = crc32fast::hash(&bytes[..end]);
    bytes[end..].copy_from_slice(&crc.to_le_bytes());
}

#[test]
fn malformed_serialized_indexes_fail_before_search() {
    let (rows, centers, assignments) = fixture(32);
    for bits in [4, 8] {
        let index = QuantizedIvf::build(&rows, 32, &centers, &assignments, bits, 4).unwrap();
        let bytes = index.to_bytes();
        for length in [0, 23, bytes.len() - 1, bytes.len() + 1] {
            let mut bad = bytes.clone();
            bad.resize(length, 0);
            assert!(QuantizedIvf::from_bytes(&bad).is_err());
        }
        let offsets = 24 + centers.len() * 4;
        let ids = offsets + 4 * 4;
        let scales = ids + 99 * 4;
        let codes = scales + 99 * 8;
        for (at, value) in [
            (8, u32::MAX.to_le_bytes()),
            (12, 4097u32.to_le_bytes()),
            (24, f32::NAN.to_le_bytes()),
            (offsets, 1u32.to_le_bytes()),
            (offsets + 12, 100u32.to_le_bytes()),
            (ids, 99u32.to_le_bytes()),
            (ids + 4, 0u32.to_le_bytes()),
            (scales, f32::NAN.to_le_bytes()),
            (scales + 4, (-1.0f32).to_le_bytes()),
        ] {
            let mut bad = bytes.clone();
            bad[at..at + 4].copy_from_slice(&value);
            repaired_crc(&mut bad);
            assert!(
                QuantizedIvf::from_bytes(&bad).is_err(),
                "invalid field at {at}"
            );
        }
        let mut bad = bytes.clone();
        bad[codes] = if bits == 4 { 8 } else { 128 };
        repaired_crc(&mut bad);
        assert!(QuantizedIvf::from_bytes(&bad).is_err());
        let mut bad = bytes.clone();
        bad[scales] ^= 1;
        assert!(QuantizedIvf::from_bytes(&bad).is_err());
    }
}

#[test]
fn pruning_preserves_every_result_bit_across_padding_widths_seeds_and_limits() {
    for dim in [1, 31, 32, 33, 127, 128, 129] {
        for seed in [0, 913] {
            let (rows, centers, assignments) = random_fixture(dim, seed);
            let mut extreme = vec![0.0; dim];
            extreme[0] = -f32::MAX;
            let mut subnormal = vec![-0.0; dim];
            subnormal[0] = f32::from_bits(1);
            for bits in [4, 8] {
                for residual_bits in [0, 4, 5, 8] {
                    let index = QuantizedIvf::build(
                        &rows,
                        dim,
                        &centers,
                        &assignments,
                        bits,
                        residual_bits,
                    )
                    .unwrap();
                    let bytes = index.to_bytes();
                    let loaded = QuantizedIvf::from_bytes(&bytes).unwrap();
                    assert_eq!(loaded.to_bytes(), bytes);
                    assert_eq!(index.auxiliary_bytes(), loaded.auxiliary_bytes());
                    let mut pruned = IvfSearcher::new(&loaded);
                    let mut reference = IvfSearcher::new(&index);
                    for query in [
                        &rows[dim..2 * dim],
                        extreme.as_slice(),
                        subnormal.as_slice(),
                    ] {
                        for nprobe in [1, 2, 4] {
                            for (k, rerank) in
                                [(1, 0), (10, 0), (10, 1), (10, 64), (10, 65), (193, 193)]
                            {
                                let params = IvfSearchParams { nprobe, rerank };
                                let expected = reference.search_unpruned(query, k, params).unwrap();
                                let actual = pruned.search(query, k, params).unwrap();
                                assert_eq!(hit_bits(&actual), hit_bits(&expected), "dim={dim} seed={seed} bits={bits}/{residual_bits} k={k} params={params:?}");
                                assert_eq!(pruned.scanned(), reference.scanned());
                                assert_eq!(reference.evaluated(), reference.scanned());
                                assert!(pruned.evaluated() <= pruned.scanned());
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn separated_blocks_reduce_evaluations_and_preserve_serialized_bytes() {
    for dim in [31, 32, 33] {
        let mut axis = vec![0.0; dim];
        axis[0] = 1.0;
        let dense = unit(
            (0..dim)
                .map(|coordinate| (coordinate % 11) as f32 - 5.0)
                .collect(),
        );
        for center in [axis, dense] {
            for count in [63, 64, 65, 127, 128, 129, 512] {
                let rows: Vec<_> = (0..count)
                    .flat_map(|row| {
                        let mut value = center.clone();
                        if row >= 64 {
                            for coordinate in &mut value {
                                *coordinate = -*coordinate;
                            }
                        }
                        value
                    })
                    .collect();
                for bits in [4, 8] {
                    for residual_bits in [0, 4, 5, 8] {
                        let index = QuantizedIvf::build(
                            &rows,
                            dim,
                            &center,
                            &vec![0; count],
                            bits,
                            residual_bits,
                        )
                        .unwrap();
                        let bytes = index.to_bytes();
                        let loaded = QuantizedIvf::from_bytes(&bytes).unwrap();
                        let mut pruned = IvfSearcher::new(&loaded);
                        let mut reference = IvfSearcher::new(&index);
                        for rerank in [0, 1, count.min(64)] {
                            let params = IvfSearchParams { nprobe: 1, rerank };
                            assert_eq!(
                                hit_bits(&pruned.search(&center, 10, params).unwrap()),
                                hit_bits(&reference.search_unpruned(&center, 10, params).unwrap())
                            );
                            assert_eq!(pruned.scanned(), count);
                            let expected = if count < 128 { count } else { 64 + count % 64 };
                            assert_eq!(
                                pruned.evaluated(),
                                expected,
                                "count={count}, {bits}/{residual_bits}"
                            );
                            assert_eq!(reference.evaluated(), count);
                        }
                        assert_eq!(loaded.to_bytes(), bytes);
                        if count == 512 && bits == 4 && residual_bits == 5 {
                            println!("{{\"fixture\":\"separated\",\"dim\":{dim},\"nonzero_coordinates\":{},\"scanned\":512,\"evaluated\":64,\"auxiliary_bytes\":{},\"serialization_unchanged\":true}}", center.iter().filter(|&&x| x != 0.0).count(), index.auxiliary_bytes());
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn equal_distances_signed_zero_and_later_lower_ids_remain_candidates() {
    let centers = [1.0, 0.0, 1.0, 0.0];
    let rows = [1.0, 0.0].repeat(128);
    let assignments: Vec<_> = (0..128).map(|row| u32::from(row < 64)).collect();
    for bits in [4, 8] {
        for residual_bits in [0, 4, 5, 8] {
            let index =
                QuantizedIvf::build(&rows, 2, &centers, &assignments, bits, residual_bits).unwrap();
            let mut bytes = index.to_bytes();
            let scales = 24 + centers.len() * 4 + 3 * 4 + 128 * 4;
            for row in 0..64 {
                bytes[scales + row * 4..scales + (row + 1) * 4]
                    .copy_from_slice(&(-0.0f32).to_le_bytes());
            }
            repaired_crc(&mut bytes);
            let loaded = QuantizedIvf::from_bytes(&bytes).unwrap();
            for query in [[1.0, 0.0], [0.0, 1.0], [-0.0, 1.0]] {
                for rerank in [0, 1, 64, 65, 128] {
                    let params = IvfSearchParams { nprobe: 2, rerank };
                    let mut pruned = IvfSearcher::new(&loaded);
                    let mut reference = IvfSearcher::new(&loaded);
                    let actual = pruned.search(&query, 10, params).unwrap();
                    let expected = reference.search_unpruned(&query, 10, params).unwrap();
                    assert_eq!(hit_bits(&actual), hit_bits(&expected));
                    assert_eq!(
                        actual.iter().map(|hit| hit.0).collect::<Vec<_>>(),
                        (0..10).collect::<Vec<_>>()
                    );
                    assert_eq!(pruned.evaluated(), 128);
                }
            }
            assert_eq!(bytes, loaded.to_bytes());
        }
    }
}

#[test]
fn maximum_dimension_and_nonzero_loaded_padding_preserve_result_bits() {
    let count = 129;
    for dim in [4095, 4096] {
        let mut center = vec![0.0; dim];
        center[0] = 1.0;
        let rows: Vec<_> = (0..count)
            .flat_map(|row| {
                let mut value = center.clone();
                if row >= 64 {
                    value[0] = -1.0;
                }
                value
            })
            .collect();
        for bits in [4, 8] {
            for residual_bits in [0, 4, 5, 8] {
                let index =
                    QuantizedIvf::build(&rows, dim, &center, &vec![0; count], bits, residual_bits)
                        .unwrap();
                let mut bytes = index.to_bytes();
                if dim == 4095 {
                    let codes = 24 + dim * 4 + 8 + count * (8 + 4 * usize::from(residual_bits > 0));
                    let stride = 4096 * usize::from(bits) / 8;
                    for row in 0..count {
                        let last = codes + (row + 1) * stride - 1;
                        bytes[last] = if bits == 4 { bytes[last] | 0x70 } else { 127 };
                    }
                    repaired_crc(&mut bytes);
                }
                let loaded = QuantizedIvf::from_bytes(&bytes).unwrap();
                let mut pruned = IvfSearcher::new(&loaded);
                let mut reference = IvfSearcher::new(&loaded);
                let queries = [
                    center.clone(),
                    vec![f32::MAX; dim],
                    vec![f32::from_bits(1); dim],
                    (0..dim)
                        .map(|coordinate| {
                            if coordinate % 2 == 0 {
                                f32::MAX
                            } else {
                                -f32::MAX
                            }
                        })
                        .collect(),
                ];
                for query in queries {
                    for rerank in [0, 64, 65, 129] {
                        let params = IvfSearchParams { nprobe: 1, rerank };
                        assert_eq!(
                            hit_bits(&pruned.search(&query, 10, params).unwrap()),
                            hit_bits(&reference.search_unpruned(&query, 10, params).unwrap())
                        );
                        assert_eq!(pruned.scanned(), count);
                        assert!(pruned.evaluated() <= count);
                    }
                }
                assert_eq!(loaded.to_bytes(), bytes);
            }
        }
    }
}
