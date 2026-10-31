// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    io::{self, BufRead, Read, Write},
    time::Instant,
};
use writerslogic_warm::{
    core::{
        counting_bloom_memory::CountingBloomMemory, encoding::encode_text_internal,
        entangled::hash_u64,
    },
    EntangledHVec,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    id: String,
    text: String,
    dim: usize,
    seed: u64,
    rare_position: usize,
}

fn word_hash(text: &str, seed: u64) -> u64 {
    text.bytes().fold(seed, |s, b| hash_u64(s, u64::from(b)))
}

fn validate(dim: usize, bytes: usize, primitives: usize) -> Result<()> {
    ensure!((1..=65536).contains(&dim), "dimension outside bounds");
    ensure!(bytes <= 1_048_576, "input byte cap");
    ensure!(primitives <= 1_000_000, "primitive cap");
    ensure!(dim.checked_mul(6).is_some(), "count allocation overflow");
    Ok(())
}

fn primitives(text: &str, dim: usize) -> Result<Vec<Vec<EntangledHVec>>> {
    validate(dim, text.len(), 0)?;
    let text = text.trim();
    let chars: Vec<char> = text.chars().collect();
    let words: Vec<&str> = text.split_whitespace().collect();
    let total = (2..=5)
        .map(|n| chars.len().saturating_sub(n - 1))
        .sum::<usize>()
        + words.len()
        + words.len().saturating_sub(1);
    validate(dim, text.len(), total)?;
    let mut levels = Vec::new();
    for n in 2..=5 {
        let mut cache = BTreeMap::<String, EntangledHVec>::new();
        let mut vectors = Vec::new();
        for window in chars.windows(n) {
            let key: String = window.iter().collect();
            let vector = cache.entry(key).or_insert_with(|| {
                let mut v = EntangledHVec::from_indices(Vec::new(), dim);
                for (i, &c) in window.iter().enumerate() {
                    let atom = EntangledHVec::new_deterministic(
                        dim,
                        hash_u64(hash_u64(1, n as u64), c as u64),
                    );
                    v = v.bind(&atom.permute(i));
                }
                v
            });
            vectors.push(vector.clone());
        }
        levels.push(vectors);
    }
    levels.push(
        words
            .iter()
            .enumerate()
            .map(|(i, w)| EntangledHVec::new_deterministic(dim, word_hash(w, 2)).permute(i))
            .collect(),
    );
    levels.push(
        words
            .windows(2)
            .enumerate()
            .map(|(i, w)| {
                EntangledHVec::new_deterministic(dim, word_hash(w[0], 3))
                    .bind(&EntangledHVec::new_deterministic(dim, word_hash(w[1], 3)).permute(1))
                    .permute(i)
            })
            .collect(),
    );
    Ok(levels)
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn denominator(ns: &[usize]) -> Result<u128> {
    ns.iter().filter(|&&n| n > 0).try_fold(1_u128, |a, &n| {
        (a / gcd(a, n as u128))
            .checked_mul(n as u128)
            .context("normalization overflow")
    })
}

fn top(mut order: Vec<usize>, budget: usize) -> Vec<u32> {
    order.truncate(budget);
    let mut selected: Vec<u32> = order.into_iter().map(|i| i as u32).collect();
    selected.sort_unstable();
    selected
}

fn varints(indices: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut previous = 0;
    for &index in indices {
        let mut delta = index - previous;
        previous = index;
        while delta >= 128 {
            bytes.push((delta as u8 & 127) | 128);
            delta >>= 7;
        }
        bytes.push(delta as u8);
    }
    bytes
}

fn project(
    counts: &[Vec<u64>],
    ns: &[usize],
    dim: usize,
    seed: u64,
    text: &str,
) -> Result<Vec<Value>> {
    let common = denominator(ns)?;
    let mut sum = vec![0_u64; dim];
    let mut normalized = vec![0_u128; dim];
    let mut minimum = vec![u128::MAX; dim];
    let mut support_levels = vec![0_u64; dim];
    for (level, &n) in counts.iter().zip(ns) {
        if n == 0 {
            continue;
        }
        for (i, &c) in level.iter().enumerate() {
            sum[i] = sum[i].checked_add(c).context("count overflow")?;
            if c != 0 {
                let term = (common / n as u128)
                    .checked_mul(c as u128)
                    .context("score overflow")?;
                normalized[i] = normalized[i]
                    .checked_add(term)
                    .context("score sum overflow")?;
                minimum[i] = minimum[i].min(term);
                support_levels[i] += 1;
            }
        }
    }
    let support: Vec<usize> = (0..dim).filter(|&i| sum[i] > 0).collect();
    let k = (dim / 256).max(1);
    let mut rows = Vec::new();
    let budgets: std::collections::BTreeSet<_> = [(k / 2).max(1), k, 2 * k].into_iter().collect();
    for budget in budgets {
        for arm in ["UC", "F", "FN", "H", "R25", "R50", "N"] {
            let now = Instant::now();
            let mut order = support.clone();
            match arm {
                "UC" => order.sort_unstable_by_key(|&i| (std::cmp::Reverse(support_levels[i]), i)),
                "F" => order.sort_unstable_by_key(|&i| (std::cmp::Reverse(sum[i]), i)),
                "FN" => order.sort_unstable_by_key(|&i| (std::cmp::Reverse(normalized[i]), i)),
                "H" => order.sort_unstable_by_key(|&i| (hash_u64(i as u64, seed), i)),
                "R25" | "R50" => {
                    order.sort_unstable_by_key(|&i| (minimum[i], hash_u64(i as u64, seed), i));
                    order.truncate(budget / if arm == "R25" { 4 } else { 2 });
                    let mut chosen = vec![false; dim];
                    for &i in &order {
                        chosen[i] = true;
                    }
                    let mut rest: Vec<usize> =
                        support.iter().copied().filter(|&i| !chosen[i]).collect();
                    rest.sort_unstable_by_key(|&i| (std::cmp::Reverse(normalized[i]), i));
                    order.extend(rest);
                }
                "N" => {
                    order = (0..dim).collect();
                    let key = word_hash(text, seed);
                    order.sort_unstable_by_key(|&i| (hash_u64(i as u64, key), i));
                }
                _ => unreachable!(),
            }
            let indices = top(order, budget);
            rows.push(json!({"arm":arm,"budget":budget,"indices":indices,"selection_ns":now.elapsed().as_nanos()}));
        }
    }
    Ok(rows)
}

fn run(req: Request) -> Result<Value> {
    ensure!(req.id.len() <= 128, "id cap");
    ensure!(req.rare_position <= 1_000_000, "position cap");
    let now = Instant::now();
    let levels = primitives(&req.text, req.dim)?;
    let primitive_ns = now.elapsed().as_nanos();
    let now = Instant::now();
    let production = encode_text_internal(&req.text, req.dim);
    let production_ns = now.elapsed().as_nanos();
    let ns: Vec<usize> = levels.iter().map(Vec::len).collect();
    let now = Instant::now();
    let mut counts = vec![vec![0_u64; req.dim]; 6];
    let mut cb = CountingBloomMemory::new(req.dim);
    for (l, vectors) in levels.iter().enumerate() {
        for vector in vectors {
            cb.insert(vector);
            for &i in vector.indices() {
                counts[l][i as usize] = counts[l][i as usize]
                    .checked_add(1)
                    .context("count overflow")?;
            }
        }
    }
    let accumulation_ns = now.elapsed().as_nanos();
    let majority: Vec<Vec<u32>> = levels
        .iter()
        .map(|vs| EntangledHVec::bundle(vs).indices().to_vec())
        .collect();
    let level_unions: Vec<EntangledHVec> = levels
        .iter()
        .map(|vs| EntangledHVec::bundle_bloom(vs))
        .collect();
    let union = EntangledHVec::bundle_bloom(&level_unions);
    let now = Instant::now();
    let mut rows = project(&counts, &ns, req.dim, req.seed, &req.text)?;
    let projection_all_ns = now.elapsed().as_nanos();
    rows.push(json!({"arm":"P","budget":0,"indices":production.indices(),"selection_ns":null}));
    rows.push(json!({"arm":"U","budget":0,"indices":union.indices(),"selection_ns":null}));
    let rare: Vec<EntangledHVec> = (0..64)
        .map(|i| {
            EntangledHVec::new_deterministic(req.dim, word_hash(&format!("rare{i:04}"), 2))
                .permute(req.rare_position)
        })
        .collect();
    let cb_scores: Vec<f64> = rare.iter().map(|v| cb.membership_score(v)).collect();
    let operand = EntangledHVec::new_deterministic(req.dim, 909);
    for row in &mut rows {
        let indices: Vec<u32> = serde_json::from_value(row["indices"].clone())?;
        let v = EntangledHVec::from_indices(indices.clone(), req.dim);
        ensure!(
            v.bind(&operand).bind(&operand).indices() == indices,
            "XOR inverse"
        );
        ensure!(
            v.permute(7).permute(req.dim - 7 % req.dim).indices() == indices,
            "permutation inverse"
        );
        ensure!(
            v.bind_ordered(&operand)
                .unbind_ordered_left(&operand)
                .indices()
                == indices,
            "ordered inverse"
        );
        let serialized = serde_json::to_vec(&v)?;
        let restored: EntangledHVec = serde_json::from_slice(&serialized)?;
        ensure!(restored.indices() == indices, "serialization inverse");
        row["json_bytes"] = json!(serialized.len());
        row["json_lz4_bytes"] = json!(lz4_flex::compress_prepend_size(&serialized).len());
        row["delta_varint_bytes"] = json!(varints(&indices).len());
        row["rare_intersections"] = json!(rare
            .iter()
            .map(|r| {
                r.indices()
                    .iter()
                    .filter(|i| indices.binary_search(i).is_ok())
                    .count()
            })
            .collect::<Vec<_>>());
        row["operators_pass"] = json!(true);
    }
    let bytes: Vec<u8> = counts
        .iter()
        .flatten()
        .flat_map(|n| n.to_le_bytes())
        .collect();
    let compressed = lz4_flex::compress_prepend_size(&bytes);
    let hex: String = compressed.iter().map(|b| format!("{b:02x}")).collect();
    Ok(
        json!({"schema":1,"id":req.id,"dim":req.dim,"seed":req.seed,"text":req.text,
        "rare_position":req.rare_position,"ns":ns,"counts_lz4_hex":hex,"majority":majority,
        "rows":rows,"cb_scores":cb_scores,"cb_count_payload_bytes":req.dim*4,
        "primitive_ns":primitive_ns,"production_ns":production_ns,"accumulation_ns":accumulation_ns,
        "projection_all_ns":projection_all_ns,"count_payload_bytes":bytes.len()}),
    )
}

fn frequency(vectors: &[EntangledHVec], dim: usize) -> Vec<u32> {
    let mut counts = vec![0_u64; dim];
    for v in vectors {
        for &i in v.indices() {
            counts[i as usize] += 1;
        }
    }
    let mut support: Vec<usize> = (0..dim).filter(|&i| counts[i] > 0).collect();
    support.sort_unstable_by_key(|&i| (std::cmp::Reverse(counts[i]), i));
    top(support, (dim / 256).max(1))
}

fn operator_probe() -> Result<()> {
    let mut out = io::BufWriter::new(io::stdout().lock());
    for dim in [1024, 4096, 16384, 65536] {
        for seed in [11, 23, 47, 89, 179] {
            for n in [
                0, 1, 2, 3, 4, 7, 8, 9, 31, 32, 33, 127, 128, 129, 511, 512, 513,
            ] {
                for repeated in [false, true] {
                    let vs: Vec<_> = (0..n)
                        .map(|i| {
                            EntangledHVec::new_deterministic(
                                dim,
                                hash_u64(seed, if repeated { 0 } else { i as u64 }),
                            )
                        })
                        .collect();
                    let freq = frequency(&vs, dim);
                    let a = EntangledHVec::from_indices(frequency(&vs[..n / 2], dim), dim);
                    let b = EntangledHVec::from_indices(frequency(&vs[n / 2..], dim), dim);
                    let partition = frequency(&[a, b], dim);
                    let union = EntangledHVec::bundle_bloom(&vs);
                    let majority = EntangledHVec::bundle(&vs);
                    let row = json!({"dim":dim,"seed":seed,"n":n,"repeated":repeated,
                        "union":union.indices(),"majority":majority.indices(),"frequency":freq,
                        "partition_frequency":partition});
                    serde_json::to_writer(&mut out, &row)?;
                    writeln!(out)?;
                }
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    if std::env::args().nth(1).as_deref() == Some("--operators") {
        return operator_probe();
    }
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let mut output = io::BufWriter::new(io::stdout().lock());
    loop {
        let mut line = Vec::new();
        let n = input
            .by_ref()
            .take(6_400_001)
            .read_until(b'\n', &mut line)?;
        if n == 0 {
            break;
        }
        ensure!(
            n <= 6_400_000 && line.last() == Some(&b'\n'),
            "request line cap or truncation"
        );
        let request: Request = serde_json::from_slice(&line)?;
        let result = run(request)?;
        serde_json::to_writer(&mut output, &result)?;
        writeln!(output)?;
        output.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unchanged_encoder_survives_document_restart() -> Result<()> {
        use writerslogic_warm::{DocumentInput, SearchOptions, WarmCore};
        for dim in [1024, 16384] {
            let temp = tempfile::tempdir()?;
            let path = temp.path().display().to_string();
            let core = WarmCore::new(dim, Some(path.clone()), None)?;
            let text = "An independent document describes memory composition and durable restart.";
            let before = core.encode_text(text).indices().to_vec();
            let document: DocumentInput =
                serde_json::from_value(json!({"id":"golden", "text":text}))?;
            core.memorize_document(document)?;
            let query = serde_json::to_value(
                core.search_documents("durable restart", &SearchOptions::default())?,
            )?;
            drop(core);
            let core = WarmCore::new(dim, Some(path), None)?;
            assert_eq!(core.encode_text(text).indices(), before);
            assert_eq!(
                serde_json::to_value(
                    core.search_documents("durable restart", &SearchOptions::default())?
                )?,
                query
            );
        }
        Ok(())
    }

    #[test]
    fn bounded_inputs() {
        for dim in [65535, 65536] {
            assert!(validate(dim, 0, 0).is_ok());
        }
        for dim in [0, 65537] {
            assert!(validate(dim, 0, 0).is_err());
        }
        for n in [1_048_575, 1_048_576] {
            assert!(validate(256, n, 0).is_ok());
        }
        assert!(validate(256, 1_048_577, 0).is_err());
        for n in [999999, 1000000] {
            assert!(validate(256, 0, n).is_ok());
        }
        assert!(validate(256, 0, 1000001).is_err());
    }

    #[test]
    fn reserve_keeps_support_and_does_not_fill_empty() -> Result<()> {
        for size in [3, 4, 5] {
            let mut counts = vec![vec![0; 1024]; 6];
            counts[0][..size].fill(1);
            let rows = project(&counts, &[1, 0, 0, 0, 0, 0], 1024, 11, "fixture")?;
            for row in rows.iter().filter(|r| r["arm"] != "N") {
                let indices: Vec<u32> = serde_json::from_value(row["indices"].clone())?;
                assert_eq!(
                    indices.len(),
                    size.min(row["budget"].as_u64().unwrap() as usize)
                );
                assert!(indices.iter().all(|&i| i < size as u32));
            }
        }
        let rows = project(&vec![vec![0; 1024]; 6], &[0; 6], 1024, 11, "")?;
        assert!(rows
            .iter()
            .filter(|r| r["arm"] != "N")
            .all(|r| r["indices"].as_array().unwrap().is_empty()));
        Ok(())
    }
}
