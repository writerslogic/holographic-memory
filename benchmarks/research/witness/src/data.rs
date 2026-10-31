// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Loaders for the pre-converted ann-benchmarks layout under
//! `~/.cache/warm-bench/<name>/`:
//!
//! * `meta.json`: `dim`, `n_train`, `n_test`, `n_neighbors` (positive ints);
//! * `train.f32`, `test.f32`: little-endian `f32`, row-major, no header;
//! * `neighbors.i32`: little-endian `i32`, `n_test x n_neighbors`, the exact
//!   angular ground truth shipped with the dataset (used only to cross-check
//!   the ground truth this binary recomputes).
//!
//! Rows are l2-normalized on load so cosine = dot product; an all-zero row
//! stays zero. `train.f32` may be truncated to a prefix of `n` rows for smoke
//! runs (the prefix is deterministic, no sampling).
//!
//! Also a deterministic xoshiro256** PRNG (same construction as the sibling
//! `scan` crate, copied rather than depended upon) used for the SRHT signs,
//! the quantile sample and the query split.

use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};

use crate::normalize_in_place;

/// xoshiro256** seeded through splitmix64; deterministic for a given seed.
pub struct Rng {
    s: [u64; 4],
}

fn splitmix(z: &mut u64) -> u64 {
    *z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = *z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

impl Rng {
    /// Seed the generator.
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0u64; 4];
        for e in s.iter_mut() {
            *e = splitmix(&mut z);
        }
        Self { s }
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)` with 53 random bits.
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Uniform integer in `[0, n)`; `n` must be positive.
    pub fn below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        ((self.uniform() * n as f64) as usize).min(n - 1)
    }

    /// Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }

    /// `m` distinct indices from `[0, n)` in random order.
    pub fn sample_indices(&mut self, n: usize, m: usize) -> Vec<u32> {
        let mut idx: Vec<u32> = (0..n as u32).collect();
        self.shuffle(&mut idx);
        idx.truncate(m.min(n));
        idx
    }
}

/// The integer fields of `meta.json`.
#[derive(Debug, Clone, Copy)]
pub struct Meta {
    /// Embedding dimension.
    pub dim: usize,
    /// Rows in `train.f32`.
    pub n_train: usize,
    /// Rows in `test.f32` and `neighbors.i32`.
    pub n_test: usize,
    /// Columns in `neighbors.i32`.
    pub n_neighbors: usize,
}

/// Parse `meta.json`.
pub fn read_meta(dir: &Path) -> Result<Meta> {
    let path = dir.join("meta.json");
    let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let meta: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    let field = |key: &str| -> Result<usize> {
        match meta.get(key).and_then(|v| v.as_u64()) {
            Some(v) if v > 0 => Ok(v as usize),
            _ => bail!("{}: missing positive integer field `{}`", path.display(), key),
        }
    };
    Ok(Meta {
        dim: field("dim")?,
        n_train: field("n_train")?,
        n_test: field("n_test")?,
        n_neighbors: field("n_neighbors")?,
    })
}

/// Read the first `max_rows` rows (all rows when 0) of a little-endian `f32`
/// row-major file; the byte length must be a multiple of `4 * dim`.
pub fn load_f32_rows(path: &Path, dim: usize, max_rows: usize) -> Result<(Vec<f32>, usize)> {
    ensure!(dim > 0, "dim must be positive");
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let len = file
        .metadata()
        .with_context(|| format!("stat {}", path.display()))?
        .len() as usize;
    ensure!(
        len % (4 * dim) == 0,
        "{}: {} bytes is not a multiple of 4 * dim ({})",
        path.display(),
        len,
        dim
    );
    let avail = len / (4 * dim);
    let rows = if max_rows == 0 { avail } else { max_rows.min(avail) };
    ensure!(rows > 0, "{}: no rows", path.display());
    let mut bytes = vec![0u8; rows * dim * 4];
    file.read_exact(&mut bytes)
        .with_context(|| format!("read {} rows from {}", rows, path.display()))?;
    let mut v = vec![0.0f32; rows * dim];
    for (dst, c) in v.iter_mut().zip(bytes.chunks_exact(4)) {
        *dst = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    Ok((v, rows))
}

/// Read a little-endian `i32` row-major file with `cols` columns.
pub fn load_i32_rows(path: &Path, cols: usize) -> Result<(Vec<i32>, usize)> {
    ensure!(cols > 0, "cols must be positive");
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(
        bytes.len() % (4 * cols) == 0,
        "{}: {} bytes is not a multiple of 4 * cols ({})",
        path.display(),
        bytes.len(),
        cols
    );
    let mut v = vec![0i32; bytes.len() / 4];
    for (dst, c) in v.iter_mut().zip(bytes.chunks_exact(4)) {
        *dst = i32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    let rows = v.len() / cols;
    Ok((v, rows))
}

/// An ann-benchmarks dataset: normalized corpus rows, normalized test rows and
/// the shipped ground truth (if present).
pub struct AnnDataset {
    /// Short name for reports and output files.
    pub name: String,
    /// Directory the files came from.
    pub dir: String,
    /// Embedding dimension.
    pub dim: usize,
    /// Rows in `train.f32` (before any `--n` truncation).
    pub n_train_total: usize,
    /// Row-major `n_docs x dim`, l2-normalized.
    pub docs: Vec<f32>,
    /// Loaded corpus rows (`min(n, n_train_total)`).
    pub n_docs: usize,
    /// Row-major `n_test x dim`, l2-normalized.
    pub test: Vec<f32>,
    /// Test rows.
    pub n_test: usize,
    /// `neighbors.i32`, row-major `n_test x file_k`, when the file exists.
    pub file_neighbors: Option<Vec<i32>>,
    /// Columns of `file_neighbors` (0 when absent).
    pub file_k: usize,
}

/// Load a dataset directory; `n = 0` loads every train row.
pub fn load_ann(dir: &Path, name: &str, n: usize) -> Result<AnnDataset> {
    let meta = read_meta(dir)?;
    let (mut docs, n_docs) = load_f32_rows(&dir.join("train.f32"), meta.dim, n)?;
    for row in docs.chunks_exact_mut(meta.dim) {
        normalize_in_place(row);
    }
    let (mut test, n_test) = load_f32_rows(&dir.join("test.f32"), meta.dim, 0)?;
    for row in test.chunks_exact_mut(meta.dim) {
        normalize_in_place(row);
    }
    ensure!(
        n_test == meta.n_test,
        "{}: test.f32 has {} rows, meta.json says {}",
        dir.display(),
        n_test,
        meta.n_test
    );
    let nb_path = dir.join("neighbors.i32");
    let (file_neighbors, file_k) = if nb_path.is_file() {
        let (v, rows) = load_i32_rows(&nb_path, meta.n_neighbors)?;
        ensure!(
            rows == n_test,
            "{}: neighbors.i32 has {} rows, test.f32 has {}",
            dir.display(),
            rows,
            n_test
        );
        (Some(v), meta.n_neighbors)
    } else {
        (None, 0)
    };
    Ok(AnnDataset {
        name: name.to_string(),
        dir: dir.display().to_string(),
        dim: meta.dim,
        n_train_total: meta.n_train,
        docs,
        n_docs,
        test,
        n_test,
        file_neighbors,
        file_k,
    })
}

/// Disjoint seeded query sets: `report` is the headline set, `select` the
/// parameter-selection set. Both hold `test.f32` row indices.
pub struct QuerySplit {
    /// Reporting queries.
    pub report: Vec<u32>,
    /// Parameter-selection queries.
    pub select: Vec<u32>,
}

/// Draw `n_report + n_select` distinct test rows with a seeded shuffle.
pub fn split_queries(n_test: usize, n_report: usize, n_select: usize, seed: u64) -> Result<QuerySplit> {
    ensure!(n_report >= 1, "need at least one reporting query");
    ensure!(
        n_report + n_select <= n_test,
        "{} report + {} select queries exceed the {} test rows",
        n_report,
        n_select,
        n_test
    );
    let mut rng = Rng::new(seed ^ 0x5155_4552_5900_0000);
    let idx = rng.sample_indices(n_test, n_report + n_select);
    Ok(QuerySplit {
        report: idx[..n_report].to_vec(),
        select: idx[n_report..].to_vec(),
    })
}

/// Copy the rows `idx` of a row-major matrix into a new contiguous matrix.
pub fn gather_rows(src: &[f32], dim: usize, idx: &[u32]) -> Vec<f32> {
    let mut out = vec![0.0f32; idx.len() * dim];
    for (dst, &i) in out.chunks_exact_mut(dim).zip(idx) {
        let i = i as usize;
        dst.copy_from_slice(&src[i * dim..(i + 1) * dim]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic_and_samples_are_distinct() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        assert_eq!(a.next_u64(), b.next_u64());
        let mut s = a.sample_indices(100, 40);
        assert_eq!(s.len(), 40);
        s.sort_unstable();
        s.dedup();
        assert_eq!(s.len(), 40);
    }

    #[test]
    fn query_split_is_disjoint() {
        let q = split_queries(1000, 600, 300, 1).expect("split");
        assert_eq!(q.report.len(), 600);
        assert_eq!(q.select.len(), 300);
        assert!(q.report.iter().all(|r| !q.select.contains(r)));
    }
}
