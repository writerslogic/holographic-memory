// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Datasets for the kill test: a deterministic PRNG, a fixed-seed synthetic
//! Gaussian-cluster corpus, and loaders for the on-disk NFCorpus (MiniLM-L6,
//! 384-d) and glove-100-angular raw `f32` row-major files.
//!
//! All vectors are l2-normalized on the way in, so cosine = dot product.
//!
//! # Synthetic model
//!
//! `C` cluster centres `c_k = normalize(w * g)`, `g ~ N(0, I)`, with an
//! anisotropic spectrum `w_j = (j + 1)^(-alpha)` (alpha = 0 is isotropic; real
//! embeddings have a steeply decaying spectrum, which is what the PCA head
//! exploits). Document `i` belongs to cluster `i mod C`:
//! `x_i = normalize(c_k + spread * (w * g') / |w|)`, so `spread` is the expected
//! within-cluster perturbation norm relative to the unit centre
//! (`spread = 0.75` gives centre-to-member cosine ~ 0.8). Held-out queries are
//! fresh draws from random clusters.
//!
//! # Real distributions
//!
//! * NFCorpus: the 3,633 real corpus rows, padded to the requested size with
//!   on-manifold distractors `normalize(x_j + noise * g / sqrt(D))` around
//!   random real rows (`noise` is the perturbation norm), queried with the
//!   real queries.
//! * glove-100-angular: a seeded subset of the 1,183,514 train rows and of the
//!   10,000 test rows. Ground truth is recomputed on the subset.

use std::fs;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};

use crate::normalize_in_place;

/// xoshiro256** seeded through splitmix64; deterministic for a given seed.
pub struct Rng {
    s: [u64; 4],
    spare: Option<f32>,
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
        Self { s, spare: None }
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

    /// Standard normal (Marsaglia polar method, pairs cached).
    pub fn gaussian(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        loop {
            let u = 2.0 * self.uniform() - 1.0;
            let v = 2.0 * self.uniform() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let m = (-2.0 * s.ln() / s).sqrt();
                self.spare = Some((v * m) as f32);
                return (u * m) as f32;
            }
        }
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

/// A corpus plus held-out queries, both row-major and l2-normalized.
pub struct Dataset {
    /// Short name for reports and output files.
    pub name: String,
    /// Provenance string.
    pub source: String,
    /// Embedding dimension.
    pub dim: usize,
    /// Row-major `n_docs x dim`.
    pub docs: Vec<f32>,
    /// Number of corpus rows.
    pub n_docs: usize,
    /// Rows that come from the real corpus (equals `n_docs` for synthetic/glove).
    pub n_real_docs: usize,
    /// Row-major `n_queries x dim`.
    pub queries: Vec<f32>,
    /// Number of query rows.
    pub n_queries: usize,
}

impl Dataset {
    /// Document row `i`.
    pub fn doc(&self, i: usize) -> &[f32] {
        &self.docs[i * self.dim..(i + 1) * self.dim]
    }

    /// Query row `i`.
    pub fn query(&self, i: usize) -> &[f32] {
        &self.queries[i * self.dim..(i + 1) * self.dim]
    }
}

/// Parameters of the synthetic corpus.
pub struct SyntheticParams {
    /// Corpus size.
    pub n: usize,
    /// Dimension.
    pub dim: usize,
    /// Cluster count.
    pub clusters: usize,
    /// Held-out query count.
    pub n_queries: usize,
    /// Seed.
    pub seed: u64,
    /// Expected within-cluster perturbation norm (unit centres).
    pub spread: f32,
    /// Spectrum decay exponent `alpha` (0 = isotropic).
    pub spectrum_alpha: f32,
}

fn sample_point(
    rng: &mut Rng,
    centre: &[f32],
    w: &[f32],
    noise_scale: f32,
    out: &mut [f32],
) {
    for j in 0..out.len() {
        out[j] = centre[j] + noise_scale * w[j] * rng.gaussian();
    }
    normalize_in_place(out);
}

/// Fixed-seed anisotropic Gaussian-cluster corpus with held-out queries.
pub fn synthetic(p: &SyntheticParams) -> Dataset {
    let d = p.dim;
    let mut rng = Rng::new(p.seed ^ 0x5EED_0000_0000_0001);
    let w: Vec<f32> = (0..d)
        .map(|j| ((j + 1) as f32).powf(-p.spectrum_alpha))
        .collect();
    let wnorm = w.iter().map(|x| x * x).sum::<f32>().sqrt();
    let noise_scale = p.spread / wnorm;
    let mut centres = vec![0.0f32; p.clusters * d];
    for c in centres.chunks_exact_mut(d) {
        for j in 0..d {
            c[j] = w[j] * rng.gaussian();
        }
        normalize_in_place(c);
    }
    let mut docs = vec![0.0f32; p.n * d];
    for (i, row) in docs.chunks_exact_mut(d).enumerate() {
        let k = i % p.clusters;
        sample_point(&mut rng, &centres[k * d..(k + 1) * d], &w, noise_scale, row);
    }
    let mut queries = vec![0.0f32; p.n_queries * d];
    for row in queries.chunks_exact_mut(d) {
        let k = rng.below(p.clusters);
        sample_point(&mut rng, &centres[k * d..(k + 1) * d], &w, noise_scale, row);
    }
    Dataset {
        name: format!("synthetic_{}d_{}c", d, p.clusters),
        source: format!(
            "synthetic gaussian clusters: seed={} clusters={} spread={} spectrum_alpha={}",
            p.seed, p.clusters, p.spread, p.spectrum_alpha
        ),
        dim: d,
        docs,
        n_docs: p.n,
        n_real_docs: p.n,
        queries,
        n_queries: p.n_queries,
    }
}

/// Read a little-endian `f32` row-major file; the byte length must be a
/// multiple of `4 * dim`. Returns the rows and the row count.
pub fn load_f32_rows(path: &Path, dim: usize) -> Result<(Vec<f32>, usize)> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(dim > 0, "dim must be positive");
    ensure!(
        bytes.len() % (4 * dim) == 0,
        "{}: {} bytes is not a multiple of 4 * dim ({})",
        path.display(),
        bytes.len(),
        dim
    );
    let mut v = vec![0.0f32; bytes.len() / 4];
    for (dst, c) in v.iter_mut().zip(bytes.chunks_exact(4)) {
        *dst = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    let rows = v.len() / dim;
    Ok((v, rows))
}

/// Read the `dim` field of a dataset `meta.json`.
pub fn read_meta_dim(dir: &Path) -> Result<usize> {
    let path = dir.join("meta.json");
    let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let meta: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    match meta.get("dim").and_then(|v| v.as_u64()) {
        Some(d) if d > 0 => Ok(d as usize),
        _ => bail!("{}: missing positive integer field `dim`", path.display()),
    }
}

/// Write `normalize(src + noise * g / sqrt(D))` into `out`.
pub fn perturb(rng: &mut Rng, src: &[f32], noise: f32, out: &mut [f32]) {
    let s = noise / (src.len() as f32).sqrt();
    for j in 0..src.len() {
        out[j] = src[j] + s * rng.gaussian();
    }
    normalize_in_place(out);
}

/// NFCorpus MiniLM corpus padded to `n_total` rows with on-manifold
/// distractors; queries are the first `n_queries` real queries after a seeded
/// shuffle (all of them when `n_queries` exceeds the file).
pub fn nfcorpus(
    dir: &Path,
    n_total: usize,
    n_queries: usize,
    noise: f32,
    seed: u64,
) -> Result<Dataset> {
    let dim = read_meta_dim(dir)?;
    let (mut docs, n_real) = load_f32_rows(&dir.join("corpus.f32"), dim)?;
    let (all_queries, nq_all) = load_f32_rows(&dir.join("queries.f32"), dim)?;
    ensure!(n_real > 0 && nq_all > 0, "nfcorpus files are empty");
    ensure!(
        n_total >= n_real,
        "--n {} is smaller than the real corpus ({} rows)",
        n_total,
        n_real
    );
    for row in docs.chunks_exact_mut(dim) {
        normalize_in_place(row);
    }
    let mut rng = Rng::new(seed ^ 0x4E46_434F_5250_5553);
    docs.resize(n_total * dim, 0.0);
    let mut src = vec![0.0f32; dim];
    for i in n_real..n_total {
        let j = rng.below(n_real);
        src.copy_from_slice(&docs[j * dim..(j + 1) * dim]);
        perturb(&mut rng, &src, noise, &mut docs[i * dim..(i + 1) * dim]);
    }
    let picked = rng.sample_indices(nq_all, n_queries);
    let mut queries = vec![0.0f32; picked.len() * dim];
    for (dst, &qi) in queries.chunks_exact_mut(dim).zip(&picked) {
        let qi = qi as usize;
        dst.copy_from_slice(&all_queries[qi * dim..(qi + 1) * dim]);
        normalize_in_place(dst);
    }
    let n_queries = picked.len();
    Ok(Dataset {
        name: "nfcorpus_minilm".to_string(),
        source: format!(
            "{} (BEIR NFCorpus, MiniLM-L6 {}-d) + {} on-manifold distractors (noise={})",
            dir.display(),
            dim,
            n_total - n_real,
            noise
        ),
        dim,
        docs,
        n_docs: n_total,
        n_real_docs: n_real,
        queries,
        n_queries,
    })
}

/// Seeded `n`-row subset of glove-100-angular train plus `n_queries` test rows.
pub fn glove(dir: &Path, n: usize, n_queries: usize, seed: u64) -> Result<Dataset> {
    let dim = read_meta_dim(dir)?;
    let (train, n_train) = load_f32_rows(&dir.join("train.f32"), dim)?;
    ensure!(
        n <= n_train,
        "--n {} exceeds the glove train set ({} rows)",
        n,
        n_train
    );
    let mut rng = Rng::new(seed ^ 0x474C_4F56_4500_0000);
    let mut picked = rng.sample_indices(n_train, n);
    picked.sort_unstable();
    let mut docs = vec![0.0f32; n * dim];
    for (dst, &ri) in docs.chunks_exact_mut(dim).zip(&picked) {
        let ri = ri as usize;
        dst.copy_from_slice(&train[ri * dim..(ri + 1) * dim]);
        normalize_in_place(dst);
    }
    drop(train);
    let (test, n_test) = load_f32_rows(&dir.join("test.f32"), dim)?;
    let qpick = rng.sample_indices(n_test, n_queries);
    let mut queries = vec![0.0f32; qpick.len() * dim];
    for (dst, &qi) in queries.chunks_exact_mut(dim).zip(&qpick) {
        let qi = qi as usize;
        dst.copy_from_slice(&test[qi * dim..(qi + 1) * dim]);
        normalize_in_place(dst);
    }
    let n_queries = qpick.len();
    Ok(Dataset {
        name: "glove100".to_string(),
        source: format!(
            "{} (ann-benchmarks glove-100-angular, seeded {}-row subset of {})",
            dir.display(),
            n,
            n_train
        ),
        dim,
        docs,
        n_docs: n,
        n_real_docs: n,
        queries,
        n_queries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic_and_gaussian_has_unit_variance() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        assert_eq!(a.next_u64(), b.next_u64());
        let mut sum = 0.0f64;
        let mut sq = 0.0f64;
        let n = 20_000;
        for _ in 0..n {
            let g = a.gaussian() as f64;
            sum += g;
            sq += g * g;
        }
        let mean = sum / n as f64;
        let var = sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "var {var}");
    }

    #[test]
    fn synthetic_rows_are_unit_norm_and_clustered() {
        let ds = synthetic(&SyntheticParams {
            n: 400,
            dim: 64,
            clusters: 4,
            n_queries: 8,
            seed: 1,
            spread: 0.75,
            spectrum_alpha: 0.5,
        });
        for i in 0..ds.n_docs {
            let n = crate::norm_f32(ds.doc(i));
            assert!((n - 1.0).abs() < 1e-4);
        }
        let same = crate::dot_f32(ds.doc(0), ds.doc(4));
        let other = crate::dot_f32(ds.doc(0), ds.doc(1));
        assert!(same > other, "same-cluster {same} <= other-cluster {other}");
    }
}
