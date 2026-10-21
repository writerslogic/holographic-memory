// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Private scoring of dense low-bit codes in the clustered layout Tiptoe uses
//! for ranking (Henzinger, Dauterman, Corrigan-Gibbs, Zeldovich, SOSP 2023).
//!
//! The database is `N` codes of `d` small signed integers, partitioned into
//! `C` clusters of at most `S` codes. The server lays it out as a matrix `M`
//! with `S` rows and `C·d` columns: block `k` of row `i` holds the `i`-th code
//! of cluster `k`, or zeros. For each of `P` clusters the client picks locally,
//! it sends a Regev encryption (the parent module's scheme and parameters) of
//! the vector that holds the query code in that cluster's block and zeros
//! elsewhere; row `i` of `M` times that vector is the score of the `i`-th code
//! of the chosen cluster. The server sees `P` ciphertexts of `C·d` words and
//! reads every stored code for each, whichever clusters were chosen. `C = 1`
//! is exact scoring of the whole database.
//!
//! Every probe has its own secret: two probes under one secret would reveal
//! the difference of their plaintexts. A score decodes wrongly only when its
//! row's summed error reaches `Δ/2`. That error is `Σ_j M_ij·e_j` over
//! independent samples of a centred discrete Gaussian of parameter `σ`
//! truncated symmetrically, which is `σ`-sub-Gaussian, so
//! `P(|E_i| ≥ Δ/2) ≤ 2·exp(-(Δ/2)² / (2σ²·‖M_i‖²))`. [`DenseParams`] takes a
//! bound on `‖M_i‖²`, applies the union bound over the `S` scores of a probe
//! and rejects parameters above [`FAILURE_LOG2_BUDGET`]; [`DenseServer::new`]
//! rejects rows over the declared norm.

use std::sync::Arc;

use rand::CryptoRng;
use rayon::prelude::*;
use zeroize::Zeroize;

use super::{
    dot, expand_words, Answer, ErrorSampler, PrivateQueryError, Query, ERROR_SIGMA, ERROR_TAIL,
    LWE_DIM, MAX_DIM, SEED_LEN,
};

/// Largest accepted probability, as a power of two, that a probe decodes any
/// of its scores wrongly. SimplePIR's correctness target (Section 4.2).
pub const FAILURE_LOG2_BUDGET: f64 = -40.0;
/// Probes answered together in one pass over the database.
const LANES: usize = 8;
/// Database rows per inner pass of the batched answer kernel.
const ROW_BLOCK: usize = 4;
/// Database rows per task of the hint kernel.
const HINT_ROW_BLOCK: usize = 8;

/// Validated layout and code bounds. `n`, `q` and `σ` are the parent module's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseParams {
    dim: usize,
    clusters: usize,
    slots: usize,
    db_bound: u32,
    query_bound: u32,
    row_norm_sq: u64,
    modulus_bits: u32,
}

impl DenseParams {
    /// Largest possible `‖M_i‖²` when every entry is at the bound.
    pub fn worst_row_norm_sq(dim: usize, clusters: usize, db_bound: u32) -> u64 {
        (dim as u64)
            .saturating_mul(clusters as u64)
            .saturating_mul(u64::from(db_bound).pow(2))
    }

    /// `dim`: code length `d`. `clusters`, `slots`: `C` and `S`. Database code
    /// entries lie in `[-db_bound, db_bound]`, query entries in
    /// `[-query_bound, query_bound]`, and every database row (one code per
    /// cluster) has squared norm at most `row_norm_sq`.
    pub fn new(
        dim: usize,
        clusters: usize,
        slots: usize,
        db_bound: u32,
        query_bound: u32,
        row_norm_sq: u64,
    ) -> Result<Self, PrivateQueryError> {
        let samples = dim.saturating_mul(clusters);
        if dim == 0 || clusters == 0 || samples > MAX_DIM {
            return Err(PrivateQueryError::Dimension(samples));
        }
        if slots == 0 || db_bound == 0 || query_bound == 0 || db_bound > i8::MAX as u32 {
            return Err(PrivateQueryError::Layout(
                "slots and bounds must be positive, database bound at most 127",
            ));
        }
        if row_norm_sq == 0 || row_norm_sq > Self::worst_row_norm_sq(dim, clusters, db_bound) {
            return Err(PrivateQueryError::Layout(
                "row norm bound must be positive and attainable",
            ));
        }
        // Smallest power of two holding every score `|<x, u>| <= d·Bx·Bu` as a
        // centred residue.
        let max_score = u128::from(dim as u64) * u128::from(db_bound) * u128::from(query_bound);
        let modulus = (2 * max_score + 1).next_power_of_two();
        let modulus_bits = modulus.trailing_zeros();
        if modulus_bits >= 31 {
            return Err(PrivateQueryError::NoiseBudget);
        }
        let params = Self {
            dim,
            clusters,
            slots,
            db_bound,
            query_bound,
            row_norm_sq,
            modulus_bits,
        };
        if params.failure_log2() > FAILURE_LOG2_BUDGET {
            return Err(PrivateQueryError::NoiseBudget);
        }
        Ok(params)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn clusters(&self) -> usize {
        self.clusters
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    /// LWE samples per secret, `C·d`: the length of one probe.
    pub fn samples(&self) -> usize {
        self.dim * self.clusters
    }

    pub fn db_bound(&self) -> u32 {
        self.db_bound
    }

    pub fn query_bound(&self) -> u32 {
        self.query_bound
    }

    pub fn row_norm_sq(&self) -> u64 {
        self.row_norm_sq
    }

    pub fn plaintext_modulus(&self) -> u32 {
        1 << self.modulus_bits
    }

    /// `Δ = 2^32 / p`.
    pub fn delta(&self) -> u32 {
        1 << (32 - self.modulus_bits)
    }

    /// `log2` of the union bound `S · 2·exp(-(Δ/2)² / (2σ²·row_norm_sq))` on a
    /// probe decoding any score wrongly.
    pub fn failure_log2(&self) -> f64 {
        let half = f64::from(self.delta()) / 2.0;
        let variance = ERROR_SIGMA * ERROR_SIGMA * self.row_norm_sq as f64;
        (2.0 * self.slots as f64).log2() - half * half / (2.0 * variance) * std::f64::consts::LOG2_E
    }

    /// Whether decoding is certain: the truncated error satisfies
    /// `|E_i| <= ERROR_TAIL·‖M_i‖_1 <= ERROR_TAIL·C·d·Bx < Δ/2`.
    pub fn exact(&self) -> bool {
        2 * u64::from(ERROR_TAIL) * self.samples() as u64 * u64::from(self.db_bound)
            < u64::from(self.delta())
    }
}

/// The public matrix `A` in `Z_q^(C·d x n)`, expanded from a public seed.
pub struct DenseMatrix {
    params: DenseParams,
    seed: [u8; SEED_LEN],
    data: Vec<u32>,
}

impl DenseMatrix {
    pub fn expand(params: DenseParams, seed: [u8; SEED_LEN]) -> Arc<Self> {
        let data = expand_words(params.samples(), seed);
        Arc::new(Self { params, seed, data })
    }

    pub fn params(&self) -> DenseParams {
        self.params
    }

    pub fn byte_len(&self) -> usize {
        self.data.len() * size_of::<u32>()
    }
}

/// Client-side hint `H = M·A`, `S x n` words. Re-downloaded when the database
/// changes.
pub struct DenseHint {
    params: DenseParams,
    seed: [u8; SEED_LEN],
    data: Vec<u32>,
}

impl DenseHint {
    pub fn as_words(&self) -> &[u32] {
        &self.data
    }

    pub fn byte_len(&self) -> usize {
        self.data.len() * size_of::<u32>()
    }
}

/// Holds the plaintext database in the clustered layout and its hint.
pub struct DenseServer {
    params: DenseParams,
    rows: Vec<i8>,
    hint: DenseHint,
}

impl DenseServer {
    /// `codes` holds `dim`-entry codes back to back; `clusters[k]` lists the
    /// indices of cluster `k`'s codes, and a code's position in that list is
    /// its slot. A code may appear in several clusters.
    pub fn new(
        matrix: &DenseMatrix,
        codes: &[i8],
        clusters: &[Vec<u32>],
    ) -> Result<Self, PrivateQueryError> {
        let params = matrix.params;
        let (d, cols) = (params.dim, params.samples());
        if codes.len() % d != 0 {
            return Err(PrivateQueryError::Layout(
                "code buffer is not a whole number of codes",
            ));
        }
        if clusters.len() != params.clusters {
            return Err(PrivateQueryError::Length {
                what: "clusters",
                expected: params.clusters,
                found: clusters.len(),
            });
        }
        let bound = params.db_bound as i32;
        if codes.iter().any(|&x| i32::from(x).abs() > bound) {
            return Err(PrivateQueryError::EntryBound {
                what: "database code",
                bound: params.db_bound,
            });
        }
        let count = codes.len() / d;
        let mut rows = vec![0i8; params.slots * cols];
        for (k, members) in clusters.iter().enumerate() {
            if members.len() > params.slots {
                return Err(PrivateQueryError::Layout(
                    "cluster larger than the slot count",
                ));
            }
            for (slot, &index) in members.iter().enumerate() {
                let index = index as usize;
                if index >= count {
                    return Err(PrivateQueryError::Layout("cluster member out of range"));
                }
                let at = slot * cols + k * d;
                rows[at..at + d].copy_from_slice(&codes[index * d..(index + 1) * d]);
            }
        }
        for (row, values) in rows.chunks_exact(cols).enumerate() {
            let norm_sq: u64 = values
                .iter()
                .map(|&x| (i64::from(x) * i64::from(x)) as u64)
                .sum();
            if norm_sq > params.row_norm_sq {
                return Err(PrivateQueryError::RowNorm {
                    row,
                    norm_sq,
                    limit: params.row_norm_sq,
                });
            }
        }

        let mut hint = vec![0u32; params.slots * LWE_DIM];
        hint.par_chunks_mut(HINT_ROW_BLOCK * LWE_DIM)
            .zip(rows.par_chunks(HINT_ROW_BLOCK * cols))
            .for_each(|(out, block)| kernels::hint(block, cols, &matrix.data, out));

        Ok(Self {
            params,
            rows,
            hint: DenseHint {
                params,
                seed: matrix.seed,
                data: hint,
            },
        })
    }

    pub fn params(&self) -> DenseParams {
        self.params
    }

    pub fn hint(&self) -> &DenseHint {
        &self.hint
    }

    /// `M·c` for every probe of one query, in a single pass over the database
    /// per group of eight probes.
    pub fn answer(&self, queries: &[Query]) -> Result<Vec<Answer>, PrivateQueryError> {
        let cols = self.params.samples();
        if let Some(bad) = queries.iter().find(|q| q.0.len() != cols) {
            return Err(PrivateQueryError::Length {
                what: "query",
                expected: cols,
                found: bad.0.len(),
            });
        }
        let slots = self.params.slots;
        if let [query] = queries {
            let words = self
                .rows
                .par_chunks(cols)
                .with_min_len(256)
                .map(|row| kernels::dot_i8(row, &query.0))
                .collect();
            return Ok(vec![Answer(words)]);
        }
        let mut answers = Vec::with_capacity(queries.len());
        for group in queries.chunks(LANES) {
            let mut interleaved = vec![[0u32; LANES]; cols];
            for (lane, query) in group.iter().enumerate() {
                for (dst, &word) in interleaved.iter_mut().zip(&query.0) {
                    dst[lane] = word;
                }
            }
            let mut acc = vec![[0u32; LANES]; slots];
            acc.par_chunks_mut(ROW_BLOCK)
                .zip(self.rows.par_chunks(ROW_BLOCK * cols))
                .with_min_len(16)
                .for_each(|(out, block)| kernels::lanes(block, cols, &interleaved, out));
            for lane in 0..group.len() {
                answers.push(Answer(acc.iter().map(|a| a[lane]).collect()));
            }
        }
        Ok(answers)
    }
}

/// Fresh secrets and `A·s + e` for a fixed number of probes. Computing this
/// does not depend on the query, so a client can do it ahead of time.
pub struct PreparedQuery {
    params: DenseParams,
    seed: [u8; SEED_LEN],
    secrets: Box<[u32]>,
    bodies: Vec<Vec<u32>>,
}

impl PreparedQuery {
    pub fn probes(&self) -> usize {
        self.bodies.len()
    }

    /// Adds `Δ·code` to block `clusters[p]` of probe `p`. Uses the prepared
    /// secrets exactly once.
    pub fn encode(
        mut self,
        code: &[i8],
        clusters: &[usize],
    ) -> Result<(Vec<Query>, DenseQueryState), PrivateQueryError> {
        let params = self.params;
        let d = params.dim;
        if code.len() != d {
            return Err(PrivateQueryError::Length {
                what: "query code",
                expected: d,
                found: code.len(),
            });
        }
        let bound = params.query_bound as i32;
        if code.iter().any(|&x| i32::from(x).abs() > bound) {
            return Err(PrivateQueryError::EntryBound {
                what: "query code",
                bound: params.query_bound,
            });
        }
        if clusters.len() != self.bodies.len() {
            return Err(PrivateQueryError::Length {
                what: "clusters",
                expected: self.bodies.len(),
                found: clusters.len(),
            });
        }
        if clusters.iter().any(|&k| k >= params.clusters) {
            return Err(PrivateQueryError::Layout("cluster index out of range"));
        }
        let delta = params.delta();
        let mut bodies = std::mem::take(&mut self.bodies);
        for (body, &k) in bodies.iter_mut().zip(clusters) {
            for (word, &u) in body[k * d..(k + 1) * d].iter_mut().zip(code) {
                *word = word.wrapping_add(delta.wrapping_mul(u as u32));
            }
        }
        let state = DenseQueryState {
            params,
            seed: self.seed,
            secrets: std::mem::take(&mut self.secrets),
        };
        Ok((bodies.into_iter().map(Query).collect(), state))
    }
}

impl Drop for PreparedQuery {
    fn drop(&mut self) {
        self.secrets.zeroize();
    }
}

/// Per-query client state: one secret per probe. Not `Clone`; decoding
/// consumes it and the secrets are wiped on drop.
pub struct DenseQueryState {
    params: DenseParams,
    seed: [u8; SEED_LEN],
    secrets: Box<[u32]>,
}

impl DenseQueryState {
    /// Signed scores of every slot of each probed cluster; empty slots score 0.
    pub fn decode(
        self,
        hint: &DenseHint,
        answers: &[Answer],
    ) -> Result<Vec<Vec<i32>>, PrivateQueryError> {
        if hint.params != self.params || hint.seed != self.seed {
            return Err(PrivateQueryError::HintMismatch);
        }
        let probes = self.secrets.len() / LWE_DIM;
        if answers.len() != probes {
            return Err(PrivateQueryError::Length {
                what: "answers",
                expected: probes,
                found: answers.len(),
            });
        }
        let slots = self.params.slots;
        if let Some(bad) = answers.iter().find(|a| a.0.len() != slots) {
            return Err(PrivateQueryError::Length {
                what: "answer",
                expected: slots,
                found: bad.0.len(),
            });
        }
        let half = self.params.delta() / 2;
        let shift = 32 - self.params.modulus_bits;
        Ok(answers
            .iter()
            .zip(self.secrets.chunks_exact(LWE_DIM))
            .map(|(answer, secret)| {
                hint.data
                    .par_chunks(LWE_DIM)
                    .zip(answer.0.par_iter())
                    .with_min_len(64)
                    .map(|(h, &a)| {
                        // `a - H_i·s = Δ·score + E` with `|E| < Δ/2`: adding `Δ/2`
                        // and shifting floors to the score, sign included.
                        let noisy = a.wrapping_sub(kernels::dot_u32(h, secret));
                        (noisy.wrapping_add(half) as i32) >> shift
                    })
                    .collect()
            })
            .collect())
    }
}

impl Drop for DenseQueryState {
    fn drop(&mut self) {
        self.secrets.zeroize();
    }
}

/// Prepares probes and holds only public material.
pub struct DenseClient {
    matrix: Arc<DenseMatrix>,
    sampler: ErrorSampler,
}

impl DenseClient {
    pub fn new(matrix: Arc<DenseMatrix>) -> Self {
        Self {
            matrix,
            sampler: ErrorSampler::new(),
        }
    }

    /// Fresh secrets from the thread-local CSPRNG and `A·s + e` for `probes`
    /// probes.
    pub fn prepare(&self, probes: usize) -> PreparedQuery {
        self.prepare_with_rng(probes, &mut rand::rng())
    }

    fn prepare_with_rng<R: CryptoRng>(&self, probes: usize, rng: &mut R) -> PreparedQuery {
        let params = self.matrix.params;
        let cols = params.samples();
        let mut secrets = vec![0u32; probes * LWE_DIM].into_boxed_slice();
        for word in secrets.iter_mut() {
            *word = rng.next_u32();
        }
        let mut products = vec![0u32; cols * probes];
        if probes > 0 {
            products
                .par_chunks_mut(probes)
                .zip(self.matrix.data.par_chunks(LWE_DIM))
                .with_min_len(64)
                .for_each(|(out, a)| {
                    for (word, secret) in out.iter_mut().zip(secrets.chunks_exact(LWE_DIM)) {
                        *word = kernels::dot_u32(a, secret);
                    }
                });
        }
        let mut bodies = vec![vec![0u32; cols]; probes];
        for (col, words) in products.chunks_exact(probes.max(1)).enumerate() {
            for (body, &word) in bodies.iter_mut().zip(words) {
                body[col] = word.wrapping_add(self.sampler.sample(rng));
            }
        }
        PreparedQuery {
            params,
            seed: self.matrix.seed,
            secrets,
            bodies,
        }
    }
}

/// Wrapping `Z_q` kernels. Each is a plain loop that LLVM vectorises; on x86-64
/// the same loop is also compiled with AVX2 (`vpmulld`) and chosen at run time.
mod kernels {
    use super::{dot, LANES, LWE_DIM, ROW_BLOCK};

    #[inline(always)]
    fn dot_i8_body(row: &[i8], c: &[u32]) -> u32 {
        row.iter().zip(c).fold(0u32, |acc, (&m, &x)| {
            acc.wrapping_add((m as u32).wrapping_mul(x))
        })
    }

    #[inline(always)]
    fn lanes_body(rows: &[i8], cols: usize, q: &[[u32; LANES]], out: &mut [[u32; LANES]]) {
        if out.len() == ROW_BLOCK {
            let (r0, rest) = rows.split_at(cols);
            let (r1, rest) = rest.split_at(cols);
            let (r2, r3) = rest.split_at(cols);
            let mut acc = [[0u32; LANES]; ROW_BLOCK];
            for ((((qv, &m0), &m1), &m2), &m3) in q.iter().zip(r0).zip(r1).zip(r2).zip(r3) {
                let m = [m0 as u32, m1 as u32, m2 as u32, m3 as u32];
                for (a, &mr) in acc.iter_mut().zip(&m) {
                    for (lane, &x) in a.iter_mut().zip(qv) {
                        *lane = lane.wrapping_add(mr.wrapping_mul(x));
                    }
                }
            }
            out.copy_from_slice(&acc);
        } else {
            for (row, a) in rows.chunks_exact(cols).zip(out.iter_mut()) {
                *a = [0; LANES];
                for (qv, &m) in q.iter().zip(row) {
                    for (lane, &x) in a.iter_mut().zip(qv) {
                        *lane = lane.wrapping_add((m as u32).wrapping_mul(x));
                    }
                }
            }
        }
    }

    #[inline(always)]
    fn hint_body(rows: &[i8], cols: usize, a: &[u32], out: &mut [u32]) {
        for (col, a_row) in a.chunks_exact(LWE_DIM).enumerate() {
            for (row, h) in rows.chunks_exact(cols).zip(out.chunks_exact_mut(LWE_DIM)) {
                let m = row[col];
                if m != 0 {
                    let m = m as u32;
                    for (acc, &x) in h.iter_mut().zip(a_row) {
                        *acc = acc.wrapping_add(m.wrapping_mul(x));
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    mod avx2 {
        use super::*;

        #[target_feature(enable = "avx2")]
        pub(super) fn dot_i8(row: &[i8], c: &[u32]) -> u32 {
            dot_i8_body(row, c)
        }

        #[target_feature(enable = "avx2")]
        pub(super) fn dot_u32(a: &[u32], b: &[u32]) -> u32 {
            dot(a, b)
        }

        #[target_feature(enable = "avx2")]
        pub(super) fn lanes(
            rows: &[i8],
            cols: usize,
            q: &[[u32; LANES]],
            out: &mut [[u32; LANES]],
        ) {
            lanes_body(rows, cols, q, out)
        }

        #[target_feature(enable = "avx2")]
        pub(super) fn hint(rows: &[i8], cols: usize, a: &[u32], out: &mut [u32]) {
            hint_body(rows, cols, a, out)
        }
    }

    macro_rules! dispatch {
        ($avx2:path, $body:path, $($arg:expr),*) => {{
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: the CPU supports AVX2, checked just above.
                return unsafe { $avx2($($arg),*) };
            }
            $body($($arg),*)
        }};
    }

    /// `Σ_j row_j · c_j` over `Z_q`, the row read as signed.
    pub(super) fn dot_i8(row: &[i8], c: &[u32]) -> u32 {
        dispatch!(avx2::dot_i8, dot_i8_body, row, c)
    }

    pub(super) fn dot_u32(a: &[u32], b: &[u32]) -> u32 {
        dispatch!(avx2::dot_u32, dot, a, b)
    }

    /// `out[r][l] = Σ_j rows[r][j] · q[j][l]` for `out.len()` rows of `cols`.
    pub(super) fn lanes(rows: &[i8], cols: usize, q: &[[u32; LANES]], out: &mut [[u32; LANES]]) {
        dispatch!(avx2::lanes, lanes_body, rows, cols, q, out)
    }

    /// `out[r] = Σ_j rows[r][j] · A_j` for `out.len() / LWE_DIM` rows.
    pub(super) fn hint(rows: &[i8], cols: usize, a: &[u32], out: &mut [u32]) {
        dispatch!(avx2::hint, hint_body, rows, cols, a, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::{ChaCha20Rng, StdRng};
    use rand::{RngExt, SeedableRng};

    fn random_codes(rng: &mut StdRng, count: usize, dim: usize, bound: u32) -> Vec<i8> {
        let b = bound as i32;
        (0..count * dim)
            .map(|_| rng.random_range(-b..=b) as i8)
            .collect()
    }

    fn plain_score(codes: &[i8], dim: usize, index: u32, query: &[i8]) -> i32 {
        let i = index as usize;
        codes[i * dim..(i + 1) * dim]
            .iter()
            .zip(query)
            .map(|(&x, &u)| i32::from(x) * i32::from(u))
            .sum()
    }

    fn run(
        client: &DenseClient,
        server: &DenseServer,
        code: &[i8],
        clusters: &[usize],
    ) -> Result<Vec<Vec<i32>>, PrivateQueryError> {
        let (queries, state) = client.prepare(clusters.len()).encode(code, clusters)?;
        let answers = server.answer(&queries)?;
        state.decode(server.hint(), &answers)
    }

    #[test]
    fn decoded_scores_equal_plaintext_scores() {
        // (d, C, S, Bx, Bu, probes): whole-database scoring, more probes than
        // one lane group, entries at the widest bounds, and 1-bit codes.
        let cases = [
            (100, 1, 300, 1, 15, 1),
            (64, 16, 40, 3, 15, 9),
            (32, 8, 33, 15, 15, 3),
            (100, 64, 20, 1, 1, 2),
        ];
        for (case, &(d, c, s, bx, bu, probes)) in cases.iter().enumerate() {
            let mut rng = StdRng::seed_from_u64(case as u64);
            let count = c * s;
            let mut codes = random_codes(&mut rng, count, d, bx);
            // Codes 0 and 1 at +Bx and -Bx everywhere give the extreme scores
            // ±d·Bx·Bu against an all-+Bu query.
            codes[..d].fill(bx as i8);
            codes[d..2 * d].fill(-(bx as i8));
            // Uneven clusters, some short of `S`, so padded slots are exercised.
            let mut clusters: Vec<Vec<u32>> = vec![Vec::new(); c];
            for i in 0..count as u32 {
                let k = if i < 2 { 0 } else { rng.random_range(0..c) };
                if clusters[k].len() < s {
                    clusters[k].push(i);
                }
            }
            let params =
                DenseParams::new(d, c, s, bx, bu, DenseParams::worst_row_norm_sq(d, c, bx))
                    .unwrap();
            let matrix = DenseMatrix::expand(params, [case as u8 + 1; SEED_LEN]);
            let client = DenseClient::new(matrix.clone());
            let server = DenseServer::new(&matrix, &codes, &clusters).unwrap();
            for trial in 0..3 {
                let query: Vec<i8> = if trial == 0 {
                    vec![bu as i8; d]
                } else {
                    random_codes(&mut rng, 1, d, bu)
                };
                let probed: Vec<usize> = (0..probes)
                    .map(|p| if p == 0 { 0 } else { rng.random_range(0..c) })
                    .collect();
                let scores = run(&client, &server, &query, &probed).unwrap();
                for (p, &k) in probed.iter().enumerate() {
                    let expected: Vec<i32> = (0..s)
                        .map(|slot| {
                            clusters[k]
                                .get(slot)
                                .map_or(0, |&i| plain_score(&codes, d, i, &query))
                        })
                        .collect();
                    assert_eq!(scores[p], expected, "case {case} trial {trial} probe {p}");
                }
                if trial == 0 {
                    let max = (d as u32 * bx * bu) as i32;
                    assert_eq!(&scores[0][..2], &[max, -max], "case {case}");
                    assert!(max < params.plaintext_modulus() as i32 / 2);
                }
            }
        }
    }

    #[test]
    fn noise_has_the_deviation_the_failure_bound_assumes() {
        // Every row at the norm bound: entries ±3 over C·d = 2048 columns.
        let (d, c, s, bx) = (64, 32, 256, 3u32);
        let params =
            DenseParams::new(d, c, s, bx, 15, DenseParams::worst_row_norm_sq(d, c, bx)).unwrap();
        assert!(params.failure_log2() <= FAILURE_LOG2_BUDGET);
        let mut rng = StdRng::seed_from_u64(11);
        let codes: Vec<i8> = (0..c * s * d)
            .map(|_| if rng.random::<bool>() { 3 } else { -3 })
            .collect();
        let clusters: Vec<Vec<u32>> = (0..c)
            .map(|k| ((k * s) as u32..((k + 1) * s) as u32).collect())
            .collect();
        let matrix = DenseMatrix::expand(params, [7; SEED_LEN]);
        let client = DenseClient::new(matrix.clone());
        let server = DenseServer::new(&matrix, &codes, &clusters).unwrap();

        let probes = 8;
        let query = vec![0i8; d];
        let prepared = client.prepare_with_rng(probes, &mut ChaCha20Rng::from_seed([3; 32]));
        let (queries, state) = prepared.encode(&query, &vec![0; probes]).unwrap();
        let answers = server.answer(&queries).unwrap();
        // With a zero query, `a_i - H_i·s` is the error `E_i` itself.
        let mut errors = Vec::new();
        for (answer, secret) in answers.iter().zip(state.secrets.chunks_exact(LWE_DIM)) {
            for (h, &a) in server.hint.data.chunks_exact(LWE_DIM).zip(&answer.0) {
                errors.push(f64::from(a.wrapping_sub(dot(h, secret)) as i32));
            }
        }
        let n = errors.len() as f64;
        let sd = (errors.iter().map(|e| e * e).sum::<f64>() / n).sqrt();
        let expected = ERROR_SIGMA * (params.row_norm_sq() as f64).sqrt();
        assert!(
            (sd / expected - 1.0).abs() < 0.06,
            "sd {sd}, expected {expected}"
        );
        let max = errors.iter().fold(0f64, |m, e| m.max(e.abs()));
        assert!(max < f64::from(params.delta()) / 2.0 / 8.0, "max {max}");
        assert_eq!(
            state.decode(server.hint(), &answers).unwrap()[0],
            vec![0; s]
        );
    }

    #[test]
    fn rejects_out_of_bound_inputs_and_parameters() {
        assert_eq!(
            DenseParams::new(1024, 4096, 8, 1, 1, 1),
            Err(PrivateQueryError::Dimension(1 << 22))
        );
        // 4-bit codes on both sides over a million columns: the error swamps Δ.
        let worst = DenseParams::worst_row_norm_sq(1000, 1000, 15);
        assert_eq!(
            DenseParams::new(1000, 1000, 8, 15, 15, worst),
            Err(PrivateQueryError::NoiseBudget)
        );
        assert!(DenseParams::new(16, 2, 4, 1, 1, 33).is_err());

        let (d, c, s) = (16, 4, 3);
        // At most two all-ones codes (norm² 16 each) per row.
        let params = DenseParams::new(d, c, s, 2, 7, 32).unwrap();
        let matrix = DenseMatrix::expand(params, [5; SEED_LEN]);
        let client = DenseClient::new(matrix.clone());
        let codes = vec![1i8; d * 6];
        let ok: Vec<Vec<u32>> = vec![vec![0, 1, 4], vec![2, 5], vec![], vec![]];
        let server = DenseServer::new(&matrix, &codes, &ok).unwrap();

        let mut heavy = codes.clone();
        heavy[d] = 3;
        assert_eq!(
            DenseServer::new(&matrix, &heavy, &ok).err(),
            Some(PrivateQueryError::EntryBound {
                what: "database code",
                bound: 2
            })
        );
        // Row 0 would hold codes 0, 2 and 1: norm² 48 > 32.
        let dense = vec![vec![0], vec![2], vec![1], vec![]];
        assert!(matches!(
            DenseServer::new(&matrix, &codes, &dense),
            Err(PrivateQueryError::RowNorm { row: 0, .. })
        ));
        let oversized = vec![vec![0, 1, 2, 3], vec![], vec![], vec![]];
        assert!(DenseServer::new(&matrix, &codes, &oversized).is_err());
        assert!(DenseServer::new(&matrix, &codes, &[vec![9], vec![], vec![], vec![]]).is_err());

        let query = vec![7i8; d];
        assert!(client.prepare(1).encode(&query, &[c]).is_err());
        assert!(client.prepare(2).encode(&query, &[0]).is_err());
        assert!(client.prepare(1).encode(&[8; 16], &[0]).is_err());
        assert!(server
            .answer(&[Query::from_words(vec![0; d * c - 1])])
            .is_err());

        let other = DenseMatrix::expand(params, [6; SEED_LEN]);
        let other_server = DenseServer::new(&other, &codes, &ok).unwrap();
        let (queries, state) = client.prepare(1).encode(&query, &[0]).unwrap();
        let answers = server.answer(&queries).unwrap();
        assert_eq!(
            state.decode(other_server.hint(), &answers).err(),
            Some(PrivateQueryError::HintMismatch)
        );
        let (_, state) = client.prepare(2).encode(&query, &[0, 1]).unwrap();
        assert!(state.decode(server.hint(), &answers).is_err());
    }
}
