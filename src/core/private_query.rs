// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experimental query-private similarity scoring (Regev / SimplePIR-style
//! linearly homomorphic encryption, single server).
//!
//! The server holds sparse binary codes in the clear and scores a client's
//! query code against every row without seeing the query. Only the query is
//! hidden, and only from an honest-but-curious server under the LWE
//! assumption. The stored data is NOT encrypted. Read `docs/PRIVATE-SEARCH.md`
//! for the threat model, the parameter sources and the limits before using
//! this module for anything.
//!
//! Protocol, with `q = 2^32` and `Δ = floor(q / p)`:
//!
//! - public: `A` in `Z_q^(D x n)`, expanded from a public seed
//! - offline: server computes the hint `H = X·A`; the client downloads it
//! - query: client samples a fresh `s`, `e` and sends `c = A·s + e + Δ·u`
//! - answer: server returns `a = X·c`
//! - decode: `score_i = round((a_i - H_i·s) / Δ) mod p`
//!
//! `a_i - H_i·s = Δ·|row_i ∩ u| + Σ_{j in row_i} e_j`, so decoding is exact
//! when `|row_i| < p` and `|row_i| · ERROR_TAIL < Δ / 2`. Both are enforced by
//! [`Params`] and [`Server::new`].

use std::sync::Arc;

use rand::rngs::ChaCha20Rng;
use rand::{CryptoRng, Rng, SeedableRng};
use rayon::prelude::*;
use zeroize::Zeroize;

use super::entangled::EntangledHVec;

/// LWE secret dimension `n`. SimplePIR (USENIX Security 2023), Section 4.2.
pub const LWE_DIM: usize = 1024;
/// Standard deviation of the discrete Gaussian error. SimplePIR, Section 4.2.
pub const ERROR_SIGMA: f64 = 6.4;
/// Hard bound on a single error sample, `ceil(12 · σ)`. Samples are drawn from
/// the Gaussian truncated to `[-ERROR_TAIL, ERROR_TAIL]`, which makes the
/// correctness bound deterministic.
pub const ERROR_TAIL: u32 = 77;
/// Largest supported code dimension. Each query publishes `D` LWE samples; the
/// SimplePIR parameter table covers up to `sqrt(2^42) = 2^21` samples.
pub const MAX_DIM: usize = 1 << 21;
/// Default plaintext modulus.
pub const DEFAULT_PLAINTEXT_MODULUS: u32 = 1024;
/// Length of the public seed that `A` is expanded from.
pub const SEED_LEN: usize = 32;

const Q: u64 = 1 << 32;

/// Errors from parameter validation and protocol-message checks.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PrivateQueryError {
    #[error("dimension {0} is outside 1..={MAX_DIM}")]
    Dimension(usize),
    #[error("plaintext modulus {0} leaves no usable row weight")]
    PlaintextModulus(u32),
    #[error("max row weight {requested} exceeds the correctness bound {bound}")]
    RowWeightBound { requested: usize, bound: usize },
    #[error("row {row} has weight {weight}, over the limit {limit}")]
    RowTooHeavy {
        row: usize,
        weight: usize,
        limit: usize,
    },
    #[error("code dimension {found} does not match the parameter dimension {expected}")]
    CodeDimension { expected: usize, found: usize },
    #[error("code indices must be strictly increasing and below the dimension")]
    MalformedCode,
    #[error("{what} has length {found}, expected {expected}")]
    Length {
        what: &'static str,
        expected: usize,
        found: usize,
    },
    #[error("hint was built for a different matrix or parameter set")]
    HintMismatch,
}

/// Validated scheme parameters. `n`, `q` and `σ` are fixed to the SimplePIR
/// values; only the dimension, the plaintext modulus and the row-weight limit
/// vary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params {
    dim: usize,
    plaintext_modulus: u32,
    max_row_weight: usize,
}

impl Params {
    /// Largest row weight that decodes correctly under plaintext modulus `p`:
    /// `min(p - 1, floor((Δ - 1) / (2 · ERROR_TAIL)))`.
    pub fn row_weight_bound(plaintext_modulus: u32) -> usize {
        if plaintext_modulus < 2 {
            return 0;
        }
        let delta = Q / u64::from(plaintext_modulus);
        let noise_bound = delta.saturating_sub(1) / (2 * u64::from(ERROR_TAIL));
        noise_bound.min(u64::from(plaintext_modulus) - 1) as usize
    }

    /// Parameters with an explicit plaintext modulus and row-weight limit.
    pub fn new(
        dim: usize,
        plaintext_modulus: u32,
        max_row_weight: usize,
    ) -> Result<Self, PrivateQueryError> {
        if dim == 0 || dim > MAX_DIM {
            return Err(PrivateQueryError::Dimension(dim));
        }
        let bound = Self::row_weight_bound(plaintext_modulus);
        if bound == 0 {
            return Err(PrivateQueryError::PlaintextModulus(plaintext_modulus));
        }
        if max_row_weight == 0 || max_row_weight > bound {
            return Err(PrivateQueryError::RowWeightBound {
                requested: max_row_weight,
                bound,
            });
        }
        Ok(Self {
            dim,
            plaintext_modulus,
            max_row_weight,
        })
    }

    /// Default modulus with the largest row weight it supports (1023).
    pub fn for_dim(dim: usize) -> Result<Self, PrivateQueryError> {
        Self::new(
            dim,
            DEFAULT_PLAINTEXT_MODULUS,
            Self::row_weight_bound(DEFAULT_PLAINTEXT_MODULUS),
        )
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn plaintext_modulus(&self) -> u32 {
        self.plaintext_modulus
    }

    pub fn max_row_weight(&self) -> usize {
        self.max_row_weight
    }

    /// Scaling factor `Δ = floor(2^32 / p)`.
    pub fn delta(&self) -> u32 {
        (Q / u64::from(self.plaintext_modulus)) as u32
    }
}

fn check_code(code: &EntangledHVec, dim: usize) -> Result<(), PrivateQueryError> {
    if code.dim != dim {
        return Err(PrivateQueryError::CodeDimension {
            expected: dim,
            found: code.dim,
        });
    }
    let idx = code.indices();
    let sorted = idx.windows(2).all(|w| w[0] < w[1]);
    let in_range = idx.last().is_none_or(|&last| (last as usize) < dim);
    if sorted && in_range {
        Ok(())
    } else {
        Err(PrivateQueryError::MalformedCode)
    }
}

fn dot(a: &[u32], b: &[u32]) -> u32 {
    a.iter()
        .zip(b)
        .fold(0u32, |acc, (&x, &y)| acc.wrapping_add(x.wrapping_mul(y)))
}

/// The public matrix `A` in `Z_q^(D x n)`, expanded from a public seed with
/// ChaCha20. Server and client must expand the same seed with the same
/// parameters. The expansion is tied to this crate's `rand` version and is not
/// a stable wire format.
pub struct PublicMatrix {
    params: Params,
    seed: [u8; SEED_LEN],
    data: Vec<u32>,
}

impl PublicMatrix {
    pub fn expand(params: Params, seed: [u8; SEED_LEN]) -> Arc<Self> {
        let mut data = vec![0u32; params.dim * LWE_DIM];
        // One independent stream per row keeps the expansion parallel and
        // independent of the thread count.
        data.par_chunks_mut(LWE_DIM)
            .enumerate()
            .for_each(|(row, chunk)| {
                let mut rng = ChaCha20Rng::from_seed(seed);
                rng.set_stream(row as u64);
                for word in chunk {
                    *word = rng.next_u32();
                }
            });
        Arc::new(Self { params, seed, data })
    }

    pub fn params(&self) -> Params {
        self.params
    }

    pub fn seed(&self) -> &[u8; SEED_LEN] {
        &self.seed
    }

    fn row(&self, j: usize) -> &[u32] {
        &self.data[j * LWE_DIM..(j + 1) * LWE_DIM]
    }
}

/// Client-side hint `H = X·A`, `N x n` words. Depends on the database, so it
/// must be re-downloaded when the database changes.
pub struct Hint {
    params: Params,
    seed: [u8; SEED_LEN],
    rows: usize,
    data: Vec<u32>,
}

impl Hint {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn as_words(&self) -> &[u32] {
        &self.data
    }

    pub fn byte_len(&self) -> usize {
        self.data.len() * size_of::<u32>()
    }
}

/// An encrypted query: `D` words of `Z_q`, independent of the query's weight.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Query(Vec<u32>);

impl Query {
    pub fn from_words(words: Vec<u32>) -> Self {
        Self(words)
    }

    pub fn as_words(&self) -> &[u32] {
        &self.0
    }

    pub fn byte_len(&self) -> usize {
        self.0.len() * size_of::<u32>()
    }
}

/// The server's answer: one word of `Z_q` per stored row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Answer(Vec<u32>);

impl Answer {
    pub fn from_words(words: Vec<u32>) -> Self {
        Self(words)
    }

    pub fn as_words(&self) -> &[u32] {
        &self.0
    }

    pub fn byte_len(&self) -> usize {
        self.0.len() * size_of::<u32>()
    }
}

/// Holds the plaintext database and its hint. Sees every stored code.
pub struct Server {
    params: Params,
    offsets: Vec<usize>,
    indices: Vec<u32>,
    hint: Hint,
}

impl Server {
    /// Validates every row against the correctness bound and builds the hint.
    pub fn new(matrix: &PublicMatrix, rows: &[EntangledHVec]) -> Result<Self, PrivateQueryError> {
        let params = matrix.params;
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        offsets.push(0);
        let mut indices = Vec::new();
        for (row, code) in rows.iter().enumerate() {
            check_code(code, params.dim)?;
            let weight = code.indices().len();
            if weight > params.max_row_weight {
                return Err(PrivateQueryError::RowTooHeavy {
                    row,
                    weight,
                    limit: params.max_row_weight,
                });
            }
            indices.extend_from_slice(code.indices());
            offsets.push(indices.len());
        }

        let mut data = vec![0u32; rows.len() * LWE_DIM];
        data.par_chunks_mut(LWE_DIM)
            .zip(offsets.par_windows(2))
            .for_each(|(out, span)| {
                for &j in &indices[span[0]..span[1]] {
                    for (acc, &a) in out.iter_mut().zip(matrix.row(j as usize)) {
                        *acc = acc.wrapping_add(a);
                    }
                }
            });

        Ok(Self {
            params,
            offsets,
            indices,
            hint: Hint {
                params,
                seed: matrix.seed,
                rows: rows.len(),
                data,
            },
        })
    }

    pub fn rows(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn hint(&self) -> &Hint {
        &self.hint
    }

    /// `a = X·c`: for each row, the sum of `c` over that row's active indices.
    pub fn answer(&self, query: &Query) -> Result<Answer, PrivateQueryError> {
        if query.0.len() != self.params.dim {
            return Err(PrivateQueryError::Length {
                what: "query",
                expected: self.params.dim,
                found: query.0.len(),
            });
        }
        let c = &query.0;
        let words = self
            .offsets
            .par_windows(2)
            .with_min_len(256)
            .map(|span| {
                self.indices[span[0]..span[1]]
                    .iter()
                    .fold(0u32, |acc, &j| acc.wrapping_add(c[j as usize]))
            })
            .collect();
        Ok(Answer(words))
    }
}

/// Table sampler for the discrete Gaussian truncated to `±ERROR_TAIL`.
/// Probabilities are computed in `f64`, so the table is accurate to about
/// 2^-53 per entry. Not constant-time.
struct ErrorSampler {
    /// `thresholds[k]` = `P(|x| <= k)` scaled to `2^64`.
    thresholds: Vec<u64>,
}

impl ErrorSampler {
    fn new() -> Self {
        let weights: Vec<f64> = (0..=ERROR_TAIL)
            .map(|k| {
                let x = f64::from(k);
                let rho = (-(x * x) / (2.0 * ERROR_SIGMA * ERROR_SIGMA)).exp();
                if k == 0 {
                    rho
                } else {
                    2.0 * rho
                }
            })
            .collect();
        let total: f64 = weights.iter().sum();
        let mut cumulative = 0.0;
        let mut thresholds: Vec<u64> = weights
            .iter()
            .map(|w| {
                cumulative += w;
                // `as` saturates, so a ratio that rounds to 1.0 maps to u64::MAX.
                (cumulative / total * 18_446_744_073_709_551_616.0) as u64
            })
            .collect();
        thresholds[ERROR_TAIL as usize] = u64::MAX;
        Self { thresholds }
    }

    /// One sample as an element of `Z_q` (negative values wrap).
    fn sample<R: CryptoRng>(&self, rng: &mut R) -> u32 {
        let r = rng.next_u64();
        let magnitude = self
            .thresholds
            .partition_point(|&t| t <= r)
            .min(ERROR_TAIL as usize) as u32;
        if rng.next_u32() & 1 == 1 {
            magnitude.wrapping_neg()
        } else {
            magnitude
        }
    }
}

/// Per-query client state: the LWE secret for exactly one query. Not `Clone`;
/// [`QueryState::decode`] consumes it and the secret is wiped on drop, so a
/// secret cannot be used for a second query.
pub struct QueryState {
    params: Params,
    seed: [u8; SEED_LEN],
    secret: Box<[u32]>,
}

impl QueryState {
    fn wipe(&mut self) {
        self.secret.zeroize();
    }

    /// Recovers `|row_i ∩ query|` for every row from the server's answer.
    pub fn decode(self, hint: &Hint, answer: &Answer) -> Result<Vec<u32>, PrivateQueryError> {
        if hint.params != self.params || hint.seed != self.seed {
            return Err(PrivateQueryError::HintMismatch);
        }
        if answer.0.len() != hint.rows {
            return Err(PrivateQueryError::Length {
                what: "answer",
                expected: hint.rows,
                found: answer.0.len(),
            });
        }
        let delta = self.params.delta();
        let p = self.params.plaintext_modulus;
        let secret: &[u32] = &self.secret;
        Ok(hint
            .data
            .par_chunks(LWE_DIM)
            .zip(answer.0.par_iter())
            .map(|(h, &a)| {
                let noisy = a.wrapping_sub(dot(h, secret));
                (noisy.wrapping_add(delta / 2) / delta) % p
            })
            .collect())
    }
}

impl Drop for QueryState {
    fn drop(&mut self) {
        self.wipe();
    }
}

impl std::fmt::Debug for QueryState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryState").finish_non_exhaustive()
    }
}

/// Encrypts queries and decodes answers. Holds only public material.
pub struct Client {
    matrix: Arc<PublicMatrix>,
    sampler: ErrorSampler,
}

impl Client {
    pub fn new(matrix: Arc<PublicMatrix>) -> Self {
        Self {
            matrix,
            sampler: ErrorSampler::new(),
        }
    }

    /// Encrypts `code` under a fresh secret drawn from the thread-local CSPRNG.
    pub fn query(&self, code: &EntangledHVec) -> Result<(Query, QueryState), PrivateQueryError> {
        self.query_with_rng(code, &mut rand::rng())
    }

    fn query_with_rng<R: CryptoRng>(
        &self,
        code: &EntangledHVec,
        rng: &mut R,
    ) -> Result<(Query, QueryState), PrivateQueryError> {
        let params = self.matrix.params;
        check_code(code, params.dim)?;

        let mut state = QueryState {
            params,
            seed: self.matrix.seed,
            secret: vec![0u32; LWE_DIM].into_boxed_slice(),
        };
        for word in state.secret.iter_mut() {
            *word = rng.next_u32();
        }

        let secret: &[u32] = &state.secret;
        let mut c: Vec<u32> = self
            .matrix
            .data
            .par_chunks(LWE_DIM)
            .map(|row| dot(row, secret))
            .collect();
        // The error is added in place and never stored separately: knowing `e`
        // turns the query into linear equations in the secret.
        for word in c.iter_mut() {
            *word = word.wrapping_add(self.sampler.sample(rng));
        }
        let delta = params.delta();
        for &j in code.indices() {
            c[j as usize] = c[j as usize].wrapping_add(delta);
        }
        Ok((Query(c), state))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::intersection::sparse_intersection_count;
    use rand::rngs::StdRng;
    use rand::RngExt;

    const DIM: usize = 16384;

    fn random_code(rng: &mut StdRng, dim: usize, weight: usize) -> EntangledHVec {
        let mut idx: Vec<u32> = rand::seq::index::sample(rng, dim, weight)
            .into_iter()
            .map(|i| i as u32)
            .collect();
        idx.sort_unstable();
        EntangledHVec::from_indices(idx, dim)
    }

    /// A row sharing a random number of indices with `query`, so the expected
    /// scores cover the whole range instead of clustering at zero.
    fn correlated_row(rng: &mut StdRng, query: &EntangledHVec, weight: usize) -> EntangledHVec {
        let shared = rng.random_range(0..=weight.min(query.indices().len()));
        let mut idx: Vec<u32> = rand::seq::index::sample(rng, query.indices().len(), shared)
            .into_iter()
            .map(|i| query.indices()[i])
            .collect();
        while idx.len() < weight {
            let j = rng.random_range(0..query.dim as u32);
            if !idx.contains(&j) {
                idx.push(j);
            }
        }
        idx.sort_unstable();
        EntangledHVec::from_indices(idx, query.dim)
    }

    fn plaintext_scores(rows: &[EntangledHVec], query: &EntangledHVec) -> Vec<u32> {
        rows.iter()
            .map(|r| sparse_intersection_count(r.indices(), query.indices()) as u32)
            .collect()
    }

    fn private_scores(
        client: &Client,
        server: &Server,
        query: &EntangledHVec,
    ) -> Result<Vec<u32>, PrivateQueryError> {
        let (q, state) = client.query(query)?;
        let answer = server.answer(&q)?;
        state.decode(server.hint(), &answer)
    }

    #[test]
    fn decoded_scores_equal_plaintext_intersections_across_seeds() {
        let params = Params::for_dim(DIM).unwrap();
        for matrix_seed in [3u8, 201] {
            let matrix = PublicMatrix::expand(params, [matrix_seed; SEED_LEN]);
            let client = Client::new(matrix.clone());
            for data_seed in 0..4u64 {
                let mut rng = StdRng::seed_from_u64(data_seed * 7919 + u64::from(matrix_seed));
                let query = random_code(&mut rng, DIM, 64);
                let rows: Vec<EntangledHVec> = (0..2000)
                    .map(|i| {
                        if i % 2 == 0 {
                            correlated_row(&mut rng, &query, 64)
                        } else {
                            random_code(&mut rng, DIM, 64)
                        }
                    })
                    .collect();
                let server = Server::new(&matrix, &rows).unwrap();
                let expected = plaintext_scores(&rows, &query);
                assert!(expected.iter().any(|&s| s > 32) && expected.contains(&0));
                assert_eq!(
                    private_scores(&client, &server, &query).unwrap(),
                    expected,
                    "matrix seed {matrix_seed}, data seed {data_seed}"
                );
                let other = random_code(&mut rng, DIM, 64);
                assert_eq!(
                    private_scores(&client, &server, &other).unwrap(),
                    plaintext_scores(&rows, &other)
                );
            }
        }
    }

    #[test]
    fn correct_at_maximum_row_weight_and_rejects_one_more() {
        // (plaintext modulus, expected bound): the first is limited by p - 1,
        // the second by the noise term 2·w·ERROR_TAIL < Δ.
        for (p, bound) in [(DEFAULT_PLAINTEXT_MODULUS, 1023usize), (65536, 425)] {
            assert_eq!(Params::row_weight_bound(p), bound);
            assert_eq!(
                Params::new(DIM, p, bound + 1),
                Err(PrivateQueryError::RowWeightBound {
                    requested: bound + 1,
                    bound
                })
            );
            let params = Params::new(DIM, p, bound).unwrap();
            let matrix = PublicMatrix::expand(params, [9; SEED_LEN]);
            let client = Client::new(matrix.clone());
            let mut rng = StdRng::seed_from_u64(u64::from(p));

            let query = random_code(&mut rng, DIM, 4096);
            let mut rows: Vec<EntangledHVec> = (0..200)
                .map(|_| correlated_row(&mut rng, &query, bound))
                .collect();
            // Largest possible score: a full-weight row contained in the query.
            rows.push(EntangledHVec::from_indices(
                query.indices()[..bound].to_vec(),
                DIM,
            ));
            let server = Server::new(&matrix, &rows).unwrap();
            let expected = plaintext_scores(&rows, &query);
            assert_eq!(*expected.last().unwrap() as usize, bound);
            assert_eq!(private_scores(&client, &server, &query).unwrap(), expected);

            rows.push(random_code(&mut rng, DIM, bound + 1));
            assert_eq!(
                Server::new(&matrix, &rows).err(),
                Some(PrivateQueryError::RowTooHeavy {
                    row: rows.len() - 1,
                    weight: bound + 1,
                    limit: bound
                })
            );
        }
    }

    #[test]
    fn parameter_and_input_validation() {
        assert_eq!(Params::for_dim(0), Err(PrivateQueryError::Dimension(0)));
        assert!(Params::for_dim(MAX_DIM).is_ok());
        assert_eq!(
            Params::for_dim(MAX_DIM + 1),
            Err(PrivateQueryError::Dimension(MAX_DIM + 1))
        );
        assert_eq!(
            Params::new(DIM, 1, 1),
            Err(PrivateQueryError::PlaintextModulus(1))
        );
        // Δ = 1 leaves no room for any error.
        assert_eq!(
            Params::new(DIM, u32::MAX, 1),
            Err(PrivateQueryError::PlaintextModulus(u32::MAX))
        );
        assert!(Params::new(DIM, 1024, 0).is_err());

        let params = Params::for_dim(4096).unwrap();
        let matrix = PublicMatrix::expand(params, [1; SEED_LEN]);
        let client = Client::new(matrix.clone());
        let wrong_dim = EntangledHVec::from_indices(vec![1, 2], 8192);
        assert_eq!(
            client.query(&wrong_dim).err(),
            Some(PrivateQueryError::CodeDimension {
                expected: 4096,
                found: 8192
            })
        );
        for bad in [vec![5u32, 5], vec![7, 3], vec![4096]] {
            let code = EntangledHVec {
                dim: 4096,
                indices: bad,
            };
            assert_eq!(
                client.query(&code).err(),
                Some(PrivateQueryError::MalformedCode)
            );
            assert_eq!(
                Server::new(&matrix, std::slice::from_ref(&code)).err(),
                Some(PrivateQueryError::MalformedCode)
            );
        }

        let rows = [EntangledHVec::from_indices(vec![1, 2, 3], 4096)];
        let server = Server::new(&matrix, &rows).unwrap();
        assert!(matches!(
            server.answer(&Query::from_words(vec![0; 4095])),
            Err(PrivateQueryError::Length { what: "query", .. })
        ));
        let (q, state) = client.query(&rows[0]).unwrap();
        let answer = server.answer(&q).unwrap();
        assert!(matches!(
            state.decode(server.hint(), &Answer::from_words(vec![0; 2])),
            Err(PrivateQueryError::Length { what: "answer", .. })
        ));

        let other_matrix = PublicMatrix::expand(params, [2; SEED_LEN]);
        let other_server = Server::new(&other_matrix, &rows).unwrap();
        let (_, state) = client.query(&rows[0]).unwrap();
        assert_eq!(
            state.decode(other_server.hint(), &answer).err(),
            Some(PrivateQueryError::HintMismatch)
        );
    }

    #[test]
    fn a_different_secret_does_not_recover_the_scores() {
        let params = Params::for_dim(DIM).unwrap();
        let matrix = PublicMatrix::expand(params, [5; SEED_LEN]);
        let client = Client::new(matrix.clone());
        let mut rng = StdRng::seed_from_u64(77);
        let query = random_code(&mut rng, DIM, 64);
        let rows: Vec<EntangledHVec> = (0..2000)
            .map(|_| correlated_row(&mut rng, &query, 64))
            .collect();
        let server = Server::new(&matrix, &rows).unwrap();
        let expected = plaintext_scores(&rows, &query);

        let (q, right_state) = client.query(&query).unwrap();
        let (_, wrong_state) = client.query(&query).unwrap();
        let answer = server.answer(&q).unwrap();

        let wrong = wrong_state.decode(server.hint(), &answer).unwrap();
        // A wrong secret decodes to values uniform in Z_p, so about N / p
        // positions agree by chance.
        let agree = wrong.iter().zip(&expected).filter(|(a, b)| a == b).count();
        assert!(agree < rows.len() / 20, "{agree} of {} agree", rows.len());
        assert!(wrong.iter().any(|&s| s > 64));
        assert_eq!(
            right_state.decode(server.hint(), &answer).unwrap(),
            expected
        );
    }

    #[test]
    fn repeated_queries_for_one_code_are_unlinkable_by_equality() {
        let params = Params::for_dim(DIM).unwrap();
        let client = Client::new(PublicMatrix::expand(params, [6; SEED_LEN]));
        let mut rng = StdRng::seed_from_u64(1);
        let code = random_code(&mut rng, DIM, 64);
        let (q1, s1) = client.query(&code).unwrap();
        let (q2, s2) = client.query(&code).unwrap();
        assert_eq!(q1.as_words().len(), DIM);
        let same = q1
            .as_words()
            .iter()
            .zip(q2.as_words())
            .filter(|(a, b)| a == b)
            .count();
        assert!(same < 8, "{same} of {DIM} ciphertext words repeat");
        assert_ne!(s1.secret, s2.secret);

        // The ciphertext length is independent of the query weight.
        let (heavy, _) = client.query(&random_code(&mut rng, DIM, 4096)).unwrap();
        assert_eq!(heavy.byte_len(), q1.byte_len());
    }

    #[test]
    fn secret_is_wiped() {
        let params = Params::for_dim(1024).unwrap();
        let client = Client::new(PublicMatrix::expand(params, [8; SEED_LEN]));
        let code = EntangledHVec::from_indices(vec![0, 9], 1024);
        let (_, mut state) = client.query(&code).unwrap();
        assert!(state.secret.iter().any(|&w| w != 0));
        // `Drop` calls exactly this.
        state.wipe();
        assert_eq!(state.secret.len(), LWE_DIM);
        assert!(state.secret.iter().all(|&w| w == 0));
    }

    #[test]
    fn error_sampler_is_bounded_with_the_stated_deviation() {
        let sampler = ErrorSampler::new();
        assert_eq!(sampler.thresholds.len(), ERROR_TAIL as usize + 1);
        assert!(sampler.thresholds.windows(2).all(|w| w[0] <= w[1]));
        let mut rng = ChaCha20Rng::from_seed([4; 32]);
        let n = 400_000;
        let (mut sum, mut sum_sq) = (0f64, 0f64);
        for _ in 0..n {
            let x = f64::from(sampler.sample(&mut rng) as i32);
            assert!(x.abs() <= f64::from(ERROR_TAIL));
            sum += x;
            sum_sq += x * x;
        }
        let mean = sum / f64::from(n);
        let sd = (sum_sq / f64::from(n) - mean * mean).sqrt();
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((sd - ERROR_SIGMA).abs() < 0.05, "sd {sd}");
    }

    #[test]
    fn matrix_expansion_is_deterministic_and_seed_dependent() {
        let params = Params::for_dim(64).unwrap();
        let a = PublicMatrix::expand(params, [1; SEED_LEN]);
        let b = PublicMatrix::expand(params, [1; SEED_LEN]);
        let c = PublicMatrix::expand(params, [2; SEED_LEN]);
        assert_eq!(a.data, b.data);
        assert_ne!(a.data, c.data);
        assert_ne!(a.row(0), a.row(1));
    }
}
