# Private search (experimental)

**Status: experimental prototype. It hides a query from the server. It is not encryption of
stored data and must not be described as such.** It has not been externally reviewed and has
not been audited for constant-time behaviour. Do not rely on it to protect anything of value.

Code: `src/core/private_query.rs`, behind the Cargo feature `private-search` (off by default).
Measurement: `src/bin/private-query-bench.rs`, results in `benchmarks/results/private_query.json`.
A second layout for dense low-bit codes, which can score a chosen part of the database, is in
`src/core/private_query/clustered.rs`; see [Clustered ranking of dense codes](#clustered-ranking-of-dense-codes).

This is unrelated to `src/core/mask.rs`, which is a keyed permutation of index labels
(obfuscation).

## What it does

A server stores sparse binary codes (`EntangledHVec`) in the clear. A client sends an
encrypted query code; the server computes, for every stored row, an encryption of the
intersection count `|row ∩ query|`; the client decrypts all `N` counts. The construction is
the secret-key Regev scheme with a shared public matrix, as used by SimplePIR [1], applied to a
sparse 0/1 database matrix instead of a PIR table. Tiptoe [2] uses the same idea for ranking.

With `q = 2^32`, `Δ = floor(q / p)`, database `X ∈ {0,1}^(N×D)`:

| Step | Who | Computation |
|---|---|---|
| Public | both | `A ∈ Z_q^(D×n)` expanded from a public 32-byte seed (ChaCha20) |
| Offline | server | hint `H = X·A ∈ Z_q^(N×n)`; client downloads `H` |
| Query | client | fresh `s ← Z_q^n`, `e ← χ^D`; sends `c = A·s + e + Δ·u` |
| Answer | server | `a = X·c` (sum of `c` over each row's active indices) |
| Decode | client | `score_i = round((a_i − H_i·s) / Δ) mod p` |

A new secret is sampled for every query. `QueryState` is not `Clone`, `decode` takes it by
value, and its `Drop` wipes the secret with `zeroize`, so one secret cannot serve two queries.

## Threat model

**Hidden:** the content of the query code `u` (which indices are active and how many), from an
honest-but-curious server that follows the protocol and sees `A`, the database, and any number
of queries. This rests on the hardness of LWE with the parameters below. Every query is `D`
words regardless of its weight.

**Not hidden, not provided:**

- The database is fully visible to the server. Stored codes are plaintext. Nothing here
  protects a user's stored memories from whoever hosts them.
- The client learns the score against every row, for every query. The database is not private
  from clients: `D` unit-vector queries reveal it entirely, and a client is not forced to send a
  binary query.
- No protection against a malicious server returning wrong answers or a wrong hint. Answers are
  not authenticated; a wrong answer decodes to a wrong score with no error.
- No hiding of query timing, query count, client identity, or network metadata. What the client
  does after decoding (for example fetching the top-scoring record) is outside the scheme and
  can reveal the query.
- The implementation is not constant-time audited. The error sampler uses a table search with
  data-dependent timing, and the arithmetic has not been checked for secret-dependent timing.
- Not externally reviewed. No formal proof accompanies this code; the security argument is the
  one in [1] and holds only to the extent this code matches that scheme.
- The secret is wiped on drop; the thread-local RNG state and any copies the allocator or OS
  made (swap, core dumps) are not.

## Parameters

| Parameter | Value | Source |
|---|---|---|
| LWE dimension `n` | 1280 | lattice estimator [5], see below ([1] uses 1024) |
| Modulus `q` | 2^32 (wrapping `u32`) | [1] Section 4.2 |
| Error distribution `χ` | discrete Gaussian, σ = 6.4 | [1] Section 4.2 |
| Secret distribution | uniform over `Z_q^n` | [1] Figure 2 |
| Samples per query `m` | `D` (16384 by default; at most 2^21 accepted) | see below |
| Plaintext modulus `p` | 1024 (default) | chosen here from the bound below |
| Error tail cut | ±77 = ceil(12σ) | chosen here |
| Max row weight | 1023 (default) | derived below |

`q` and σ are SimplePIR's [1]: Section 4.2 of its USENIX Security 2023 text states `n = 2^10`,
`q = 2^32`, discrete Gaussian with σ = 6.4, chosen "to have 128-bit security, according to
modern lattice-attack-cost estimates"; Figure 2 samples `s` uniformly from `Z_q^n` and builds
the query as `A·s + e + Δ·u`; Section 4.1 notes that `A` may be reused across many encryptions.
The dimension is raised from SimplePIR's 1024 to 1280 because the lattice estimator does not
reproduce the 128-bit figure at 1024.

**Security estimate.** The lattice estimator [5] (commit `e4333d9`, run under the
pip-installable SageMath distribution `passagemath-standard`) on LWE with `q = 2^32`, a secret
uniform mod `q`, a discrete Gaussian error and `2^21` samples, the most one query can publish
(`MAX_DIM`; a clustered probe publishes `C·d`, 6,400 at `C = 64, d = 100`). Script
`benchmarks/private_search/lattice_estimate.py`, output
`benchmarks/results/private_query_lattice_estimate.json`. Bits are `log2` of the estimated
ring operations of the cheapest attack. "Default" is `LWE.estimate` with its default
reduction cost model (MATZOV); "core-SVP" is `LWE.estimate.rough`, one sieving call of cost
`2^(0.292 β)` for BKZ block size β, the conservative count used for Kyber-style claims.

| n | σ | Default, cheapest attack (bits) | Core-SVP (bits) | uSVP block size | Note |
|---|---|---|---|---|---|
| 1024 | 6.4 | 121.5 (bdd) | 95.2 | 326 | SimplePIR [1]; previous set |
| 1024 | 11 | 128.4 (bdd) | 102.5 | 351 | YPIR [3], `s = 11·sqrt(2π)` |
| 1152 | 6.4 | 137.5 (bdd) | 111.8 | 383 | |
| **1280** | **6.4** | **153.7 (bdd)** | **128.8** | 441 | **this crate** |
| 1408 | 6.4 | 170.3 (bdd) | 146.0 | 500 | |
| 2048 | 6.4 | 257.1 (bdd) | 236.8 | 811 | |

The attacks run are uSVP, BDD (primal decoding), dual and dual-hybrid; Arora-GB, BKW and the
BDD and meet-in-the-middle hybrids are skipped as slow and not competitive at this error and
modulus. `n = 1280` is the smallest multiple of 128 at which both columns exceed 128 bits;
`n = 1152` clears the default model but not core-SVP. As a cross-check, the Homomorphic
Encryption Standard [6] (Table 1, BKZ.sieve, σ ≈ 3.2) allows `log2 q` up to 29 at `n = 1024`
and 56 at `n = 2048` for 128 bits with a uniform secret, so `q = 2^32` at `n = 1024` is outside
it; and the hand-written core-SVP script of the previous version
(`benchmarks/private_search/lwe_core_svp.py`, `n = 1024` only) agrees with the estimator's
95.2 and 102.5. These are estimates of known attacks under stated cost models, not proofs, and
no quantum figure is claimed.

Not verified, stated so rather than assumed:

- Tiptoe [2] ranks with different parameters: `q = 2^64`, `n = 2048`, σ = 81,920 and a ternary
  secret, stated as 128-bit for encrypted vectors of dimension up to 2^27 (Appendix C). They are
  not used here.
- Deviations from the idealised scheme: the Gaussian is truncated at ±77 (tail mass below
  2^-100) and its table is computed in `f64` (about 2^-53 precision per entry), and `A` is
  pseudorandom (ChaCha20) rather than uniform. Their effect on the estimate was not analysed.
- Multiple queries reuse `A` with independent secrets. The standard hybrid argument loses a
  factor equal to the number of queries; no concrete query budget was derived.

## Correctness bound

For row `i` with active set `R_i`:

```
a_i − H_i·s = Σ_{j∈R_i} (c_j − A_j·s) = Σ_{j∈R_i} (e_j + Δ·u_j) = Δ·|R_i ∩ u| + E_i   (mod q)
E_i = Σ_{j∈R_i} e_j,   |E_i| ≤ |R_i| · 77
```

Rounding to the nearest multiple of `Δ` returns `|R_i ∩ u|` exactly when

1. `|R_i| < p`, so the score, at most `|R_i|`, does not wrap modulo `p`; and
2. `|E_i| < Δ/2`, guaranteed by `2 · |R_i| · 77 < Δ`.

Because every error sample is bounded, this holds with certainty, not with a failure
probability. The secret dimension `n` does not appear: `H_i·s` cancels exactly, so raising `n`
changes neither bound. The largest admissible row weight is

```
w_max(p) = min(p − 1, floor((Δ − 1) / 154)),   Δ = floor(2^32 / p)
```

`p = 1024` gives `Δ = 2^22` and `w_max = min(1023, 27235) = 1023`. `p = 65536` gives
`Δ = 65536` and `w_max = 425`, where the noise term binds. `Params::new` rejects a weight limit
above `w_max(p)` and `Server::new` rejects any row heavier than the limit. The query weight is
unconstrained: the error accumulates over the row's indices only. The bound is worst-case and
loose; at weight 64 the typical `|E_i|` is about `6.4·sqrt(64) ≈ 51` against `Δ/2 = 2,097,152`.

## Measured cost

From `benchmarks/results/private_query.json`, **measured at the previous `n = 1024`**, not
re-run at 1280: hint bytes and the hint build, client encrypt and decode times scale with `n`
(hint `4·n` bytes per row, 5,120 at `n = 1280`); server answer time and query and answer bytes
do not. D = 16384, row weight 64, release build, Apple M4, 10 rayon threads, crate 0.6.1. Medians of 20 queries (3 hint builds). One run on a machine that
was not idle; the per-query maxima in the JSON are 10 to 200 times the medians, so treat these
as indicative. Tested range is N = 10^3 to 10^5; nothing is claimed beyond it.

| N | Server answer (ms) | Hint build (ms) | Client encrypt (ms) | Client decode (ms) | Query (bytes) | Answer (bytes) | Hint (bytes) |
|---|---|---|---|---|---|---|---|
| 1,000 | 0.08 | 9.6 | 2.6 | 0.45 | 65,536 | 4,000 | 4,096,000 |
| 10,000 | 0.39 | 136 | 3.3 | 0.87 | 65,536 | 40,000 | 40,960,000 |
| 100,000 | 0.88 | 816 | 1.3 | 5.1 | 65,536 | 400,000 | 409,600,000 |

The hint is large: `4·n` bytes per stored row (4,096 at `n = 1024`, 5,120 at 1280), 16 to 20
times the 256 bytes of a weight-64 row's plaintext indices, and 409.6 MB at N = 10^5 in the
table (512 MB at `n = 1280`). The client must hold it and must download it
again whenever the database changes. Server and client each also hold `A`, 64 MiB at D = 16384 and
`n = 1024` (43 ms to expand), 80 MiB at 1280. The answer is 4 bytes per row, so communication and client work are linear
in N; this scheme returns all scores and does no private top-k selection.

Regenerate with:

```sh
cargo run --release --features private-search --bin private-query-bench
```

## Tests

`cargo test --features private-search private_query` checks: decoded scores equal plaintext
intersection counts (N = 2000, D = 16384, two matrix seeds × four data seeds × two queries);
correctness at the maximum row weight for both regimes of the bound, and rejection one above
it; rejection of malformed codes, wrong lengths and mismatched hints; that decoding with a
different secret does not return the scores; that two encryptions of one code share almost no
ciphertext words and ciphertext length does not depend on query weight; that the secret buffer
is zeroed by the routine `Drop` calls; and the error sampler's bound and deviation.

These tests establish correctness and the stated mechanical properties. They cannot establish
that the query is hidden; that rests on the LWE assumption and the parameter analysis in [1].

## Clustered ranking of dense codes

Code: `src/core/private_query/clustered.rs`, same feature. Measurement:
`src/bin/private-rank-bench.rs`. Recall and sizes: `benchmarks/results/private_rank_sweep_100k.json`
and `private_rank_sweep_1m.json`. Protocol timings: `private_rank_protocol_100k.json` and
`private_rank_protocol_1m.json`.

The database is `N` codes of `d` small signed integers in `C` clusters of at most `S` codes,
laid out the way Tiptoe [2] lays out ranking: a matrix of `S` rows and `C·d` columns whose block
`k` of row `i` is the `i`-th code of cluster `k` (zeros past the end of the cluster). The client
keeps the `C` centroids, picks the `P` nearest clusters locally, and sends one ciphertext of
`C·d` words per cluster, each under its own secret, encrypting the query code in that cluster's
block and zeros elsewhere. The server multiplies the whole matrix by every ciphertext and returns
`S` words per probe; the client decodes the `P·S` scores of its clusters with the hint
`H = M·A` (`S × n` words). `C = 1` scores the whole database. The LWE parameters and threat model
are those above, plus: `C`, `P` and `S` are public and fixed, the cluster layout is public, and
the client receives `P·S` scores, not ten.

**Decoding.** Row `i` has error `E_i = Σ_j M_ij·e_j`. A centred discrete Gaussian of parameter
σ is σ-sub-Gaussian, and symmetric truncation keeps that, so
`P(|E_i| ≥ Δ/2) ≤ 2·exp(-(Δ/2)² / (2σ²·‖M_i‖²))`. `DenseParams` takes a bound on `‖M_i‖²`, which
`DenseServer::new` checks for every row, sets `p` to the smallest power of two above `2·d·Bx·Bu`
(`Bx`, `Bu`: database and query entry bounds), and rejects parameters whose union bound over a
probe's `S` scores exceeds 2^-40. SimplePIR's δ = 2^-40 is per query; here it is per probe, so a
query of `P` probes fails with probability at most `P` times the bound (2^-83 or less for every
configuration in the tables). Decoding is correct with that probability, not with certainty,
unless `DenseParams::exact` reports that the truncated worst case
`2·77·C·d·Bx < Δ` holds too (it does for the 1-bit rows below). Tests
(`cargo test --features private-search clustered`) compare decoded scores with plaintext dot
products for whole-database scoring, nine probes, entries at ±15 with the extreme scores
`±d·Bx·Bu`, and 1-bit codes; check that the error deviation over 2,048 rows at the norm bound is
within 6% of `σ·‖M_i‖`; and check rejection of out-of-bound entries, oversized clusters, rows
over the norm bound, bad cluster indices and mismatched hints.

**Workload.** glove-100-angular: the first `N` training rows and the first 1,000 test queries,
unit-normalised and quantized to odd-integer levels, written `database bits:query bits` (1 bit is
the sign, a bipolar hypervector; 4 bits is ±1 … ±15). k-means with 12 Lloyd iterations on a
100,000-row sample; cluster capacity 1.25× the mean size, each row placed in its nearest cluster
with room. Recall@10 is measured against the exhaustive plaintext ranking of the same codes and
is tie-aware: a returned code counts when its score reaches the tenth-best plaintext score. The
recall columns come from the plaintext sweep over 1,000 queries (regenerated at `n = 1280`; every
recall value equals the `n = 1024` sweep's, since the plaintext ranking involves no LWE
parameter); the protocol runs compared every
decoded score of their 30 queries with the plaintext score and found no mismatch in any
configuration (9.4 million scores at `N = 10^6, C = 64, P = 16`). For context, recall@10 against
the exact f32 neighbours is also given; the 4:4 codes' own exhaustive ranking reaches 0.738 at
`N = 10^6`.

Timing: writerslogic-02 (Intel Core Ultra 9 185H, Linux, AVX2, 22 rayon threads), release build,
Rust 1.99.1, 2 warm-up and 30 timed queries per configuration, one run. Server time covers all
`P` probes of one query, batched eight to a pass over the database; nothing is batched across
queries. The tables are the `n = 1280` runs. In the same locked session the `n = 1024` build of
the previous version ran the same configurations (order 1024, 1280, 1280, 1024 at `10^6`; files
in `benchmarks/results/private_rank_n1280_ab/`). Upload, download and the decoded scores are
identical; server time is within run-to-run spread (`C = 64, P = 16`: 24.9 and 27.4 ms at
`n = 1024`, 25.1 and 23.3 ms at 1280), as expected since the server never touches `A` or `s`.
What grows with `n` is the client side: the hint by exactly 25%, the hint build from 1.3–1.5 s
to 1.6–1.7 s, and decoding (`C = 1`: 57.5 and 53.1 ms to 65.7 and 66.2 ms). "Scored" is `P·S/N`, an upper bound on the share of rows whose scores the client
receives: `S` is the fullest cluster's size and smaller clusters are padded, so the mean share is
close to `P/C`. The server reads the whole database for every probe.

`N = 10^6`:

| Codes | C | P | Scored | Recall@10, same codes | Recall@10, f32 | Server ms, median [p10–p90] | Upload | Download | Hint | Client prepare / decode ms | Failure bound per probe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4:4 | 1 | 1 | 100% | 1.000 | 0.738 | 4.5 [3.8–5.3] | 400 B | 4.00 MB | 5,120 MB | 0.0 / 65.7 | 2^-5911 |
| 4:4 | 16 | 4 | 31% | 0.896 | 0.688 | 10.7 [10.2–11.2] | 26 KB | 1.25 MB | 400 MB | 1.2 / 26.7 | 2^-387 |
| 4:4 | 32 | 8 | 31% | 0.935 | 0.709 | 12.2 [10.3–15.9] | 102 KB | 1.25 MB | 200 MB | 2.4 / 32.5 | 2^-188 |
| 4:4 | 64 | 8 | 16% | 0.902 | 0.694 | 10.4 [9.6–13.4] | 205 KB | 0.63 MB | 100 MB | 3.3 / 21.0 | 2^-87 |
| 4:4 | 64 | 16 | 31% | 0.961 | 0.723 | 25.1 [21.8–28.1] | 410 KB | 1.25 MB | 100 MB | 5.4 / 42.2 | 2^-87 |
| 1:4 | 64 | 16 | 31% | 0.924 | 0.252 | 23.9 [20.1–25.9] | 410 KB | 1.25 MB | 100 MB | 4.3 / 40.2 | exact |

`N = 10^5`:

| Codes | C | P | Scored | Recall@10, same codes | Recall@10, f32 | Server ms, median [p10–p90] | Upload | Download | Hint | Client prepare / decode ms | Failure bound per probe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4:4 | 1 | 1 | 100% | 1.000 | 0.759 | 1.1 [0.8–1.6] | 400 B | 0.40 MB | 512 MB | 0.0 / 8.4 | 2^-6020 |
| 4:4 | 16 | 8 | 63% | 0.962 | 0.740 | 1.8 [1.6–2.0] | 51 KB | 0.25 MB | 40 MB | 1.0 / 12.0 | 2^-392 |
| 4:4 | 32 | 8 | 31% | 0.912 | 0.723 | 1.6 [1.5–1.7] | 102 KB | 0.13 MB | 20 MB | 1.5 / 9.4 | 2^-192 |
| 4:4 | 64 | 8 | 16% | 0.860 | 0.694 | 1.6 [1.4–1.7] | 205 KB | 0.06 MB | 10 MB | 2.2 / 5.6 | 2^-91 |
| 4:4 | 64 | 16 | 31% | 0.933 | 0.732 | 3.3 [3.1–3.5] | 410 KB | 0.13 MB | 10 MB | 3.3 / 11.0 | 2^-91 |
| 1:4 | 64 | 16 | 31% | 0.878 | 0.269 | 3.3 [3.1–3.5] | 410 KB | 0.13 MB | 10 MB | 3.1 / 11.0 | exact |

Offline per database version, besides the hint: the centroids (`C·d` floats, 25.6 KB at `C = 64`)
and the public matrix `A` (`C·d × n` words, 33 MB at `C = 64`, expanded from a 32-byte seed in
3–10 ms on the test machine). Building the hint took 1.6–1.7 s at `N = 10^6` for `C ≥ 16` and
2.6 s at `C = 1`. Client prepare (fresh secrets and `A·s + e` for every probe) does not depend on the
query and can run ahead; adding the query took under 0.01 ms. Results are (cluster, slot)
positions; unless records are addressed by position, the client also needs the position-to-record
map, 4 bytes per row (4 MB at `N = 10^6`). The sweep files cover `C` from 1 to
1,024 at `10^5` and 1 to 256 at `10^6`, `P` from 1 to 64, and widths 1:4, 2:4 and 4:4; 4:4 codes
fail the 2^-40 bound from `C = 128` at `10^6` and `C = 256` at `10^5` (`params_valid: false`).

**Published systems, not like-for-like.** SimplePIR [1] reports 10 GB/s/core of server
throughput (DoublePIR 7.4) on 1 GB databases, single-threaded on an AWS c5n.metal, with a 121 MB
hint (DoublePIR 16 MB) and 242 KB per query (DoublePIR 345 KB); it retrieves a record and does
not rank. Tiptoe [2] searches 364 million pages with 145 core-seconds of server compute on 45
servers, 56.9 MiB per query of which 42.2 MiB (a per-query token that replaces the hint) moves
before the query, and 2.7 s end to end. It scores a single cluster of about 50,000 documents with
192-dimensional signed 4-bit embeddings, puts 20% of documents in two clusters, and reports an
average rank of 7.7 on MS MARCO (non-private embedding search 2.3, tf-idf 6.7). Its corpus
is 364 times larger than the one here, it keeps no client hint, and it measures end-to-end
quality rather than recall against its own codes.

**The claim at `N = 10^6`.** Targets: at most 100 ms of server time per query on one node, at
most a few MB of online communication, the offline hint stated, and recall@10 of at least 0.9
against the plaintext ranking of the same codes. Kill criterion: no configuration meets all four
at `N = 10^6`. It did not fire. On glove-100 quantized to 4-bit codes, `C = 64, P = 16` meets all
four: 25.1 ms median server time (28.1 ms at p90), 1.66 MB online (410 KB up, 1.25 MB down), a
100 MB hint per database version, recall 0.961. `C = 32, P = 8` does too: 12.2 ms, 1.35 MB, a
200 MB hint, recall 0.935. `C = 64, P = 8` (10.4 ms, 0.83 MB, 0.902) is within sampling error of
the recall threshold over 1,000 queries and is not counted as a pass. Exact
scoring of the whole database (`C = 1`) takes 4.5 ms of server time and 4 MB down but needs a
5,120 MB hint and 66 ms of client decoding. Limits of the result: one dataset, one machine, one
run, 30 timed queries per configuration; recall against the share of the database scored
depends on how well the data clusters, and on glove-100 the passing points score at most 31% of
the rows (`P/C` = 25%).
The query's hiding rests on LWE with `n = 1280, q = 2^32, σ = 6.4`: 2^153.7 by the lattice
estimator's default model and 2^128.8 by core-SVP (see [Parameters](#parameters)). Raising `n`
rather than σ leaves every failure bound above unchanged. The alternative of YPIR's σ = 11 at
`n = 1024` reaches only 2^128.4 by the default model and 2^102.5 by core-SVP, and would multiply
the failure-bound exponent by (6.4/11)² ≈ 0.34, which by that scaling (not measured) pushes the
4:4 rows at `C = 64` past 2^-40. Not implemented: dropping low bits of the
answer words (16-bit plaintext in a 32-bit word), and removing the hint with Tiptoe's token or a
hintless scheme such as YPIR [3].

Regenerate with:

```sh
cargo run --release --features private-search --bin private-rank-bench -- \
  sweep --data <glove-100-angular dir> --sizes 1000000 --clusters 1,16,32,64,128,256 \
  --out benchmarks/results/private_rank_sweep_1m.json
cargo run --release --features private-search --bin private-rank-bench -- \
  protocol --data <glove-100-angular dir> --sizes 1000000 --queries 64 --protocol-queries 30 \
  --configs 1:1:4:4,16:4:4:4,32:8:4:4,64:8:4:4,64:16:4:4,64:16:1:4 \
  --out benchmarks/results/private_rank_protocol_1m.json
```

The `10^5` files use `--sizes 100000`, the default cluster list for the sweep, and the configs
`1:1:4:4,16:8:4:4,32:8:4:4,64:8:4:4,64:16:4:4,64:16:1:4`. The data directory holds `train.f32`
and `test.f32` (little-endian f32 rows of 100).

## References

1. Henzinger, Hong, Corrigan-Gibbs, Meiklejohn, Vaikuntanathan. One Server for the Price of
   Two: Simple and Fast Single-Server Private Information Retrieval. USENIX Security 2023.
   https://eprint.iacr.org/2022/949
2. Henzinger, Dauterman, Corrigan-Gibbs, Zeldovich. Private Web Search with Tiptoe. SOSP 2023.
   https://dl.acm.org/doi/10.1145/3600006.3613134 (read: https://par.nsf.gov/servlets/purl/10530535)
3. Menon, Wu. YPIR: High-Throughput Single-Server PIR with Silent Preprocessing. USENIX Security
   2024. https://www.usenix.org/system/files/usenixsecurity24-menon.pdf
4. Alkim, Ducas, Pöppelmann, Schwabe. Post-quantum key exchange: a new hope. USENIX Security
   2016, Section 6.
5. Albrecht, Player, Scott. On the concrete hardness of Learning with Errors. J. Math. Cryptol.
   2015; software: https://github.com/malb/lattice-estimator
6. Albrecht et al. Homomorphic Encryption Security Standard, v1.1, 2018.
   https://homomorphicencryption.org/standard/
