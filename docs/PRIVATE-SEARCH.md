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
| LWE dimension `n` | 1024 | [1] Section 4.2, also Remark 4.1 |
| Modulus `q` | 2^32 (wrapping `u32`) | [1] Section 4.2 |
| Error distribution `χ` | discrete Gaussian, σ = 6.4 | [1] Section 4.2 |
| Secret distribution | uniform over `Z_q^n` | [1] Figure 2 |
| Samples per query `m` | `D` (16384 by default; at most 2^21 accepted) | see below |
| Plaintext modulus `p` | 1024 (default) | chosen here from the bound below |
| Error tail cut | ±77 = ceil(12σ) | chosen here |
| Max row weight | 1023 (default) | derived below |

Verified against the USENIX Security 2023 text of [1]: Section 4.2 states `n = 2^10`,
`q = 2^32`, discrete Gaussian with σ = 6.4, and that these were chosen "to have 128-bit
security, according to modern lattice-attack-cost estimates"; Figure 2 samples `s` uniformly
from `Z_q^n` and builds the query as `A·s + e + Δ·u`; Section 4.1 notes that `A` may be reused
across many encryptions.

Not verified, stated so rather than assumed:

- The 128-bit figure is the paper's claim, which cites the attack-cost estimates of Albrecht,
  Player and Scott (2015) and names no attack. A primal-uSVP core-SVP estimate (the 2016
  method of [4]; `benchmarks/private_search/lwe_core_svp.py`, output in
  `benchmarks/results/private_query_core_svp.json`) needs BKZ block size 326 for
  `n = 1024, q = 2^32, σ = 6.4`: 2^95.2 classically and 2^86.4 quantumly. Core-SVP counts a
  single sieving call; it is a conservative yardstick, not an attack cost, and the lattice
  estimator's cost models add BKZ tours and polynomial factors on top. For comparison, YPIR [3]
  uses the same `n` and `q` with `s = 11·sqrt(2π)` (σ ≈ 11) for 128-bit security by the lattice
  estimator; the same script gives block size 351 (2^102.5) there. No security level is claimed for this
  implementation; σ = 6.4 is kept as SimplePIR's published set, not re-derived.
- The number of LWE samples the estimate covers: Section 4.2 does not state one, Appendix C.1
  claims security for `sqrt(N)` samples, and the parameter table runs to `N = 2^42`, i.e.
  `sqrt(N) = 2^21` samples per query; that is why `MAX_DIM = 2^21`. The primal attack above uses
  about 1,100 samples however many a query publishes.
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
probability. The largest admissible row weight is

```
w_max(p) = min(p − 1, floor((Δ − 1) / 154)),   Δ = floor(2^32 / p)
```

`p = 1024` gives `Δ = 2^22` and `w_max = min(1023, 27235) = 1023`. `p = 65536` gives
`Δ = 65536` and `w_max = 425`, where the noise term binds. `Params::new` rejects a weight limit
above `w_max(p)` and `Server::new` rejects any row heavier than the limit. The query weight is
unconstrained: the error accumulates over the row's indices only. The bound is worst-case and
loose; at weight 64 the typical `|E_i|` is about `6.4·sqrt(64) ≈ 51` against `Δ/2 = 2,097,152`.

## Measured cost

From `benchmarks/results/private_query.json`: D = 16384, row weight 64, release build, Apple M4,
10 rayon threads, crate 0.6.1. Medians of 20 queries (3 hint builds). One run on a machine that
was not idle; the per-query maxima in the JSON are 10 to 200 times the medians, so treat these
as indicative. Tested range is N = 10^3 to 10^5; nothing is claimed beyond it.

| N | Server answer (ms) | Hint build (ms) | Client encrypt (ms) | Client decode (ms) | Query (bytes) | Answer (bytes) | Hint (bytes) |
|---|---|---|---|---|---|---|---|
| 1,000 | 0.08 | 9.6 | 2.6 | 0.45 | 65,536 | 4,000 | 4,096,000 |
| 10,000 | 0.39 | 136 | 3.3 | 0.87 | 65,536 | 40,000 | 40,960,000 |
| 100,000 | 0.88 | 816 | 1.3 | 5.1 | 65,536 | 400,000 | 409,600,000 |

The hint is large: `4·n = 4096` bytes per stored row, 16 times the 256 bytes of a weight-64
row's plaintext indices, and 409.6 MB at N = 10^5. The client must hold it and must download it
again whenever the database changes. Server and client each also hold `A`, 64 MiB at D = 16384
(43 ms to expand). The answer is 4 bytes per row, so communication and client work are linear
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
probe's `S` scores exceeds 2^-40, SimplePIR's δ. Decoding is correct with that probability, not
with certainty, unless `DenseParams::exact` reports that the truncated worst case
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
recall columns come from the plaintext sweep over 1,000 queries; the protocol runs compared every
decoded score of their 30 queries with the plaintext score and found no mismatch in any
configuration (9.4 million scores at `N = 10^6, C = 64, P = 16`). For context, recall@10 against
the exact f32 neighbours is also given; the 4:4 codes' own exhaustive ranking reaches 0.738 at
`N = 10^6`.

Timing: writerslogic-02 (Intel Core Ultra 9 185H, Linux, AVX2, 22 rayon threads), release build,
Rust 1.99.1, 2 warm-up and 30 timed queries per configuration, one run. Server time covers all
`P` probes of one query, batched eight to a pass over the database; nothing is batched across
queries. "Scored" is `P·S/N`, the share of the database whose scores the client receives; the
server reads all of it for every probe.

`N = 10^6`:

| Codes | C | P | Scored | Recall@10, same codes | Recall@10, f32 | Server ms, median [p10–p90] | Upload | Download | Hint | Client prepare / decode ms | Failure bound per probe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4:4 | 1 | 1 | 100% | 1.000 | 0.738 | 4.1 [3.1–5.0] | 400 B | 4.00 MB | 4,096 MB | 0.0 / 53.2 | 2^-5911 |
| 4:4 | 16 | 4 | 31% | 0.896 | 0.688 | 11.1 [10.4–15.9] | 26 KB | 1.25 MB | 320 MB | 1.2 / 22.4 | 2^-387 |
| 4:4 | 32 | 8 | 31% | 0.935 | 0.709 | 14.2 [11.7–16.1] | 102 KB | 1.25 MB | 160 MB | 2.4 / 31.2 | 2^-188 |
| 4:4 | 64 | 8 | 16% | 0.902 | 0.694 | 10.4 [9.5–13.8] | 205 KB | 0.63 MB | 80 MB | 3.1 / 18.4 | 2^-87 |
| 4:4 | 64 | 16 | 31% | 0.961 | 0.723 | 24.4 [22.7–27.4] | 410 KB | 1.25 MB | 80 MB | 4.6 / 37.7 | 2^-87 |
| 1:4 | 64 | 16 | 31% | 0.924 | 0.252 | 24.2 [21.2–26.1] | 410 KB | 1.25 MB | 80 MB | 4.8 / 35.9 | exact |

`N = 10^5`:

| Codes | C | P | Scored | Recall@10, same codes | Recall@10, f32 | Server ms, median [p10–p90] | Upload | Download | Hint | Client prepare / decode ms | Failure bound per probe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4:4 | 1 | 1 | 100% | 1.000 | 0.759 | 1.0 [0.8–1.3] | 400 B | 0.40 MB | 410 MB | 0.0 / 6.6 | 2^-6020 |
| 4:4 | 16 | 8 | 63% | 0.962 | 0.740 | 1.6 [1.5–1.7] | 51 KB | 0.25 MB | 32 MB | 0.9 / 9.8 | 2^-392 |
| 4:4 | 32 | 8 | 31% | 0.912 | 0.723 | 1.5 [1.4–1.7] | 102 KB | 0.13 MB | 16 MB | 1.4 / 8.7 | 2^-192 |
| 4:4 | 64 | 8 | 16% | 0.860 | 0.694 | 1.5 [1.5–1.6] | 205 KB | 0.06 MB | 8 MB | 2.0 / 5.2 | 2^-91 |
| 4:4 | 64 | 16 | 31% | 0.933 | 0.732 | 3.1 [2.9–3.2] | 410 KB | 0.13 MB | 8 MB | 2.7 / 10.2 | 2^-91 |
| 1:4 | 64 | 16 | 31% | 0.878 | 0.269 | 3.1 [2.9–3.2] | 410 KB | 0.13 MB | 8 MB | 2.8 / 10.0 | exact |

Offline per database version, besides the hint: the centroids (`C·d` floats, 25.6 KB at `C = 64`)
and the public matrix `A` (`C·d × n` words, 26 MB at `C = 64`, expanded from a 32-byte seed in
4–10 ms on the test machine). Building the hint took 1.3 s at `N = 10^6` for `C ≥ 16` and 2.1 s at
`C = 1`. Client prepare (fresh secrets and `A·s + e` for every probe) does not depend on the
query and can run ahead; adding the query took under 0.01 ms. The sweep files cover `C` from 1 to
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
against the plaintext ranking of the same codes. On glove-100 quantized to 4-bit codes, `C = 64,
P = 16` meets all four: 24.4 ms median server time (27.4 ms at p90), 1.66 MB online (410 KB up,
1.25 MB down), an 80 MB hint per database version, recall 0.961; `C = 64, P = 8` gives 10.4 ms,
0.83 MB and 0.902, and `C = 32, P = 8` gives 14.2 ms, 1.35 MB, a 160 MB hint and 0.935. Exact
scoring of the whole database (`C = 1`) takes 4.1 ms of server time and 4 MB down but needs a
4,096 MB hint and 53 ms of client decoding. Limits of the result: one dataset, one machine, one
run, 30 timed queries per configuration; recall against the fraction of the database scored
depends on how well the data clusters, and glove-100 needs 16–31% of rows scored to pass 0.9.
The security rests on SimplePIR's parameter set as stated above (128-bit by SimplePIR's
estimate, 2^95 by the core-SVP estimate here). Raising σ to YPIR's 11 would multiply the
failure-bound exponent by (6.4/11)² ≈ 0.34, which pushes the 4:4 rows at `C = 64` past 2^-40 and
leaves the `C ≤ 32` rows and the 1-bit rows within it. Not implemented: dropping low bits of the
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
