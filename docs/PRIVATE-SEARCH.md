# Private search (experimental)

**Status: experimental prototype. It hides a query from the server. It is not encryption of
stored data and must not be described as such.** It has not been externally reviewed and has
not been audited for constant-time behaviour. Do not rely on it to protect anything of value.

Code: `src/core/private_query.rs`, behind the Cargo feature `private-search` (off by default).
Measurement: `src/bin/private-query-bench.rs`, results in `benchmarks/results/private_query.json`.

This is unrelated to `src/core/mask.rs`, which is a keyed permutation of index labels
(obfuscation; see section 3 of `docs/RESEARCH-OPPORTUNITIES.md`).

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

- The 128-bit figure is the paper's claim. It was not recomputed here with a lattice estimator,
  and the paper's detailed analysis is in its full version (ePrint 2022/949), which could not be
  retrieved while writing this. No security level is claimed for this implementation.
- The number of LWE samples the paper's estimate covers is not stated in Section 4.2. Its
  parameter table lists database sizes up to 2^42, i.e. `sqrt(N) = 2^21` samples per query;
  treating 2^21 as the covered range is an inference, and it is why `MAX_DIM = 2^21`.
- Tiptoe's [2] parameters could not be retrieved and were not compared.
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

## References

1. Henzinger, Hong, Corrigan-Gibbs, Meiklejohn, Vaikuntanathan. One Server for the Price of
   Two: Simple and Fast Single-Server Private Information Retrieval. USENIX Security 2023.
   https://eprint.iacr.org/2022/949
2. Henzinger, Dauterman, Corrigan-Gibbs, Zeldovich. Private Web Search with Tiptoe. SOSP 2023.
   https://dl.acm.org/doi/10.1145/3600006.3613134
