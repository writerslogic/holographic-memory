<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Posting allocation repair, 2026-10-22

The paired replay saves **199,873,688 bytes (190.61 MiB)** of requested block
capacity on the saved first-600,000-input diagnostic index. It contains 548,246
live articles, 1,221,454 term lists and 78,640,902 postings. The original allocation
policy is from `d6a38eb11ab0ff320929249d0ec9e26f16e46dad`; the candidate applies
only the [initial reservation and frontier scratch repair](allocation-repair.patch).
The test-only census is identical in both binaries.

Both policies reopen the same journal, identified by [SHA-256](index.sha256), on
macOS arm64 with Rust 1.96.0 and four Rayon threads. `Block` is 56 bytes and
`PostingList` is 64 bytes. All 1,189,039 one-block lists change from capacity four
to capacity one. Posting counts and block lengths match; the only changed census
byte totals are block capacity and its unused subset.

The independent [Linux layout probe](linux-layout.json), using the extracted record
definitions and Rust 1.99.1 on x86_64, confirms the same 56/64-byte sizes and initial
block-vector capacities of four before reservation and one after reservation. The
[probe source](linux-layout.rs) and [compiler receipt](linux-layout-rustc.txt) are saved.

| Requested ownership | Original MiB | Candidate MiB |
|---|---:|---:|
| Block records and boxed frontiers | 328.68 | 138.06 |
| Posting data capacity | 180.53 | 180.53 |
| Posting-list record capacity | 128.00 | 128.00 |
| Shared forward dictionaries, counted once | 111.41 | 111.41 |
| Forward records, term IDs and frequencies | 259.92 | 259.92 |
| Position records, bytes and offsets | 400.07 | 400.07 |
| Document/chunk owners and strings | 172.96 | 172.96 |
| Document/slot lookup payload and arrays | 72.09 | 72.09 |

The raw [original](baseline.json) and [candidate](candidate.json) censuses also
contain the complete joint histograms, dictionary capacities and spare posting-data
bytes. Unused block capacity falls from 214,172,504 to 14,298,816 bytes; unused
posting-data capacity remains 61,528,338 bytes. Tree node overhead, allocator
metadata/rounding and other engine components are excluded; map payloads are lower
bounds. These figures are not a complete heap total.

The [comparison receipt](comparison.json) records a single process-RSS pair:
2,070,446,080 versus 1,818,066,944 peak bytes, an observed 240.69 MiB reduction.
RSS includes allocator behavior and process state, so this is not an additive
saving. Other local builds ran during these samples; replay time is not a controlled
performance comparison. No full-corpus 12-GiB/no-swap build, reopen or query gate
was executed for this repair, and no competitive latency result is claimed.

The isolated [allocation probe](../../research/postings/README.md) constructs 1,000
lists per case using the actual posting source. On macOS, singleton lists change
from 4,001 to 2,001 allocation/reallocation calls and from 296,000 to 128,000 live
requested bytes. The common owner vector is included. Each singleton block avoids
the two temporary frontier allocations and 168 bytes of spare block records. For
128-posting single-impact blocks, calls fall from 11,001 to 4,001; the original
filtered vector grows repeatedly. These synthetic measurements do not attribute
the prior whole-engine allocation total to this function.

The [ranked validation](ranked-validation.json) checks 11 queries, including all
seven benchmark categories, a long union, repeated terms, prohibited phrases and
optional phrases, at k=0/10/1000. Complete results are compared with the exhaustive
evaluator through DAAT and WAND: 132 comparisons across both binaries. The 33
reference results also match across binaries, including all IDs, scores and order.
The compact receipt retains full-result hashes and the first ten hits per case.

The [validation receipt](validation.json) records formatting, default/all-feature
clippy, 604 default-feature tests, 952 all-feature tests, and the Rust 1.89 default
feature check. Regressions cover first-run capacities and subsequent growth,
empty/zero/duplicate/extreme impacts, pairwise dominance, 127/128/129 limits,
partial-block extension and invalidated fallback bounds. Existing format golden,
mutation, replay and WAND differential tests pass.
