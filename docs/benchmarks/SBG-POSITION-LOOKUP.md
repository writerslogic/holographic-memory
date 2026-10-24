# Posting-aligned phrase position lookup

Packed term ordinals remove per-candidate term-table searches during phrase verification. Position payloads remain compressed inside each document. No journal or checkpoint bytes change. The exhaustive evaluator keeps its independent lookup path.

This is a targeted query improvement with material costs: serving RSS rose **7.4%**, ingest took **6.8% longer**, and a 128-document replacement batch took **137.3% longer**. Adding an early-sorting term changes surviving term ordinals even when their frequencies stay unchanged. Deletion took 1.8% longer.

All numbers below are development measurements, not standings. Parent: `7db61aa01b19d4ac6e4b5fec15526025a30fd17c`; production: `7e216c6a5afd5a0f518c619fb95b1bb990e08aab`. Machine: Intel Core Ultra 9 185H, 22 logical CPUs, Rust 1.99.1, native release build, one query thread pinned to CPU 0. Builds used two nice-19 jobs on CPUs 18–19.

The query kernel uses the existing 600k-line Wikipedia index and the first 16 queries of each primary category in original order: 82 queries across TOP_10, TOP_100_COUNT, and COUNT. Five rounds keep both engines open and alternate their execution for each query. A parent-calibrated repeat count targets 50 ms per cell (5–500,000 repeats). All timing holds the shared lock. Start/end load averages ranged 4.66–5.74. Every returned document ID, score, and count matched.

The table reports the median of five paired candidate/parent ratios of mean query latency; values below 1 are faster. These ratios are not ratios of separately computed latency medians.

| Category | TOP_10 | TOP_100_COUNT | COUNT |
|---|---:|---:|---:|
| term | 0.978 | 1.032 | 0.987 |
| union | 1.007 | 0.952 | 1.031 |
| intersection | 1.034 | 0.999 | 1.030 |
| intersection_union | 1.025 | 1.048 | 1.004 |
| negated | 0.997 | 0.999 | 1.040 |
| phrase | 0.308 | 0.417 | 0.328 |
| two-phase-critic | 0.790 | 0.713 | 0.706 |

The original three sequential-open pairs remain in the raw evidence. They showed large swings in unchanged controls and are not the primary query claim. Their initial/ending loads ranged 4.98–9.83; the later per-query confirmation above reduces drift between processes. No production variant or query sample was changed.

With both 600k processes open, RSS snapshots were 1,700,624 KiB for the parent and 1,826,420 KiB for the candidate. The mutation kernel uses the first 10,000 nonempty corpus rows, positions enabled, whole documents, no stemming, batches of 256, and flushes included. It replaces the first 128 documents by prepending `aardvark `, then deletes them. Inputs are prepared before timing. Three interleaved A/B, B/A, A/B pairs use four threads on CPUs 0–3 under the same lock, with loads 6.74–7.07.

| Mutation/storage metric | Median paired change |
|---|---:|
| Ingest | +6.8% |
| Replacement batch | +137.3% (about 18.4 → 43.7 ms) |
| Delete | +1.8% |
| RSS after 10k-row ingest | +6.2% |
| RSS with both 600k query processes open | +7.4% |

Validation: the complete local fmt/clippy/default test gate passed on Apple M4 / Rust 1.96, including 553 unit tests, all integration tests, golden files, format layout, upgrades, and doc tests. Initial source CI passed 68 checks with 3 skipped and no failures. The independent token-window protocol oracle passed 17,478 answers. New regressions cover repeated words, mixed analyzers, 127/128/129 postings, block splits and merges, changed sorted ordinals with unchanged frequencies, changed positions, deletion, slot reuse, and replay. A separate invariant proves positional metadata leaves checkpoint bytes unchanged.

The 600k corpus passed 962 queries × 3 optimized commands against the independent exhaustive evaluator, with zero mismatches. All 962 full-corpus COUNT answers also matched saved Tantivy counts. The full check ran untimed on Apple M4/arm64 with four threads and nice19 after all ten copied index files and the query/reference files matched their source SHA256 values; source commit remained 7e216c6a. Peak RSS was 12.7 GiB and opening took 272.6 seconds, recorded only for resource accounting. The x86 protocol and exhaustive checks remain separate receipts. The efficient verifier computes one exhaustive top100+count reference per query and derives its top10 prefix and count; all 2,886 optimized results are compared. Exhaustive sorts the full result before truncation, so reference reuse does not remove a comparison.

Raw receipts, scripts, adapter snapshot, source hashes, and summaries are in [the development artifact directory](../../benchmarks/results/sbg_positions_development/). Extract `timing-evidence.tar.gz` there to inspect every original and confirmation round. `setup.sh`, `paired.py`, `paired_ingest.py`, `persistent.py`, and `correctness.py` reproduce the work in `~/hms-sbg-positions` using separate parent/candidate source archives and copied indexes.

Unmeasured: full-corpus x86 latency and memory, full indexing cost, other CPUs beyond the reported checks, and interaction with SIMD posting decode. This revision establishes no new leaderboard standing. After the freeze, repin and rebuild the HMS adapter at the merged commit, then run the standard harness:

```sh
cd ~/sbg-hms/sbg
flock ~/hms-timing.lock make bench ENGINES='hms-main tantivy-0.26 lucene-10.4.0 pisa-0.8.2 iresearch-26.03.1' COMMANDS='TOP_10 TOP_100 TOP_1000 TOP_100_COUNT COUNT' QUERIES=queries.txt
```
