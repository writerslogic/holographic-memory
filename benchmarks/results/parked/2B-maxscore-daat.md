# Parked: exact MaxScore document-at-a-time lexical ranking under the store lock (2B lever, 2026-10-19)

Patch: `2B-maxscore-daat.patch` (apply to `src/core/documents/search.rs` and `index.rs` at the V38 tree).
Mechanism: for lexical-only queries, rank under the read lock by merging only the posting lists whose
score bound can reach the current limit-th score, galloping into the rest, summing in query-word order so
scores are bit-identical to scoring every candidate (28 document tests and a 48-query scan-reference test passed).
Measured on the NFCorpus campaign input (3633 docs, 324 queries, k=100, candidateLimit=100), Apple M4, load 8-11,
`cargo test --release --lib documents::search::latency_bench -- --ignored` with `HMS_LEX_BENCH_INPUT`:
materialized capture (kept) total mean 27.9-32.0 us, p99 95.6-122.5 us; MaxScore total mean 26.2-33.9 us, p99 138-181 us.
Why parked: the campaign's queries match 68 chunks on average, so the limit-th score rarely exceeds a bound and the
merge does strictly more work per candidate than the dense-slot materialization; the mean is within noise and p99 is worse.
Revisit only with a corpus where candidate sets exceed roughly ten times the candidate limit.
