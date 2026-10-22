<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Exact Boolean lexical execution

Baseline: `08d37778a66b58709461456e5b9a7ab6a05b399f` (2026-10-22).
The existing lexical executor visits every query posting, accumulates BM25 in a slot
array, and selects chunks. Lists contain delta/frequency varints without skipping
or positions. Deletes physically remove postings; replacement may reuse slots.
Captures pin document owners while the mutation gate and document read lock establish
one visible generation. Existing plain-text/hybrid APIs retain their semantics.

## Query contract

Add explicit lexical query APIs, separate from plain-text/hybrid search. The parser
accepts flat optional, required (`+`) and prohibited (`-`) clauses, each a word or
quoted phrase. Required clauses all match; optional clauses affect scores and are
required only when there is no required clause. Prohibited clauses exclude matches.
An empty or solely prohibited query matches nothing (Lucene Boolean semantics).
Missing optional/prohibited terms do not constrain a query; a missing required term
makes it empty. Duplicate clauses are idempotent. Contradictions match nothing.
Unsupported syntax fails explicitly, including nonzero phrase slop, fields, groups,
wildcards and boosts; none occurs in the benchmark's 962 queries.

The benchmark's `intersection_union` means MUST plus optional scoring, not a
conjunction of all words. `two-phase-critic` is `+"the who" +uk`: intersect rare term
candidates before verifying an adjacent, ordered phrase. Phrase terms retain order,
repeats and token offsets. Slop is zero. Overlapping phrase occurrences count.
COUNT never scores and counts live parent documents once. Ranked hits are chunks,
ordered by raw BM25 descending, parent document ID ascending, then chunk index.
`k=0` returns no hits, with an exact count when requested.

The analyzer is HMS's Unicode alphanumeric splitter, lowercase and optional English
Snowball stemming. Benchmark ingest disables stemming. On its lowercase ASCII
corpus this matches the whitespace analyzer except external token-length filters.
BM25 is `idf * tf * (k1+1) / (tf + k1*(1-b+b*length/avg))`, defaults 1.2/0.75,
`idf=ln(1+(N-df+.5)/(df+.5))`, using current live chunks and exact lengths.
Term clauses score once; phrases use their occurrence frequency and the sum of their
terms' IDFs, including repeated phrase terms. Optional and required clauses score;
prohibited clauses do not. Nonfinite scores are rejected. Parameters and sizes are
validated. Lucene/Tantivy norms, float width and duplicate-query rewrites differ;
score/rank parity is claimed only for equivalent configurations.

Ordinary ingest retains HMS's established chunk statistics and match unit. The
benchmark enables opt-in `IngestOptions::whole_document`, requiring lexical encoding
and positions, so each article is a single chunk up to the existing 8 MiB document
limit. This preserves article-level conjunctions, exclusions and phrases across the
old 4,096-word boundary. The larger chunk is accepted only with version-2 positional
postings on one lexical chunk; ordinary 64 KiB chunk limits remain unchanged.

## Reference and incremental implementation

First add a full-scan reference evaluator over stored term tables and token positions,
with full-sort top-k. Test the parser independently using truth tables. Add seeded
random differential tests over mixed analyzers, replacement, deletion, slot reuse,
reopen, missing/repeated terms, ties, k=0 and 127/128/129-posting boundaries before
selecting a pruned executor. CI owns compilation, tests, features and MSRV.

Postings become blocks of at most 128 entries, with independently bitpacked document
gaps and frequencies, block endpoints and maximum frequency. New cursors implement
next and monotone advance, skipping blocks by their last slot. Mutation keeps bounds
conservative; legacy journal replay constructs the metadata from live terms. Blocks
are derived memory state, not a new interpretation of any persisted bytes.

The first optimized executor uses document-at-a-time union, rarest required-term
lead with advance for conjunctions, and exclusion cursors. Exact COUNT uses the same
matching pass without scoring. A df shortcut is legal only for a single term with
one chunk per live document and no visibility/filter ambiguity. Candidate selection
buffers at most approximately 2k and uses select_nth_unstable, followed by deterministic
sorting; equal scores cannot escape the tie-break.

Only after differential tests pass, add conservative block-max skipping to disjunctions.
Bounds use live IDFs, maximum tf and a lower bound on length (zero is always valid),
nonnegative finite boosts/parameters, and outward floating-point rounding. If an
arithmetic or sign precondition fails, use exhaustive execution. Skip only when the
rounded sum is strictly below the kth score; equality is never pruned. MaxScore is
not promoted without repeatable measurements showing a benefit.

## Positions and compatibility

`IngestOptions::positions` defaults off. Store delta-coded positions alongside packed
terms, with a versioned optional positional payload. Existing non-positional JSON and
binary records keep their bytes. Binary positional records set a previously reserved
chunk flag and include a positional subformat version (1 for ordinary chunks,
2 for whole lexical articles); old readers reject that flag.
New readers reject unknown versions, truncated/overflowing positions, inconsistent
term offsets and allocation sizes. JSON uses an explicit versioned envelope only for
positional terms. Legacy chunks never acquire inferred positions: phrase queries on
an index without positions return an explicit capability error. Old stores and new
non-positional records remain readable and writable without conversion.

Add immutable byte fixtures for both forms, round trips, prefix truncation tests,
and old-store loading; do not change existing golden files. Position opt-in must
survive compaction, replacement and reopen. Snapshot queries observe one generation.

## Measurement and delivery

Use an isolated fork checkout and target directory, at most two release builds from
CI-passing HMS commits, and the shared remote timing lock for all heavy work. Reuse
baseline JSON receipts; baseline data, indexes and binaries are currently absent on
writerslogic-02 and must be reacquired before comparable timings. Record dataset,
queries, analyzer, hardware, threads, cache policy, index time/size and RSS. At least
three timed repetitions per type; report median and min/max, including regressions.
No performance run with failing CI and no claim of competitive success on missing
or incompatible evidence. Main PR and upstream #99 remain draft; do not merge.

Approved cross-ownership integration includes the lexical API additions in
`src/core/engine/document_api.rs` and scoped whole-article validation in
`src/core/engine/document_batch.rs` and `src/core/engine/mutation.rs`. Graph and Python code remain outside this change.
