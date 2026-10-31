# Ingest postings: bulk index build and one-pass commit (R2 package `ingest`, part 2)

Written against origin/main e2f353d in the worktree, no build, test or run executed (R2 rule).
Numbers below are estimates unless they cite `out/*.json` or `INGEST-PROFILE.md`.

## Where the 0.5 s went (read, not measured)

`out/ingest-lexical.json`: 3,633 documents = 3,633 chunks, batches of 512 (8 transactions),
median 0.602 s with no vectors built. The lexical path after tokenize/stem was:

1. `PostingList::insert` (postings.rs, old 230-238) decoded and re-encoded the whole list for
   every posting added: a term with document frequency `df` cost `O(df^2)` over the corpus.
   For the few hundred corpus-wide terms of NFCorpus that is on the order of 10^8 varint
   operations per batch sequence, i.e. the bulk of the 0.5 s. One dictionary lookup per
   (chunk, term) as well, all serial under the store write lock.
2. The batch was serialized to JSON twice, serially: once to measure it
   (`document_batch::validate_prepared`), once for the journal frame (`commit_with_append`).
   About 10 MB of JSON per corpus, two passes, single thread: ~0.05-0.1 s.
3. `validate_inputs` chunked every document serially before the parallel prepare chunked it
   again; `validate_mutation` ran serially; `prepare_mutations` took one store read lock per
   document; `apply_mutations` took one store write lock per document.
4. One journal frame per batch was already the case (`commit_with_append`, one `append`), and
   remains so: the frame bytes are unchanged.

What the code already had of the tantivy design the coordinator described: the compact sorted
term dictionary with an overlay folded in at a quarter of the base, and delta-varint posting
lists. What it lacked: per-thread term accumulation, per-term delta buffers appended once per
batch, and one dictionary merge per batch. Those are what this change adds.

## What changed

1. `src/core/documents/postings.rs` (module head documents the phases):
   - `index_chunks(postings, &[(slot, &StoredChunk)])`: contiguous chunk ranges scanned in
     parallel into per-range `FxHashMap<&str, Vec<(slot, freq)>>` keyed by the chunk's own
     term bytes (no term copied or allocated), bucketed by an FNV hash of the term (16 buckets,
     one bucket below 64 chunks); buckets merge in parallel in range order into one
     slot-sorted run per term, sorted by term; `PostingLists::insert_runs` looks every run up
     once (parallel, read-only above 256 runs), appends it to its list in one delta-varint
     pass, and hands the new terms to `TermDictionary::insert_many`.
   - `PostingList` keeps `last` (slot of its final posting): `append_run` encodes a run above
     it straight onto the byte vector; a run at or below it (slots reused after deletes)
     merges that one list in a single decode/encode. `PostingLists::insert` (single posting,
     used by replacements) takes the same step.
   - `TermDictionary::insert_many`: terms that fit the overlay budget go in one by one; more
     merge with the overlay into a new sorted base in one pass (`merged_base`, which also
     folds tombstones and revives tombstoned terms).
2. `src/core/documents/index.rs`: `DocumentStore::insert_batch(Vec<Arc<StoredDocument>>)`
   strips text in order, then installs maximal runs of new, distinct documents through
   `install_new` (slot assignment chunk by chunk exactly as before: smallest free slot, else a
   new one, so the run's chunks are in ascending slot order; owners, word counts, counters,
   filters; then one `index_chunks`; generation advanced by the run length). A document that
   replaces a stored one, or repeats an id earlier in the batch, breaks the run and takes the
   replacement path at its turn (`replace`). `insert` is `insert_batch` of one. The
   many-frees rebuild in `remove` re-inserts all documents as one batch.
   `replace_postings` skips terms whose frequency did not change (a deferred encoding landing
   or a re-memorized document no longer rewrites every list).
3. `src/core/engine/mutation.rs`: `transaction_payload` builds the frame (magic + JSON array),
   serializing elements in parallel above 16 mutations and joining them with `serde_json`'s
   separators, so the bytes equal `serde_json::to_writer(&slice)`; `commit_serialized`
   takes a payload built by the caller; `validate_transaction` validates in parallel above 64
   mutations and reports the first failure in batch order; `prepare_mutations` holds one read
   lock; `apply_mutations` accumulates consecutive documents and installs them in one
   `insert_batch` under one write lock (the run breaks at a repeated id so the second copy is
   diffed against the first, as one at a time).
4. `src/core/engine/document_batch.rs`: `validate_inputs` computes chunk counts and metadata
   sizes in parallel and folds the budgets in input order (same error, same input);
   `serialize_transaction` replaces `validate_prepared`: the batch is serialized once, checked
   against the 32 MiB limit, and that payload is what `memorize_documents_with` commits.
5. `src/core/engine/document_api.rs`: `memorize_documents_with` calls `serialize_transaction`
   then `commit_serialized`.
6. `tests/ingest_postings.rs`: see below.

Search results and the on-disk format are unchanged: slot assignment is identical to the
sequential path (so captured chunk order, tie-breaking by `(document id, chunk index)`, BM25
sums in query-term order, and candidate truncation are identical), posting lists hold the
same `(slot, frequency)` sets, and journal frames are byte-identical (`tests/format_golden.rs`
and `complete_transaction_bytes_match_legacy_for_all_mutations` in mutation.rs check that).
Nothing in `src/core/documents/search.rs` changed.

## Commands

```sh
cd /Volumes/A/writerslogic-warm            # or the worktree holding this branch
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --all-features -- -D warnings
cargo test --test ingest_postings             # the new test, both functions
cargo test --lib documents::postings          # unit tests: map model, bulk vs single, dictionary merge
cargo test --lib documents                    # store invariants (assert_valid after every insert path)
cargo test --test documents --test ingest_encode --test format_golden --test format_crash \
  --test document_dense --test store_lifecycle --test purge
cargo test
cargo test --all-features
```

Timing, the harness of `RUN.md` (`$INPUT` is the warm-bench NFCorpus `{documents, queries}`
JSON; `--batch 512`, dim 1024). The harness now defaults to one cold build per process
(`--repeats 1 --warmups 0`): the encoder memo tables are per-thread and survive between
builds of one process, so a warm repeat measures cache-hit encoding, not an ingest. Run the
process five times for a cold median; `--repeats 5` in one process gives `cold_ingest_s`
plus `warm_ingest_s`:

```sh
cd benchmarks/research/ingest
cargo build --release
for mode in lexical eager deferred; do
  for i in 1 2 3 4 5; do
    ./target/release/warm-research-ingest --input "$INPUT" --encode $mode \
      --batch 512 --dim 1024 --out /tmp/ingest-$mode-$i.json
  done
  jq -s 'map(.cold_ingest_s) | sort | .[2]' /tmp/ingest-$mode-[1-5].json   # cold median
done
jq '{mode, cold_ingest_s, median_catch_up_s, chunks, vectors_after_ingest, parity}' /tmp/ingest-*-1.json
# warm figure (same process, memo tables full), for the encoder's share only:
./target/release/warm-research-ingest --input "$INPUT" --encode eager --repeats 5 \
  --batch 512 --dim 1024 --out /tmp/ingest-eager-warm.json
jq '{cold_ingest_s, warm_ingest_s, median_warm_ingest_s}' /tmp/ingest-eager-warm.json
# the whole corpus as one transaction (the bulk path at its widest):
./target/release/warm-research-ingest --input "$INPUT" --encode lexical \
  --batch 3633 --dim 1024 --out /tmp/ingest-lexical-1.json
```

The competitive harness (`benchmarks/competitive`) now runs two WARM text rows, each build
repeat a cold process: `warm` (`competitive_text build --encode lexical`, BM25 index only,
like-for-like with FTS5/tantivy/Lucene/bm25s) and `warm-eager` (default encoding); the eager
row is reported and never a verdict comparator (`table.best_other`).

Pass/fail: cold-median `cold_ingest_s` for `lexical` under 0.090 s (SQLite FTS5) with both
parity flags true; eager is reported, not gated (see below).

## Expected numbers (estimates; replace with the measurement)

Baseline (`out/`): eager 0.915 s, deferred 0.640 s, lexical 0.602 s.

- `lexical`: tokenize/stem/chunk in parallel ~0.03-0.04 s wall (0.3 CPU-s in the old profile
  plus `chunk_document`, now parallel in `validate_inputs` too), serialization ~0.01 s wall,
  journal append (8 frames, ~10 MB, no fsync in `write_slice`) ~0.01 s, bulk postings
  ~0.01-0.02 s wall (per-range scan ~0.03 CPU-s over 10 threads, merges parallel, serial
  apply of ~550k postings ~5 ms, dictionary merges ~2 ms per batch). Estimate 0.06-0.09 s,
  i.e. 7-10x over the baseline and at or under FTS5. If it lands at 0.09-0.12 s the residual
  is in `documents::prepare_with_options` (tokenizer, stemmer, `StoredTerms` packing), which
  is outside this package; `--batch 3633` isolates the per-batch fixed costs.
- `deferred`: lexical plus the text segment writes (one `pwrite` per chunk, 3,633 syscalls,
  ~0.01-0.02 s). `median_catch_up_s` drops as well: the catch-up's `install_deferred_encoding`
  replaced every document with unchanged terms, which rewrote every posting list per term
  (`set_frequency`); that is now skipped.
- `eager`: lexical plus the encoder (the measured eager - lexical gap on main is 0.31 s wall,
  which is the memoized encoder's `prepare` plus per-chunk `ShardSet::insert`, ~0.02 s).
  Estimate 0.35-0.40 s. The 0.2 s goal needs the encoder's wall time under ~0.13 s
  (`src/core/encoding.rs`, not in this package); the postings side contributes under 0.02 s
  of it either way. Measure the gap as `eager - lexical` to attribute it.

Memory: `last` adds 8 bytes per term (~0.3 MB at 30k terms). Appended lists grow by amortized
`Vec` doubling (up to 2x slack over the posting bytes, ~1.5 MB at 550k postings) instead of a
`shrink_to_fit` per insert; a list created by a batch is sized exactly. The transient batch
state is ~16 bytes per posting of the batch (the `(slot, frequency)` runs), freed before the
write lock is released.

## Root-crate APIs relied on (file:line on origin/main e2f353d before this change)

- `StoredChunk` documents.rs:256 (`terms: StoredTerms` 268, `stemmed` 273,
  `vector_state` 278), `&StoredTerms: IntoIterator<Item = (&str, &u32)>` packed_terms.rs:201,
  `StoredDocument` 282.
- `DocumentStore` index.rs:13 (`insert` 281, `insert_shared` 338 now `insert_batch` /
  `replace` / `install_new`, `replace_shared` 438, `remove` 552 with the rebuild at 604-616,
  `stripped` 322, `slot_words` 41, `replace_postings` 133).
- `PostingLists` postings.rs:262 (`get` 269, `insert` 279, `remove` 297, `set_frequency` 311,
  `assert_valid` 321), `TermDictionary` 50 (`insert` 89, `rebuild` 125), `PostingList` 170
  (`for_each` 201, `iter` 185, `encode` 217).
- Readers of postings unchanged: search.rs:245, 262-290, 316-334 (`for_each` in slot
  order, `owned_chunk`, `slot_word_count`), documents.rs:812, 840, 886, 1026, 1109.
- `WarmCore::commit` mutation.rs:152, `commit_with_append` 156 (tests call it with injected
  appends, signature kept), `validate_mutation` 185, `prepare_mutations` 339,
  `apply_mutations` 379 (Document arm 432-457), `arena_write` engine/mod.rs:1307,
  `MutationGate::begin_write`/`publish`, replay at engine/mod.rs:546 (same `apply_mutations`,
  so a reopened store takes the bulk path per frame).
- `document_batch::validate_inputs` 59, `validate_prepared` 129 (removed), `serialized_size`
  45, `MAX_TRANSACTION_BYTES` 15; `memorize_documents_with` document_api.rs:77-131.
- Crates: `rayon` (`par_iter`, `par_chunks`, `into_par_iter`, `par_sort_unstable_by`,
  `current_num_threads`), `fxhash::FxHashMap` (both already root dependencies). No new
  dependency; no Cargo change.

## Test (`tests/ingest_postings.rs`)

- `batching_does_not_change_search_results_live_or_reopened`: a 150-document corpus
  (10..=59 words each, head-heavy vocabulary, punctuation, non-ASCII, several chunks per
  document) memorized in one batch, in batches of 7 and one document at a time; 50 generated
  queries (1..=4 words, some with an absent token) under five option sets (deep unfiltered
  pool, shallow cut, metadata filter, source filter, id filter) serialize to identical JSON
  (id, document id, offsets, text, score, lexical and semantic score) across the three
  stores, and again after each is reopened (two from the journal, one after `compact`).
- `reused_slots_replacements_and_deferred_catch_up_match_a_fresh_store`: every third
  document deleted and re-added (half one at a time, half as a batch) into freed slots, two
  replacements (in place and with a changed chunk count), and a deferred store after
  `encode_pending_documents`, all equal to a fresh store holding the same documents, live
  and reopened.

## Open risks (unverified: nothing compiled)

- Lifetimes in `postings::index_chunks`: `RunMap<'a>` keys borrow the chunks' term bytes
  through `&'a StoredChunk`; `scan_range`/`merge_bucket` are generic over `'a`. If rustc
  wants an explicit bound on the `par_chunks` closure, name it `for<'r> ...` or move
  `buckets` into the closure.
- `index.rs::install_new` borrows `&StoredChunk` from the local `group` while mutating
  `self`; the group is moved into `self.documents` only after `index_chunks` returns.
- `apply_mutations` matches `(mutation, &prepared)` with `matches!` before the by-value
  `match`; both patterns use default binding modes.
- `validate_transaction` reports the first error in batch order from a `Vec<Option<Error>>`;
  if clippy objects to `map_or(Ok(()), Err)`, write it as a `match`.
- `document_batch::serialize_transaction` reports "document batch size limit exceeded" from
  an `ensure!` rather than through a `serde_json` io error; no test matched on that text.
- The text segment still takes one `pwrite` per chunk (`TextSegment::append`, text.rs:70,
  not in this package); an `append_many` that seals records at precomputed offsets and
  writes one buffer would remove ~3,600 syscalls per corpus for `deferred`/`storeText`.
- Dictionary ids for new terms are assigned in merged-run order (sorted by term within a
  bucket, buckets in hash order), deterministic for a given corpus and batching; ids are not
  observable through search.
