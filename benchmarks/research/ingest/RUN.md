# Ingest encoding: memoized encoder and `IngestOptions::encode` (R2 package `ingest`)

Written against origin/main 58e19f4 in the worktree, no build, test or run executed (R2 rule).
Everything below that is a number is an estimate unless it cites `INGEST-PROFILE.md`.

## What changed

1. `src/core/encoding.rs`: `TextEncoder::encode` (line 70) now runs `encode_cached` (414)
   through per-thread memo tables (`EncodeState`, thread-local): base vectors memoized by seed
   (`term_span` 349), character n-gram bound vectors memoized by exact `(scale, chars)` key
   (`ngram_span` 233), `permute` as a rotation of the sorted set (`rotate_sorted` 376), and
   each level bundled from a histogram over `[0, dim)` instead of a sort (`bundle_flat` 283).
   The pre-change formulation stays as `TextEncoder::encode_reference` (89, `#[doc(hidden)]`)
   and the memoized path is reachable as `TextEncoder::encode_text` (81). The module head
   states the five invariants that make the two paths produce identical indices; the one that
   needs care is `select_nth_unstable_by` in `bundle_flat` (same std call, same slice contents,
   same tie behavior as `EntangledHVec::bundle`, `src/core/entangled.rs:403-406`).
   Memory: at most 4 MiB (terms) + 8 MiB (n-grams) per thread, cleared when exceeded; output
   never depends on cache state.
2. `src/core/documents.rs`: `EncodeMode { Eager (default) | Deferred | Lexical }` (76),
   `IngestOptions { encode }` (115), `VectorState { Encoded | Pending | Lexical }` (124) stored
   per chunk as `StoredChunk.vector_state` (268, `#[serde(default, skip_serializing_if =
   is_encoded)]`), `prepare_with_options` (516; `prepare` and `prepare_with_keys` delegate with
   `Eager`), `document_fingerprint` (648).
3. `src/core/documents/index.rs`: `vector_state` serialized only when not encoded (105);
   `pending_chunks` / `lexical_chunks` counters kept by `insert_shared` (338), `replace_shared`
   (438) and `remove`; `encode_gate` (169) serializes lazy encoders.
4. `src/core/engine/mutation.rs`: new journal mutation `EncodeDocument { id, version,
   fingerprint, vectors }` (38), validated at 239, applied by `install_deferred_encoding`
   (551). `stale_chunks` (103) and the chunk list handed to the shards (357) skip chunks
   without an encoded vector, so a pending or lexical replacement drops the old shard entry.
5. `src/core/engine/document_api.rs`: `memorize_document_with` (41),
   `memorize_documents_with` (76), `encode_pending_documents` (203, batches of `ENCODE_BATCH`
   = 1024 documents per transaction), `train_document_index` encodes first (171).
6. `src/core/engine/query.rs`: `query` encodes a pending backlog before searching (18-23, a
   failure is logged and retried next call), `explain_query` appends the pending and
   lexical-only chunk counts to `rationale` (53), `maintain_indices` encodes (118).
7. `src/core/format/scan.rs`: `MUTATION_VARIANTS` has 14 entries, `DocumentSummary.indexed`
   counts chunks the engine indexes (`vector_total` uses it, so `verify`'s live-vector
   cross-check at `src/core/format/verify.rs:1173` still matches), `check_encode_document`,
   and chunk `vector_state` validation.
8. `docs/file-format/MUTATIONS.md`: the `EncodeDocument` row and the `vector_state` field.
9. `tests/ingest_encode.rs`: bit-identity of memoized vs reference over a fixed corpus at
   D = 1, 255, 256, 257, 1024, 4096, 16384 for both text encoders; the three modes keep every
   lexical result, defer or skip vectors, survive reopen and compaction, replacement while
   pending, and reject deferred encoding without stored text.
10. This crate (`src/main.rs`): timing of the three modes with search parity.

## Commands

```sh
cd /Volumes/A/holographic-memory            # or the worktree holding this branch
cargo fmt --all                               # rustfmt was not run on the new code
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --all-features -- -D warnings
cargo test --test ingest_encode               # the new test, both functions
cargo test --test encoder_versions            # pinned vectors at density boundaries
cargo test --test documents --test format_golden --test format_crash --test document_dense
cargo test --lib encoding                     # unit tests incl. memoized_path_matches_reference
cargo test
cargo test --all-features
```

Timing, same harness as `INGEST-PROFILE.md` (release build, `HmsCore::new(1024, store, None)`,
batches of 512, `examples/competitive_text.rs:67-75`); `$INPUT` is the hms-bench
`{documents, queries}` JSON for NFCorpus:

```sh
cargo build --release --example competitive_text
for i in 1 2 3 4 5; do
  rm -rf /tmp/hms-ingest-$i
  ./target/release/examples/competitive_text build --input "$INPUT" \
    --store /tmp/hms-ingest-$i --out /tmp/hms-ingest-$i.json
  jq .ingest_s /tmp/hms-ingest-$i.json
done
```

The three modes, with parity against an eager store (pass = both parity flags true):

```sh
cd benchmarks/research/ingest
cargo build --release
for mode in eager deferred lexical; do
  ./target/release/hms-research-ingest --input "$INPUT" --encode $mode \
    --repeats 5 --warmups 1 --batch 512 --dim 1024 --out /tmp/ingest-$mode.json
done
jq '{mode, median_ingest_s, median_catch_up_s, chunks, vectors_after_ingest, parity}' /tmp/ingest-*.json
```

`--store-text auto` (default) keeps the input's `storeText` except in deferred mode, which
needs the text and forces it on; the eager harness input has `storeText=false`, so for a
like-for-like deferred comparison also run `--encode eager --store-text true`. Output schema is
in the crate's module head.

Serving RSS of a lexical or deferred store through the existing orchestrator needs a mode
switch in the example; a mediator-owned 3-line change would be, in
`examples/competitive_text.rs:71-73`:

```rust
let options = holographic_memory::core::documents::IngestOptions {
    encode: std::env::var("HMS_ENCODE").as_deref().unwrap_or("eager").parse()?,
};
for chunk in input.documents.chunks(512) { core.memorize_documents_with(chunk.to_vec(), &options)?; }
```

## Root-crate APIs relied on (file:line, pre-change numbering where unchanged)

- `EntangledHVec::new_deterministic` entangled.rs:43, `from_indices` 117, `indices` 240,
  `bind` 300 (`sorted_symmetric_difference` 564), `permute` 309, `bundle` 356,
  `bundle_bloom` 422, `DEFAULT_RHO_DENOM` 12, `hash_u64` 556, field `dim` 33 (pub(crate)).
- `TextEncoder::encode` (was encoding.rs:35), `encode_text_internal` 94, `hash_str_seed` 201.
- `prepare_with_keys` documents.rs:416 (now `prepare_with_options`), the encode call 479
  (now 598), `StoredChunk` 177, `chunk_document` 302, `terms` 388.
- `DocumentStore::insert_shared` index.rs:288, `replace_shared` 387, `remove` 491,
  `chunk_text` 177, `shared_values` 172, `text_of` 103, `ChunkWithText::serialize` 84-100.
- `Mutation` mutation.rs:19, `commit` 129, `commit_with_append` 133 (one validate, one
  serialize, one `append` 153 = one journal frame + flush per call, then `apply_mutations`
  156), `prepare_mutations` 298 (chunk vectors to shards 311-315), `apply_mutations` 335
  (Document arm 388-413), `stale_chunks` 91.
- `memorize_documents` document_api.rs:51-99 (`into_par_iter` 72, single `commit` 97),
  `train_document_index` 140, `remove_document_from_memory` 207.
- `ShardSet::insert` shard.rs:404, `remove` 411 (missing id returns `Ok(false)`, 100-104).
- `HmsCore::query` query.rs:12, `explain_query` 46 (`QueryExplanation.rationale`
  types.rs:45), `maintain_indices` 96, `encode_text` engine/mod.rs:561, `vector_count` 723.
- `MutationGate::read` mutation_gate.rs:89, `begin_write` 97 (not reentrant: nothing holding
  `read()` may commit; `query` calls `encode_pending_documents` before taking its read guard
  and only when the pending count is nonzero).
- Format scanner: `MUTATION_VARIANTS` scan.rs:234, `Replay::apply` 290, `check_document`
  400, `check_mutation` 437, `vector_total` 278 (used by verify.rs:700 and :1173).
- Harness: `examples/competitive_text.rs:42-53` input schema, 67-75 build phase.

## Mediator levers

(a) Parallel encode with order-stable merge: already the shape of the batch path.
`memorize_documents` prepares documents with `into_par_iter().map(..).collect::<Result<Vec<_>>>()`
(document_api.rs:71-88 before, 99-118 now); rayon's indexed collect keeps input order, and the
encoder's memo tables are per-thread pure caches, so the stored vectors are independent of the
thread count. The equivalence test calls the encoder directly (single thread).

(b) k-way merge instead of accumulate-then-sort: the bundle only needs the ascending
`(index, count)` run-length list of the multiset, which a histogram over `[0, dim)` yields in
`O(total + dim)` with no comparisons, strictly cheaper than a heap merge's `O(total log k)`
whenever `dim <= 8 * total` (`COUNTING_RATIO`, encoding.rs:166); above that the fallback sorts
the (then short) list. Both produce the identical list; a merge would too.

(c) One commit per batch: already so. `memorize_documents` issues a single `commit` for the
whole batch (document_api.rs:97 before / 128 now), which validates, serializes the batch into
one transaction payload and appends one journal frame (`mutation.rs:133-160`, `append` at 153
before / 176 now). The serial part the profile saw (~0.4-0.5 s wall, 4.1% of CPU samples) is
not fsync but the in-memory install under the write gate: `apply_mutations` ->
`DocumentStore::insert_shared` -> `PostingLists::insert` per term per chunk
(index.rs:338-342 before / 389-391 now) plus `shards.insert` per chunk (mutation.rs:409-411).
That is the next lever and is untouched here.

## Expected effect per mode (inference from the profile, to be replaced by the measurement)

Profile baseline: 1.70 s wall = ~1.2 s parallel `prepare` (93% of 12.2 CPU-s is
`TextEncoder::encode`) + ~0.45 s serial install/commit.

- `eager` (default; bit-identical vectors): per character the encoder now does four n-gram
  memo lookups (~25 ns each, 24-byte key hashed with FxHash), ~56 histogram increments and six
  1024-slot scans per chunk; roughly 150 ns per character against ~2 µs before (14
  `new_deterministic` of 4 hashed indices each, 14 allocations, four sorts). Estimate: encode
  CPU 11.3 s -> ~0.9 s, prepare wall ~1.2 s -> ~0.1-0.15 s; ingest ~0.60-0.65 s (2.6-2.8x).
  The remaining time is the serial install (~0.45 s) plus tokenize/stem (0.3 CPU-s, parallel).
- `deferred`: prepare is tokenize/stem only; ingest ~0.5 s. Catch-up
  (`encode_pending_documents`) then costs about the eager encode (~0.1-0.15 s wall) plus one
  commit per 1024 documents; it runs on the first `query`, `maintain_indices`,
  `train_document_index` or an explicit call, and the store needs the chunk text (text
  segment writes add to the serial part when the eager run had `storeText=false`).
- `lexical`: same as deferred without the catch-up, ~0.5 s.

So with these changes alone HMS lands near 2x tantivy (0.258 s) and ~5x FTS5 (0.090 s); the
serial install is the floor and must drop to about 0.05 s to beat FTS5. Nothing here changes a
search result: lexical results are identical in every mode (asserted by the test and the
harness parity), eager vectors are bit-identical, and deferred vectors are the eager vectors
computed later from the same text.

## Serving RSS effect of deferred or lexical storage (estimate, D = 1024, 4 active indices)

Per chunk the vector costs: document-store copy `EntangledHVec` 32 B inline + 16 B heap
(skipped entirely: ~32-48 B with allocator rounding); shard `vectors: HashMap<String,
EntangledHVec>` entry (key String 24 B + ~25-35 B id, value 48 B, table overhead) ~120 B;
`registry` id String ~55 B; `slots` map entry ~70 B; inverted index 4 x 4 B = 16 B. A lexical
or pending chunk has none of the shard side: roughly 260-310 B per chunk, i.e. ~1.3-1.6 MB for
NFCorpus if it holds ~4,700 chunks (`chunks` in the harness output gives the exact count). The
journal is not smaller (`"vector":{"dim":1024,"indices":[]}` plus `"vector_state":"lexical"`
is ~10 B longer than four indices). Whether that flips 1.07x SQLite depends on the absolute
RSS; measure with the `HMS_ENCODE` switch above through the orchestrator's `serve` phase.

## Format notes for the mediator

- Eager stores serialize byte for byte as before (`vector_state` is skipped when encoded;
  `tests/format_golden.rs` is the check). A store that holds deferred or lexical documents
  carries `vector_state` on those chunks and `EncodeDocument` frames after a catch-up; an
  older binary ignores the unknown chunk field but refuses the unknown mutation variant on
  replay, which is the "versioned" part: such stores need this build.
- Per-chunk rather than per-document flag, following `stemmed`; a document is pending iff any
  chunk is (ingest sets all chunks of a document alike).
- `EncodeDocument` applies only when the live document still has the same version, chunk
  count and content fingerprint (u64 over version, byte spans, chunk text through the crate's
  splitmix64 chain) and pending chunks; otherwise it is a no-op. The decision uses live state
  only, so replay reaches the same state. The scanner (`scan.rs`) cannot check the
  fingerprint without text and assumes it matches; a document replaced with identical
  version and chunk count while a catch-up was in flight would make `verify`'s live-vector
  cross-check disagree by that document's chunks (the engine is right in that case).

## Open risks (unverified: nothing compiled)

- Borrow checker on the field-level borrows inside `EncodeState::ngram_span` and
  `encode_cached` (`state.flat.extend(... state.term_arena ...)`); if rustc objects, copy the
  span's slice into `state.rotated` first.
- `install_deferred_encoding` is called from `apply_mutations` with `shards` already locked
  for writing by `commit_with_append`; it takes `documents.read()` then `documents.write()`
  in separate scopes like the `Document` arm.
- The deferred catch-up inside `query` commits; a caller that already holds
  `mutation_gate.read()` and has pending documents would deadlock. In-crate callers of
  `query` (knowledge.rs, mod.rs federated, document_api.rs query_vector, query.rs batch) hold
  no gate. If a new one appears, move the catch-up out of `query` and keep it on
  `maintain_indices`/`train_document_index`/`encode_pending_documents`.
- `tests/ingest_encode.rs` compares `query` results as sets because slot order differs between
  a store encoded at ingest and one encoded later; k = 4096 exceeds the corpus so no
  truncation happens.
- No new third-party crates in the root; this crate uses root versions of anyhow, clap, serde,
  serde_json, tempfile.
