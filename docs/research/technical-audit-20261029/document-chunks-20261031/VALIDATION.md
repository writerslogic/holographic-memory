# Retained chunk reads and text-cache integrity

Base source tree: PR 363 head `79ce353407fee9df816bf60b5fda2a16dc3589cc`,
identical to main merge `68e016dba33eee777e3630ce27ec4b9a953995b3`.
Local macOS arm64, Rust 1.99.0; debug profile, no paid services.

## Reproduced defects

1. Truncate `documents.text` to its eight-byte header after ingestion; force a
   checkpoint at close. The checkpoint writer used infallible `text_of` and
   panicked during Drop (`cache-close-baseline.log.gz`). The preceding fixture
   failure assumed CURRENT existed before first compaction; it is retained but
   is not defect evidence. JSON and checkpoint writers now propagate read errors.
2. Change one plaintext byte from `alpha beta` to `olpha beta`, without changing
   length or UTF-8 validity. Closing and reopening returned the altered text
   (`cache-bitflip-baseline.log.gz`): unchecked cache data entered the checkpoint.
   This run used the new read API and fallible-writer candidate before adding
   checksums. It was not a separate checkout of main. The supplied baseline
   patch uses only preexisting public APIs and can be applied to main 68e016d:
   `gzip -dc baseline-repro.patch.gz | git apply`, then
   `cargo +1.99.0 test --locked --features security --test cache_recovery_regression`.
   That extracted reproduction patch has not separately been executed.

Each process-local cache location now includes a CRC32 from the authoritative
plaintext at append/replay; reads verify it. The checksum is not persisted and
adds no new dependency or durable format. Encrypted records keep AES-GCM.
This is accidental-corruption detection, not hostile-tamper authentication or
protection against CRC collisions. Old infallible get/search APIs can still panic
on cache I/O errors; the new chunk API and persistence writers return errors.

## Read contract and bounds

Rust `read_document_chunk`, tenant/read-authorized HTTP
`GET /api/v1/documents/{id}/chunks/{index}`, and sync/async Python SDK methods
return one retained passage, byte span, metadata and opaque store revision.
Absent records/indices return 404; unretained text is null. Replacement/deletion
or canonical segment replacement causes up to three retries; persistent races
return a typed error / HTTP 503 + Retry-After. File I/O holds no mutation/store
lock; captured document and segment identities are revalidated afterward.
Tests deliberately replace the canonical segment without moving the revision;
full compaction DOES advance the revision. Two preliminary failing assertions
incorrectly assumed otherwise; their logs are retained and are not product defects.

Equal revision tokens across sequential reads provide a consistency check, not
retention of an old snapshot. Reopen randomizes the revision. Chunk spans cannot
restore omitted whitespace/gaps, unretained text or original embeddings; this
is not lossless full-document export or complete Haystack/Semantic Kernel support.
Node and Python native bindings have not gained this method.

Checkpoint decoding enforces ingest bounds before allocating record arrays:
version 1..1024 bytes, URI <=8192 bytes, object metadata <=64 KiB encoded,
1..4096 chunks, valid IDs <=65535 bytes, nonempty spans within 8 MiB, retained
text length equal to the span. Unit tests exercise limits -1/0/+1 and malformed
spans. Native one-chunk text is <=8 MiB; JSON escaping and copies mean neither
HTTP response size nor peak RSS has that bound.

OpenAPI path parameters are required as specified in
[OpenAPI 3.1.0 §4.8.12.2](https://spec.openapis.org/oas/v3.1.0.html#parameter-object).
Generated schema includes nullable text, metadata and revision, and 304 response.
ETags use the existing weak representation hash; they are not snapshot handles.

## Validation

Completed locally: five text-cache integrity units, five read-concurrency units,
five initial chunk/recovery integration tests, ten checkpoint tests (including
purge and encryption), the allocation regression, and SDK wheel/sdist builds.
The later journal-only compaction regression makes six chunk integration tests;
its local rerun, golden-store/API/Python suites, strict Clippy and generated checks
remain pending as this evidence snapshot is committed. Exact-head CI is required.

The allocation fixture contains 18 documents with 64 KiB single-word passages.
Reading one chunk requested 66,233 bytes / eight allocations live and after
compaction, and 66,281 bytes / nine allocations after reopening; the regression
requires <128 KiB. This counts allocation requests, not RSS or retained memory,
and establishes neither HTTP latency nor comparative performance.

Commands (shared target directory is optional):
- `cargo +1.99.0 test --locked --features security --lib core::documents::text::tests`
- `cargo +1.99.0 test --locked --features security --lib chunk_read_tests`
- `cargo +1.99.0 test --locked --features security --test document_chunks`
- `cargo +1.99.0 test --locked --features security --test document_checkpoint`
- `cargo +1.99.0 test --locked --test allocation_bounds metadata_listing -- --nocapture`
- `cargo +1.99.0 test --locked --features security --test format_golden`
- `cargo +1.99.0 test --locked --features server,security --test server_api -- --test-threads=1`
- `cargo +1.99.0 clippy --locked --features server,node-api,security --lib --bin warm-server --tests -- -D warnings`
- `RUSTUP_TOOLCHAIN=1.99.0 bash scripts/check-generated.sh`
- `uv build --package warm-sdk` from `bindings/python/warm-sdk`.

