# WebAssembly runtime

`@holographic-memory/wasm` runs the document store locally in browsers, Node, Deno and Bun. It is the
same engine code as the native and Node bindings: ingest, stemmed BM25 passage search with metadata
filters, `titleWeight`, `bm25K1` / `bm25B`, optional caller-supplied dense embeddings with blend or RRF
fusion, delete, stats, and a serialized store you can export, import and persist. No service calls.

Package documentation (entry points, usage, storage, encryption): `bindings/wasm/npm/README.md`.
Encryption decision and measurements: [ENCRYPTION.md](ENCRYPTION.md).
Browser demo: `examples/wasm` (see its README section below).

## What is and is not the engine

The wasm module runs `MemoryDocumentStore` (`src/core/documents/memory.rs`): the engine's document
chunking, term statistics, postings, filters and dense index, without the journal, shards, file locks
or mmap that the native `HmsCore` adds for durability and multi-process use. A test
(`matches_the_engine_document_path`) asserts its search results are byte-identical to `HmsCore`'s, and the
benchmark compares top-10 digests across wasm, the native store and `HmsCore`.
Not in the wasm build: the vector/graph/knowledge-graph engine, local model stages, the HTTP server,
signing and provenance.

The core crate compiles for `wasm32` by gating the file- and thread-bound modules
(`engine`, `storage`, `admin`, `store_lock`, ...) with `cfg(not(target_arch = "wasm32"))` in
`src/core/mod.rs`; `fs2` is a non-wasm dependency. A new module on the document path must stay
wasm-clean: check with
`cargo check --manifest-path bindings/wasm/Cargo.toml --target wasm32-unknown-unknown`.

## Building

```sh
cd bindings/wasm/npm
npm install
node scripts/build.mjs          # both variants, all targets, TypeScript layer
```

Needs the `wasm32-unknown-unknown` target and a `wasm-bindgen` CLI that matches `Cargo.lock` (the script
uses wasm-pack's cache; `cargo install wasm-bindgen-cli --version <v>` also works). The script runs one
`cargo build --release` per variant, then `wasm-bindgen` for each target (`web`, `bundler`, `nodejs`,
`deno`), then `wasm-opt -Oz` with an explicit feature list (SIMD only on the SIMD variant, so a scalar
file containing a SIMD instruction fails the optimizer's validation). Output is
`dist/wasm/<target>/<scalar|simd>/` plus `dist/wasm-sizes.json`. Bun uses the `nodejs` target.

Release profile: `opt-level = "s"`, fat LTO, `panic = "abort"`. A trap (including a panic) leaves that
module instance unusable; the store validates every input and imported snapshot so that hostile bytes
return errors instead of reaching an assertion.

## SIMD and threads

Two builds ship: scalar and `+simd128`. `openStore` validates a probe module with
`WebAssembly.validate` and loads the SIMD build when the runtime accepts it; `simd: false` forces scalar.
Both builds return identical results and identical snapshots (tested). The scalar build assumes the
features rustc enables by default for `wasm32-unknown-unknown` (bulk memory, reference types,
multivalue, sign extension, non-trapping float-to-int, mutable globals).

There is no threaded build. The document path is single-threaded (ingest of a batch is sequential), so
threads would need `SharedArrayBuffer`, cross-origin isolation headers and a nightly `build-std` rebuild
for no gain. Run the store in a Worker to keep the page responsive during large ingests.

## Persistence

A store saves as one snapshot (magic `HMSD`, checksum, LZ4-compressed JSON) under a key. A
`StorageAdapter` is `get`, `set` (atomic replace), `delete` and optional `keys`:

| Adapter | Where | Atomicity |
| --- | --- | --- |
| `IndexedDbStorage` | browsers | one put is one transaction |
| `OpfsStorage` | browsers (origin private file system) | `createWritable()` swap file; Worker fallback uses a sync access handle |
| `FileStorage` | Node, Deno, Bun | write temp file, fsync, rename |
| `MemoryStorage` | anywhere | n/a |

Autosave (default on when a storage adapter is given, 500 ms debounce) re-exports the whole store, so
its cost grows with store size; set `autosave: false` and call `save()` at your own cadence for large
stores. `exportSnapshot()` / `openStore({ snapshot })` move a store between hosts.

## Tests

| Surface | Command (from `bindings/wasm/npm`) |
| --- | --- |
| Rust unit tests incl. forged-snapshot rejection | `cargo test --lib documents::memory` (repo root) |
| wasm-bindgen-test, headless Chrome and Firefox | `node scripts/wasm-test.mjs` |
| Node | `npm run test:node` |
| Deno | `npm run test:deno` |
| Bun | `npm run test:bun` |
| Browser pages and the demo, offline, Chrome and Firefox | `HMS_FIREFOX=... node test/browser/run.mjs` |
| webpack 5 (`bundler` and `web` entries) in Chrome | `node test/bundler/run.mjs` |
| Packed tarball installed and imported from Node, Deno, Bun, plus type-check | `node scripts/pack-check.mjs` |

The Node, Deno and Bun runners execute one shared suite (`test/shared/suite.mjs`).

## Demo

`examples/wasm` is a static page: add notes or text files, search with BM25 knobs, persist to OPFS or
IndexedDB with an optional passphrase, download or restore a snapshot. A service worker caches the app so
it reloads with the server stopped. `node examples/wasm/serve.mjs` serves it on localhost (a secure
context, which OPFS and service workers need) with the built package at `/pkg/`.

## Benchmarks

`benchmarks/results/wasm_v1.json`: synthetic Zipf-like corpus (100 words per document, 20 000-word
vocabulary, 3-term queries sampled from the corpus), wasm under Node and headless Chrome/Firefox against
the native in-memory store and the native `HmsCore`, with wasm module size (raw, gzip, brotli), module
load time, ingest rate, search latency percentiles, memory, snapshot size, export and import time, and
top-10 agreement. Regenerate with `node bench/run.mjs` in `bindings/wasm/npm` after a build.
