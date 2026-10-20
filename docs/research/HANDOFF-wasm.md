# Handoff: wasm area

Source of truth for numbers: `benchmarks/results/wasm_v1.json` (sweep 1k, 5k, 10k, 25k documents, 100 words each).

## Measured (this machine, Apple Silicon, Node 26, Chrome 155, Firefox 158)

- Module: SIMD build 565 584 B raw, 243 566 B gzip, 196 304 B brotli; scalar 575 847 / 246 627 / 197 813.
- Top-10 results are digest-identical across wasm (Node, Chrome, Firefox), the native in-memory store and the
  native `HmsCore` at every size tested.
- Search p50 at 25k documents: native store 0.045 ms, `HmsCore` 0.051 ms, wasm/Node SIMD 0.079 ms,
  wasm/Chrome 0.1 ms (timer resolution limits browser figures). Wasm is about 1.5 to 2x slower than native.
- Ingest at 25k: native `HmsCore` 546 docs/s (parallel), native store 109, wasm/Node SIMD 87, Chrome 86,
  Firefox about 67 (10k). Ingest is the weak point: the wasm module is single-threaded and encoding a
  chunk at 16384 dimensions dominates.
- Import (load) of a 25k-document snapshot: 8.4 s in wasm, 3.8 s native. Slow; see below.
- Memory at 25k: wasm linear memory 81 MB; native store peak RSS 255 MB; `HmsCore` 141 MB (different metrics).
- SIMD versus scalar: no consistent difference (noise dominates); the SIMD build is not faster on this workload.
- Encryption: WebCrypto AES-GCM is 20 to 90x faster than in-wasm AES (`docs/runtimes/wasm/ENCRYPTION.md`).

## Verified

Rust unit tests incl. forged-snapshot rejection and exact parity with `HmsCore`; clippy (no features and
`--all-features`, `-D warnings`) clean; fmt clean; MSRV 1.89 `cargo check --lib` clean; wasm-bindgen-test
in headless Chrome and Firefox; the shared suite under `node --test`, `deno test`, `bun test`; browser
pages and the demo (ingest, search, persistence, offline reload with the server stopped) in Chrome and
Firefox on OPFS and IndexedDB; webpack 5 builds of the `bundler` and `web` entries run in Chrome;
`npm pack` tarball installed and imported from Node, Deno, Bun with a tsc type-check.

## Not done / limits

- The scaling cliff is not found: the sweep stops at 25 000 documents because a single point at 50 000 took
  over 10 minutes natively (ingest). Do not extrapolate past 25k.
- Firefox at 25 000 documents timed out in the benchmark page (recorded as a failure in the JSON).
- Import is slow (rebuilds postings through `DocumentStore::insert`, JSON decode). A binary snapshot or
  bulk index build would help; ingest could batch-encode chunks in parallel natively.
- Safari, mobile browsers and Firefox/Safari WebCrypto-versus-wasm AES were not tested.
- No CI job builds wasm32. Add: `cargo check --manifest-path bindings/wasm/Cargo.toml --target wasm32-unknown-unknown`
  (with `RUSTFLAGS='--cfg getrandom_backend="wasm_js"'`).
- Not published (by instruction). Publish: `npm publish --access public` from `bindings/wasm/npm` after
  `node scripts/build.mjs`.

## Next command

`cd bindings/wasm/npm && npm install && node scripts/build.mjs && node --test test/node/`
