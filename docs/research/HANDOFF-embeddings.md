# Handoff: local embeddings (branch feat/embeddings)

## Done and verified
- `src/core/embed/`: pure-Rust BERT/RoBERTa/XLM-R engine, int8 (grouped scales)/f16/f32 packs, sha256-verified
  loader, pinned registry, offline/env/cache resolution, optional download (`local-embed-download`),
  `SemanticIndex`, `hms-embed` tool. 15 unit tests + `tests/embed_store.rs` (4) pass; clippy `-D warnings`
  clean with and without features (`local-embed-download,python`).
- `TextEncoder::LocalEmbedV1` + `HmsConfig::local_embed` + `HmsCore::set_text_embedder`; existing encoders
  untouched. Default build: no new dependencies (`cargo tree` shows none of tokenizers/libm/ureq/sha2).
- Python: wheel built, installed in a fresh venv, `Embedder`/`SemanticIndex` verified offline; companion
  `holographic-vsa-models` wheel (30 MB) auto-discovered with no env vars.
- Engine vs fp32 reference: min cosine 0.99963 (granite int8), bar 0.999. Results:
  `benchmarks/results/embedding_selection_v1.json`, `embedding_int8_group_v1.json`.

## Not done
1. Node runtime check: the napi build compiles (typings in `index.d.ts`), but the strip=true release `.node`
   fails to dlopen on this macOS ("mis-aligned LINKEDIT"; environment issue). Rebuild with
   `CARGO_PROFILE_RELEASE_STRIP=false npx napi build --release --features node-api,security,local-embed-download`
   and run a `LocalEmbedder.load()` / `SemanticIndex` smoke test.
2. `hms-embed fetch` end to end: a bug (ureq body limit equals the pinned size exactly) was fixed in
   `fetch.rs` (`limit(max + 1)`) but not re-run. Run `hms-embed fetch granite-embedding-30m-english --out DIR`
   and check the pack hash equals the registry pin.
3. Engine runs for gte-small / bge-small / all-MiniLM-L6 / e5 (`benchmarks/embedding/run_engine_eval.py
   --skip-lme`) and `bench_torch_cpu.py`; then rerun `assemble_selection.py`.
4. Store gain/cost: done for SciFact and NFCorpus (`embedding_store_v1.json`); LongMemEval-dev was not run there (`run_store_eval.py --datasets lme-dev`). The remainder of this item is obsolete: rerun only for other models.
   ingest docs/s) writes `benchmarks/results/embedding_store_v1.json`; if absent it did not finish. LME-dev
   was not run in the store eval.
5. Serving RSS gate and default-binary size/`nm` diff were not run; only the dependency-tree and diff
   evidence exists. `libm` is used only by one test; the engine now uses its own exp/erf.
6. MSRV 1.89 check of `--features local-embed` and CI `--all-features` (the new feature compiles under
   `--all-features` only together with the existing `local-models`; not run).
7. Model packs must be hosted nowhere: they are built from pinned upstream files
   (`bindings/embed-models/python/build_packs.sh`). Publishing `holographic-vsa-models` and
   `holographic-memory-models` is the coordinator's release step; add `embed` extra is already in
   `pyproject.toml`, and `features` there now includes `local-embed-download`.
8. Throughput: ~280 ms CPU per 330-token passage per core (no AMX/int8 dot); an int8 SIMD dot kernel is the
   next speedup.
