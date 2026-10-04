# Application review implementation

These changes implement the ten review recommendations in the working tree. They are not a
published release. Storage format 2 requires re-encoding legacy inputs; follow the
[migration guide](production-readiness.md) before using existing stores.

| Recommendation | Implementation | Evidence |
|---|---|---|
| 1. Crash-safe compaction | Verified arena generations, flushed atomic CURRENT publication, shared transaction gate | Child-process interruption tests at three publication phases; concurrent writers and compaction |
| 2. Enforce security guarantees | Unsupported encryption/signing fails; encrypted reopen and wrong-key checks; runtime capability status; contribution-clipped full-domain DP bundling | Default-feature rejection and native encryption tests; corrected privacy/security analysis |
| 3. Preserve embedding sign | Signed projection version 2, finite/nonzero input validation, distinct opposite vectors | Opposite-vector regression and measured dense-neighbor fidelity |
| 4. Make structural ingestion functional | Triplets populate atoms, role-bound composites, symbolic triples, and retrieval vectors together | Actual capital-of and two-hop answers through Rust and Node APIs |
| 5. Keep restart behavior consistent | Typed transactions for relation removals/types/rules; atomic batches; derived-fact ownership; ANN cache checkpoints/checksums | Update/delete/restart/compaction tests; stale and damaged cache recovery |
| 6. Version the embedding space | Immutable store schema for encoders, dimensions, model/revision/normalization/metric, encryption and meaning mode; atomic JSONL re-encoding tool | Incompatible reopen rejection; failed migration leaves source intact and destination absent |
| 7. Add real semantic retrieval | Explicit local ONNX embedding and cross-encoder factories; BM25/cosine rank fusion; optional reranking | Real MiniLM evaluation with pinned artifact revisions and distinct labeled queries |
| 8. Add a document lifecycle | Chunking, source offsets, metadata filters, versions, passages, replacement, and deletion; bounded UTF-8 file ingestion | Unicode/source/filter/version tests and native file-ingestion test |
| 9. Control write costs | Stable posting slots, incremental update/delete, explicit snapshot-based index maintenance, background Rust maintenance, bounded batches and JS queues | Queue backpressure/failure recovery; concurrent p95/p99 reports at 1,200 and 10,000 vectors |
| 10. Test the product surface | Behavioral regressions, actual native CJS/ESM tests, strict consumer TypeScript checks, packed-install test, deterministic relevance floor, cross-platform CI | Local default/all-feature Rust, Clippy, MSRV, benchmark compilation, Node, type, and package checks |

The [evaluation report](evaluation.md) includes reproducible inputs, commands, raw metrics, and
measurement limits. On ten curated paraphrase queries, first-result accuracy was 3/10 lexical,
5/10 hybrid, and 10/10 hybrid with reranking. Sparse projection retained 60% of dense top-five
neighbors; document semantic search uses exact dense cosine instead. These figures are small
regression results, not general quality or capacity claims.

Compaction still blocks main queries while snapshotting/publishing. Exact dense document
search scans eligible chunks and retains embeddings in memory. Training can require retry if
writes change its snapshot. DP is a per-bundle mechanism, not whole-application privacy;
source passages are retained by default. Audit/provenance sidecars are outside the main arena
transaction. Cross-platform CI is configured; local verification ran on macOS arm64.
