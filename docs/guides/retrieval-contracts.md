# Retrieval contracts

`search_vectors` in exact mode selects and ranks with the same sequential `f64` cosine
used by document search. Stored embeddings remain normalized `f32` values. Graph mode
selects approximate candidates and then computes their canonical scores. Equal scores
use document ID and chunk index order.

Sparse `query` results report Jaccard similarity for every route. Hopfield attention is
available through `query_hopfield`; its weights are a separate retrieval contract. Index
training can affect approximate candidate membership, but it does not change the meaning
of `similarity`.

After `train_document_index`, broad metadata-filtered hybrid queries reuse the exact
bounding-box tree and retain at most `candidateLimit` lexical owners before semantic
fusion. The tree checks document filters before scoring a row. Small filtered sets use a
direct scan. Both paths retain the filtered corpus's BM25 statistics and canonical scores.
This changes the execution plan, not the search or filtering contract.

Filtered lexical queries and trained hybrid queries use filtered postings when their work
is smaller than scanning every eligible chunk for every term. The captured eligibility is
also used by dense traversal. Posting acceleration is capped at 524,288 store slots; the
dense eligibility set is separately capped at that many matching documents. Beyond these
limits the existing scan or predicate path applies. BM25 normalization belongs to the
captured filter and generation, including legacy zero-frequency terms.

The trained exact tree remains usable after mutations through a bounded changed-document
delta. Changed base documents are excluded before top-k selection, and current changed
chunks participate in canonical scoring and fusion. The delta holds at most 128 document
IDs, 1,024 live chunks and 64 KiB of ID bytes. Exceeding a bound drops the acceleration;
queries retain their exact fallback and maintenance can rebuild. Replacing the same ID
updates its delta entry. Explicit training publishes only at its captured generation and
resets the delta. The `trained` status still means the base itself is current.

Dense training shares the existing vector-build admission gate and reports a retryable
maintenance error when occupied. Queries add no background workers or queues. Server
admission retains its bounded queue and busy response; cancellation of a waiting request
releases its queue entry, while a running blocking query owns its permit until completion.
These bounds describe additional search state, not a process RSS guarantee.

Phrase queries use persisted positions where available. For a candidate chunk without
positions, the engine reconstructs them from retained text using that chunk's analyzer.
An unrelated chunk without positions does not disable phrase search. A candidate whose
phrase cannot be resolved because both positions and text are absent returns an error;
it is never silently excluded. Persisting positions avoids reconstruction work.

Whole-document lexical ingest still requires positions, but it can also accept one dense
embedding in the configured embedding space. This supports phrase matching across chunk
boundaries and vector/hybrid retrieval of the same article. It does not create a sparse
text hypervector in lexical ingest mode.

The standalone `FilteredIndex` accepts dimensions from 1 through 66,051 so every squared
u8 Euclidean distance fits its `u32` result. It accepts at most 1,048,576 labels in its
dense label tables, validates training and CSR shapes, and bounds result capacity by the
live candidate count. Use a compact label mapping for sparse external label identifiers.

## Reasoning budgets

Use `multi_hop_with_limits` and `synthesize_concepts_with_limits` with
`holographic_memory::core::query_limits::QueryLimits` to handle resource limits explicitly.
Their defaults are 10,000,000 work units, 16,384 frontier entries and 64 MiB of accounted
bytes. Work includes scanned triples, candidate visits, sorting and comparisons. Byte
accounting conservatively covers retained inputs, intermediate frontiers and working
buffers; it is not a process RSS measurement.

Budget exhaustion returns an error identifying the resource and limit. Results are
complete or absent because of the error; there are no silently truncated answers.
Multi-hop retrieval preserves distinct paths even when they reach the same endpoint.
Concept synthesis generates one candidate frontier per seed instead of retaining every
pair of neighbors. Common dimensions can still require quadratic work, so the work
limit remains necessary.

The legacy Rust `multi_hop` and `synthesize_concepts` signatures retain their `Vec` return
types and apply the default limits. They panic on budget exhaustion. Call the checked
companions in applications that must report or recover from an exhausted budget.


## Durable meaning governance

`HmsCore::govern_memory` returns `Result<GovernanceReport>` and uses the default
`QueryLimits`; `govern_memory_with_limits` accepts explicit limits. It captures a coherent
revision and bounded live snapshots, plans on isolated memories outside the mutation gate,
and publishes one journal transaction only if the revision is still current.
`GovernanceConflict` means another writer invalidated the plan; no governance change was
committed, and callers may retry. Work, frontier, byte, operation-count and serialized-size
exhaustion also fail before publication. The atomic transaction is capped at 4,096 operations
and 32 MiB; byte accounting is a conservative allocation budget, not an RSS measurement.

Live triple owners survive consolidation even when their vectors are identical. Scoped
atom/composite deletions preserve unrelated raw vectors with the same ID. Refinement
recomposes affected owned composites and their matching raw-vector mirrors; ambiguous or
missing ownership fails explicitly. Reopen and compaction retain these changes. The
standalone `MemoryGovernor` primitive operates only on the supplied in-memory stores;
applications persisting an `HmsCore` use its durable governance methods.

New stores use schema 5. Meaning-enabled stores with schemas 2–4 keep their writer format
and must first be explicitly copied with `hms-format upgrade <source> <destination>` before
durable governance. The source remains unchanged, and current arenas/checkpoints are copied
without re-encoding their existing journal frames.
