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
