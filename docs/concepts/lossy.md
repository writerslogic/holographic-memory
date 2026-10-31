# What is lossy

Some parts of WARM keep exactly what you gave them; others keep a lossy sketch. Knowing which is
which decides whether you can rebuild the original and what a leaked store would reveal.

## Kept exactly

| Data | Where | Notes |
|------|-------|-------|
| Document text | chunk text, in the store | Unless you set `storeText: false`. Hits return the passage and its source byte range. |
| Metadata, IDs, source URIs, versions | with the document | Filters match exact JSON values. |
| Embeddings you supply | with the chunk, full precision | Scored by exact cosine, no quantisation. |
| Stemmed term counts | with the chunk | What BM25 scores. Word order inside a chunk is not kept in the term counts. |
| Triples | triple store and composite vectors | The symbolic `(subject, relation, object)` index is exact. |

## Lossy by construction

- **Hypervector sketches.** A text chunk encodes to a sparse vector of character n-grams,
  words and word pairs (`multiscale-words-v1`). The text cannot be rebuilt from the vector, but an attacker who can guess candidate texts can encode them
  and compare; see [privacy limits](privacy-limits.md).
- **Bundles.** A bundle of `N` items keeps each item only as faint overlap. You can ask "is this
  item in the bundle" with a rising error rate as `N` grows ([capacity](capacity.md)), but you
  cannot list the members.
- **Sparse codes of dense vectors.** `memorize_vector` converts a dense embedding to a sparse
  code (`from_dense`, a ternary sketch that keeps the top indices). Nearest-neighbour recall drops
  against the dense original; the committed
  [glove-100-angular run](https://github.com/writerslogic/warm/blob/main/benchmarks/results/public_glove-100-angular.json)
  measured recall@10 of 0.252 for this path, against 0.832 for FAISS HNSW (M=16, efSearch=128)
  and 1.0 for exact search on the same corpus. The document API does not use this path for dense embeddings.
- **Unbinding.** `bind(bind(a, b), a)` returns a noisy copy of `b`, cleaned up by nearest-atom
  search.

## Lossy by storage

- **Compression.** Arena frames are LZ4-compressed; that is lossless.
- **Deletion.** `delete_document` removes the logical record. Old bytes can remain in obsolete
  segments, snapshots and backups until compaction rewrites the store, and compaction is not
  secure erasure.

## Decision guide

If you need the original text back, keep it in the store (the default) or keep it elsewhere and
store only a source URI. If you need to search without retaining text, `storeText: false` keeps
term counts and embeddings, which still reveal a lot; it is not a privacy control.
