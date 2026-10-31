# Concepts

WARM has two layers that share one store and one persistence format.

| Layer | What it does | Built on |
|-------|--------------|----------|
| **Document search** | Chunk text, rank passages for a query, return the source passage and its byte range | Stemmed BM25 and exact cosine over embeddings you supply |
| **Vector-symbolic memory** | Store facts as bound hypervectors, answer relational and multi-hop queries, clean up noisy results | Sparse binary hypervectors, bind/bundle/permute, Hopfield cleanup |

Read these in order:

1. [What "holographic" means here](holographic.md): the vectors, the three operations, and which parts of the engine use them.
2. [What is lossy](lossy.md): where information is discarded, and what survives.
3. [Capacity limits](capacity.md): how many items fit in one superposition, with the tested range only.
4. [Privacy limits](privacy-limits.md): what the stored representation reveals, and what encryption does and does not cover.

For the module map see [Architecture](../ARCHITECTURE.md); for storage, locking and migration see
[Production readiness](../production-readiness.md).
