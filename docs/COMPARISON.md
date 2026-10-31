# WARM vs. Alternatives

**Hybrid search, graph indexing, and vector-symbolic memory.**

WARM (WARM) combines several retrieval and memory paths in one local Rust engine. The paths have different roles; the vector-symbolic layer does not rank ordinary document-search results.

| Need | WARM path | Useful comparison |
|------|----------|-------------------|
| Search exact words and phrases | Stemmed BM25 with Boolean and phrase queries | Full-text search engines |
| Search by meaning | Exact cosine over embeddings you supply, fused with lexical results for hybrid search | Embedding-based retrieval systems |
| Search a large vector collection | Graph index with quantized traversal and exact reranking | Approximate nearest-neighbor libraries |
| Store and query structured facts | Vector-symbolic bind, bundle and associative recall, plus a triple store | Symbolic and graph memory systems |

The document search path needs supplied embeddings for semantic matching. Its index can contain source text; enable the encrypted store when confidentiality is required. Hypervector sketches and term counts are not encryption.

The vector-symbolic layer supports composition and associative recall for facts. It is not a substitute for a strong language embedding model when the task is nuanced semantic search over prose. The older sparse-vector path is experimental and trails established ANN libraries; use the graph index for vector search.

For measured wins and losses, see [public benchmarks](PUBLIC-BENCHMARKS.md). For the boundary between document retrieval and holographic memory, see [what “holographic” means](concepts/holographic.md) and the [architecture map](ARCHITECTURE.md).
