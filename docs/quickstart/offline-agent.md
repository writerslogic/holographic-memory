# Offline agent memory demo

A personal assistant that remembers things about one person, using no embedding model, no
network and no API key. One command from a clean clone:

```sh
cargo run --example demo-offline-agent
```

It is deterministic. Nothing depends on the clock, the machine or a random seed, so the
transcript below is exactly what you will see, and CI diffs every run against it.

## What it shows

1. **Learn.** Eight notes go in as documents, in one atomic batch, each with a `kind`
   (`fact` or `episode`) and a day in its metadata. Three relations go into the meaning memory.
2. **Recall by content.** Plain-language questions return the right note first. Retrieval is
   stemmed BM25, so questions that share words with a note find it; a question that shares none
   does not. That is the honest limit of embedding-free search, and the reason the engine also
   accepts embeddings you supply.
3. **Recall by metadata.** The same query with `kind = episode` excludes a fact that would
   otherwise match. The filter runs before ranking.
4. **A fact changes.** Writing the same ID again replaces the old version atomically, and the
   old wording is gone from search.
5. **Forgetting.** Deleting a note that held a pasted secret removes it from search, and a
   compaction rewrites the store. Compaction is not secure erasure: old bytes can survive in
   snapshots and backups, as the [privacy limits](../concepts/privacy-limits.md) say.
6. **Relations.** "Where is Mira's employer based" is a two-hop query over stored triples
   (`works_at`, then `based_in`) answered by the vector-symbolic meaning memory, not by text search.
7. **Restart.** The store is dropped and reopened from disk; every answer still holds.

## Transcript

```text
--8<-- "examples/demo-offline-agent/expected-output.txt"
```

## Source

```rust
--8<-- "examples/demo-offline-agent/main.rs"
```

## Adapting it

- Replace the temporary directory with a path you keep. The store is a directory and takes an
  exclusive lock while open.
- To rank by meaning rather than shared words, pass `embeddings` on each document and an
  `embedding` in `SearchOptions`. WARM then fuses exact cosine with BM25.
- The retrieval here is lexical on purpose, so it is easy to reason about and test. It is not a
  claim that BM25 alone is enough for a production assistant; see the
  [benchmarks](../PUBLIC-BENCHMARKS.md) for how the hybrid does on public data.
