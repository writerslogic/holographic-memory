# Cargo features

The crate builds with `default = []`: the default build is a local, offline library with no
network code, no bindings and no research modules. Every capability below is off until you turn it
on, for example `cargo add writerslogic-warm --features security,server`. Dependencies for each
feature are in [`Cargo.toml`](https://github.com/writerslogic/warm/blob/main/Cargo.toml).

"Stable" features are covered by the [stability policy](../STABILITY.md); "experimental" features
can change in any release.

| Feature | What it turns on | Status | Network code | Docs |
|---|---|---|---|---|
| `security` | AES-256-GCM encrypted stores, Argon2 passphrase keys, Ed25519-signed audit logs | stable | no | [Security](../SECURITY.md), [Encryption format](../file-format/ENCRYPTION.md) |
| `provenance` | COSE_Sign1 signed statements and DID identities; implies `security` | experimental | no | [Privacy](../PRIVACY.md) |
| `provenance-scitt` | Experimental statement submission returning unverified receipt bytes; no VDS inclusion/trust verification; implies `provenance` | experimental | yes (`ureq`), only when you call it | [FAQ](faq.md) |
| `server` | `warm-server`, a self-hosted HTTP service for the Python SDK and multi-process access | stable | listens only on the address you give it | [Server](../SERVER.md) |
| `mcp` | `warm-mcp`, a Model Context Protocol server over stdio or HTTP | stable | listens only when run over HTTP | [MCP](../integrations/mcp/README.md) |
| `local-embed` | Pure-Rust text embeddings from a pinned, sha256-checked model pack on disk | stable | no | [Embeddings](../embeddings/README.md) |
| `local-embed-download` | First-use download of the pinned model pack into the cache; implies `local-embed` | stable | yes, the one download | [Embeddings](../embeddings/README.md) |
| `local-models` | On-device embedding, re-ranking, fact extraction and query rewriting (candle, Qwen3); weights load from local directories only | experimental | no | [Embeddings](../embeddings/README.md) |
| `conversation-memory` | Immutable original conversation turns, role-aware recall and bounded source premises | experimental | no | [Conversation evidence](../memory-api/README.md) |
| `cli` | `clap` argument parsing for the command-line binaries and examples (`warm-admin`, `warm-format`, `public-bench`, the benchmark drivers); implied by `server` and `mcp` | stable | no | [Evaluation](../evaluation.md) |
| `mimalloc` | mimalloc as the global allocator of the `public-bench` binary (the library never sets one) | stable | no | [Public benchmarks](../PUBLIC-BENCHMARKS.md) |
| `node-api` | The Node.js bindings published to npm | stable | no | [Node API](../api/node/index.md) |
| `python` | The `writerslogic-warm-py` Python wheel (vector search and phase vectors) | experimental | no | [Python API](../api/python/vsa.md) |
| `private-search` | Query-private similarity scoring over an index held by someone else (LWE, SimplePIR-style): the index holder ranks an encrypted query and learns nothing about it | experimental, research | no (you transport the messages) | [Private search](../PRIVATE-SEARCH.md) |
| `experimental` | VSA research modules (block codes, Bloom memories, algebraic codebooks, operand retriever, connection and phase graphs) and the experiment binaries | experimental, research | no | [Research](../RESEARCH.md) |

## Which features need a server

None of them. Search, storage, encryption, deletion and provenance all run in your process.
`server` and `mcp` are for when you want to run that process as a service yourself.
`private-search` is for searching an index that someone else holds; it does nothing for a store on
your own machine.
