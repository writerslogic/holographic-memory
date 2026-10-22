# Cargo features

The crate builds with `default = []`: the default build is a local, offline library with no
network code, no bindings and no research modules. Every capability below is off until you turn it
on, for example `cargo add holographic-memory --features security,server`. Dependencies for each
feature are in [`Cargo.toml`](https://github.com/writerslogic/holographic-memory/blob/main/Cargo.toml).

"Stable" features are covered by the [stability policy](../STABILITY.md); "experimental" features
can change in any release.

| Feature | What it turns on | Status | Network code | Docs |
|---|---|---|---|---|
| `security` | AES-256-GCM encrypted stores, Argon2 passphrase keys, Ed25519-signed audit logs | stable | no | [Security](../SECURITY.md), [Encryption format](../file-format/ENCRYPTION.md) |
| `provenance` | COSE_Sign1 signed statements and DID identities; implies `security` | experimental | no | [Privacy](../PRIVACY.md) |
| `provenance-scitt` | Submits signed statements to a SCITT transparency service; implies `provenance` | experimental | yes (`ureq`), only when you call it | [FAQ](faq.md) |
| `server` | `hms-server`, a self-hosted HTTP service for the Python SDK and multi-process access | stable | listens only on the address you give it | [Server](../SERVER.md) |
| `mcp` | `hms-mcp`, a Model Context Protocol server over stdio or HTTP | stable | listens only when run over HTTP | [MCP](../integrations/mcp/README.md) |
| `local-embed` | Pure-Rust text embeddings from a pinned, sha256-checked model pack on disk | stable | no | [Embeddings](../embeddings/README.md) |
| `local-embed-download` | First-use download of the pinned model pack into the cache; implies `local-embed` | stable | yes, the one download | [Embeddings](../embeddings/README.md) |
| `local-models` | On-device embedding, re-ranking, fact extraction and query rewriting (candle, Qwen3); weights load from local directories only | experimental | no | [Embeddings](../embeddings/README.md) |
| `node-api` | The Node.js bindings published to npm | stable | no | [Node API](../api/node/index.md) |
| `python` | The `holographic-vsa` Python wheel (quantized-phase substrate) | experimental | no | [Python API](../api/python/vsa.md) |
| `private-search` | Query-private similarity scoring over an index held by someone else (LWE, SimplePIR-style): the index holder ranks an encrypted query and learns nothing about it | experimental, research | no (you transport the messages) | [Private search](../PRIVATE-SEARCH.md) |
| `experimental` | VSA research modules (block codes, Bloom and CLS memories, resonator, ternary and sparse-autoencoder encoders) and the experiment binaries | experimental, research | no | [Research](../RESEARCH.md) |

## Which features need a server

None of them. Search, storage, encryption, deletion and provenance all run in your process.
`server` and `mcp` are for when you want to run that process as a service yourself.
`private-search` is for searching an index that someone else holds; it does nothing for a store on
your own machine.
