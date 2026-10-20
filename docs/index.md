# Holographic Memory System

Local-first document search and vector-symbolic associative memory, as a Rust library with
Node.js bindings and an optional HTTP server. Everything runs in your process: no service
calls, no telemetry, no model downloads. Apache-2.0.

## What you get

- **Document search**: chunking, stemmed BM25, exact cosine over embeddings you supply, metadata
  filters, byte-offset hits, atomic replace and delete, crash-safe persistent storage.
- **Vector-symbolic memory**: sparse binary hypervectors with bind, bundle and permute, a
  relational triple store with multi-hop queries, Hopfield cleanup, and a resonator network.
- **Optional hardening** behind Cargo features: AES-256-GCM encrypted stores, signed audit logs,
  credential-gated agent access, COSE/SCITT provenance.

## Start here

<div class="grid cards" markdown>

- **[Quickstart](quickstart/index.md)**: ingest, search, compare with SQLite FTS5, and encrypt, in Rust, Python and Node.
- **[Offline agent demo](quickstart/offline-agent.md)**: a personal agent memory with no embeddings, no network, and a deterministic transcript.
- **[Concepts](concepts/index.md)**: what "holographic" means in this code, what is lossy, the tested capacity range, and the privacy limits.
- **[Security and privacy](PRIVACY.md)**: exactly what is and is not protected.

</div>

## Honest scope

The document pipeline is conventional retrieval (stemmed BM25 fused with exact cosine). The
vector-symbolic layer powers the meaning memory, composition and the experimental vector index.
Where HMS loses to another system on a measured benchmark, the [benchmark pages](PUBLIC-BENCHMARKS.md)
say so. Every number on this site traces to a committed file under
[`benchmarks/results/`](https://github.com/writerslogic/holographic-memory/tree/main/benchmarks/results).

## Versions

The selector in the header switches between `latest` (the newest tagged release that carries
these docs), `dev` (the `main` branch), and each tagged release.
