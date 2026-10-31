---
description: Choose between the Rust library, Node module, Python HTTP client and MCP server for editor memory.
search:
  boost: 2
---

# Choose an integration

Start with where your application runs and who should own the store.

| Your task | Start here | Where the engine runs |
|---|---|---|
| Give an editor or coding agent persistent memory | [MCP and editor setup](../integrations/editors/README.md) | A local `warm-mcp` process, or a shared MCP HTTP server |
| Embed document search in a Rust application | [Rust quickstart](../quickstart/index.md) | Inside your application |
| Add search to a Node application | [Node installation](install.md) | A native Rust module inside Node |
| Access a store from Python or several processes | [Python SDK and HTTP server](../SERVER.md) | A server you run; Python is the client |
| Search vectors or experiment with phase hypervectors in Python | [Python runtime](../api/python/vsa.md) | The `warm-py` extension |

## Which Python package?

`warm-sdk` imports as `writerslogic_warm_sdk`. It is an HTTP client for `warm-server`.
The server must be running, and client and server must use the same API key, tenant and
embedding dimensions. Installing the SDK does not start a server or download an embedding model.

`warm-py` imports as `writerslogic_warm`. It exposes native vector search and
phase-vector algebra. Choose it when you need those APIs in-process.

## Which retrieval path?

- **Document search:** ingest text, then query with words. BM25 works without a model.
  Add compatible embeddings for semantic retrieval and hybrid search.
- **Fact memory:** use the [memory API](../memory-api/README.md) to track assertions,
  replacements and history.
- **Editor memory:** MCP exposes `remember`, `recall`, `search` and other tools to your host.
  Read the [MCP tool reference](../integrations/mcp/README.md) before enabling write operations.

A similarity score is a ranking signal, not a probability that a statement is true.
Keep source text and inspect returned evidence before using it in an answer.

## Decide who opens the store

An embedded application owns its local store. Multiple clients should connect to one server
instead of opening the same writable directory in independent processes. MCP offers stdio
for a single client and HTTP for shared access. Its namespace settings determine which
memories clients share.

## Decide what must stay private

The engine's `security` feature supports encrypted stores. The Python HTTP server has different
storage guarantees; see [server limitations](../SERVER.md).
Embeddings, sketches and vector masking are not encryption. Read the
[privacy limits](../concepts/privacy-limits.md) before choosing a deployment.

Continue with [installation](install.md), then run the [complete quickstart](../quickstart/index.md).
