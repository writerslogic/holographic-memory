---
description: Fix Python import errors, Node native builds, authentication failures, embedding dimension mismatches and locked stores.
search:
  boost: 2
---

# Troubleshooting setup and retrieval

Start with the smallest failing step: package import, server connection, write, then query.
Keep the original store intact while investigating.

## Python import fails

For `from holographic_sdk import HolographicClient`, install `holographic-sdk` in the same
virtual environment as your application. `holographic-vsa` has a different import name and API.

```sh
uv run --with holographic-sdk python -c "from holographic_sdk import HolographicClient; print(HolographicClient.__name__)"
```

If this works but your application fails, compare the interpreter and environment used to run it.
The SDK is an HTTP client; a successful import does not mean the server is running.

## Node cannot load the native module

Build from the repository root with a Rust toolchain installed:

```sh
npm ci --omit=optional
npm run build
node -e "const { HolographicMemorySystem } = require('./index.js'); console.log(typeof HolographicMemorySystem)"
```

The expected result is `function`. A binary built for another OS or CPU cannot be reused.
The quickstart also uses `node:sqlite`, so run it with Node 24 or newer.

## Connection refused or authentication rejected

1. Confirm `hms-server` is running and that the client URL matches its listening address.
2. Use the same API key in `HMS_API_KEY` on the server and `api_key` in the client.
3. Confirm the tenant ID is the same for ingestion and querying.

Do not paste API keys into issue reports. For authorization rules and tenant isolation, use the
[server reference](../SERVER.md) and [authentication guide](../server/auth.md).

## Embedding dimensions or model identity do not match

The server's `--input-dim` must match every document vector and query vector. Use the same
model and revision for both. Equal vector lengths alone do not make different models compatible.

The store records model identity and dimensions and refuses incompatible reopen settings.
Create a separate store and re-embed the original documents when changing models; do not
rewrite metadata to force an incompatible store to open.

## The store is locked

Stop the other writer cleanly, or connect both clients to one HTTP server. For MCP, use a shared
HTTP process or separate namespaces and store directories. Do not remove lock files while a
writer may still be active. See [MCP sharing](../integrations/mcp/README.md).

## A query returns no useful results

- Check that the document exists in the same tenant or namespace you are querying.
- Remove filters temporarily to distinguish a retrieval problem from a filtering problem.
- Start with distinctive words that occur in the source. A successful lexical query establishes
  that text ingestion and retrieval work before you add an embedding model.
- Compare document and query preprocessing, model revision and vector dimensions.
- Inspect source passages; rank and score alone are insufficient evidence.

The [search walkthrough](../guides/evaluate-search.md) provides a repeatable way to check relevance.

## Report a reproducible problem

Include the package version or commit, OS and CPU, enabled Cargo features, the exact error,
and a minimal example using synthetic text. State whether the problem occurs with a new store.
Exclude credentials, private documents, encryption keys and complete production stores.
