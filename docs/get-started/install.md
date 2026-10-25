---
description: Install Holographic Memory for Rust, Python or Node, with the version requirements and a one-line check that it works.
next_steps:
  - title: Quickstart
    text: Ingest, search, compare with SQLite and encrypt, in five minutes.
    url: quickstart/index.md
  - title: Concepts
    text: What "holographic" means in this code, and what is lossy.
    url: concepts/index.md
  - title: Server
    text: Run hms-server for the Python client.
    url: SERVER.md
---

# Install

HMS is a Rust engine. Use it directly, build the Node module, connect an editor through MCP,
or call a self-hosted server from Python. [Choose an integration](choose.md) before installing.
Package installation needs network access; the engine can run offline after installation.

## Requirements

| You need | For |
|---|---|
| Rust 1.89 or newer (the MSRV) | the Rust crate, the HTTP server and the Node build |
| Node 22 or newer | the Node bindings; Node 24 for the quickstart, which uses `node:sqlite` |
| Python 3.10 or newer | the Python SDK (`holographic-memory-py` needs 3.9) |

For Python, create and activate a virtual environment first: `uv venv`, then
`source .venv/bin/activate` (PowerShell: `.venv\Scripts\Activate.ps1`).

## Install for your runtime

=== "Rust"

    ```sh
    cargo add holographic-memory --git https://github.com/writerslogic/holographic-memory
    ```

    `default = []`: nothing is on unless you ask. Turn on what you need, for example
    `cargo add holographic-memory --git https://github.com/writerslogic/holographic-memory --features security` for AES-256-GCM stores and signed audit logs.
    Every feature, what it turns on and whether it is experimental: [Cargo features](../reference/features.md).

=== "Python"

    The Python SDK is an HTTP client for `hms-server`, which holds the engine and stores the
    vectors you supply:

    ```sh
    uv pip install holographic-memory-sdk
    ```

    Build and start the server from a checkout, then point the client at it:

    ```sh
    git clone https://github.com/writerslogic/holographic-memory
    cd holographic-memory
    cargo build --release --features server --bin hms-server
    HMS_API_KEY=change-me target/release/hms-server --input-dim 384
    ```

    The `holographic-memory-py` wheel provides native vector search and phase vectors,
    imported as `holographic_memory`. It does not include document search.

    ```sh
    uv pip install holographic-memory-py
    ```

=== "Node"

    The native module is built from source, which needs a Rust toolchain:

    ```sh
    git clone https://github.com/writerslogic/holographic-memory
    cd holographic-memory
    npm ci --omit=optional
    npm run build
    ```

    The npm package is deprecated until the next release; use the source build until then.

## Check that it works

=== "Rust"

    ```sh
    cargo run --release --manifest-path examples/quickstart/rust/Cargo.toml
    ```

=== "Python"

    ```sh
    uv run --with holographic-memory-sdk python -c "from holographic_sdk import HolographicClient; print(HolographicClient.__name__)"
    ```

=== "Node"

    ```sh
    node -e "const { HolographicMemorySystem } = require('./index.js'); console.log(typeof HolographicMemorySystem)"
    ```

The [quickstart](../quickstart/index.md) runs a full ingest and search against the same checkout.

The Python check verifies the SDK import; it does not contact the server. For a complete
round trip, run the Python tab in the [quickstart](../quickstart/index.md).

## Connect an editor

Build `hms-mcp` from this checkout, then configure your editor to launch the absolute binary path:

```sh
cargo build --release --features mcp,security --bin hms-mcp
```

Follow the [editor setup guide](../integrations/editors/README.md) for your host. The
[MCP reference](../integrations/mcp/README.md) covers tools, storage and shared HTTP access.

## If setup fails

See [troubleshooting](troubleshooting.md) for import errors, missing native binaries,
server authentication, vector dimensions and locked stores.
