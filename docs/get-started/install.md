---
description: Install WARM for Rust, Python or Node, with the version requirements and a one-line check that it works.
next_steps:
  - title: Quickstart
    text: Ingest, search, compare with SQLite and encrypt, in five minutes.
    url: quickstart/index.md
  - title: Concepts
    text: What "holographic" means in this code, and what is lossy.
    url: concepts/index.md
  - title: Server
    text: Run warm-server for the Python client.
    url: SERVER.md
---

# Install

WARM is a Rust engine. Use it directly, install the Node bindings, connect an editor through MCP,
or call a self-hosted server from Python. [Choose an integration](choose.md) before installing.
Package installation needs network access; the engine can run offline after installation.

## Requirements

| You need | For |
|---|---|
| Rust 1.89 or newer (the MSRV) | the Rust crate, the HTTP server and the Node build |
| Node 22 or newer | the Node bindings; Node 24 for the quickstart, which uses `node:sqlite` |
| Python 3.10 or newer | the Python SDK (`warm-python` needs 3.9) |

For Python, create and activate a virtual environment first: `uv venv`, then
`source .venv/bin/activate` (PowerShell: `.venv\Scripts\Activate.ps1`).

## Install for your runtime

=== "Rust"

    ```sh
    cargo add warm-rust
    ```

    `default = []`: nothing is on unless you ask. Turn on what you need, for example
    `cargo add warm-rust --features security` for AES-256-GCM stores and signed audit logs.
    Every feature, what it turns on and whether it is experimental: [Cargo features](../reference/features.md).

=== "Python"

    The Python SDK is an HTTP client for `warm-server`, which holds the engine and stores the
    vectors you supply:

    ```sh
    uv pip install warm-sdk
    ```

    Build and start the server from a checkout, then point the client at it:

    ```sh
    git clone https://github.com/writerslogic/warm
    cd warm
    cargo build --release --features server --bin warm-server
    WARM_API_KEY=change-me target/release/warm-server --input-dim 384
    ```

    The `warm-python` wheel provides native vector search and phase vectors,
    imported as `writerslogic_warm`. It does not include document search.

    ```sh
    uv pip install warm-python
    ```

=== "Node"

    Install the Node bindings. npm selects the native binary for your platform:

    ```sh
    npm install writerslogic-warm
    ```

    To build from a checkout instead, run `npm ci --omit=optional && npm run build` with a Rust
    toolchain. The separate `writerslogic-warm-mcp` package launches the MCP server.

## Check that it works

=== "Rust"

    ```sh
    cargo run --release --manifest-path examples/quickstart/rust/Cargo.toml
    ```

=== "Python"

    ```sh
    uv run --with warm-sdk python -c "from writerslogic_warm_sdk import WarmClient; print(WarmClient.__name__)"
    ```

=== "Node"

    ```sh
    node -e "const { WarmMemorySystem } = require('writerslogic-warm'); console.log(typeof WarmMemorySystem)"
    ```

The [quickstart](../quickstart/index.md) runs a full ingest and search against the same checkout.

The Python check verifies the SDK import; it does not contact the server. For a complete
round trip, run the Python tab in the [quickstart](../quickstart/index.md).

## Connect an editor

Run the separate MCP launcher from npm:

```sh
npx -y writerslogic-warm-mcp
```

To build `warm-mcp` from this checkout instead and configure your editor to launch its absolute path:

```sh
cargo build --release --features mcp,security --bin warm-mcp
```

Follow the [editor setup guide](../integrations/editors/README.md) for your host. The
[MCP reference](../integrations/mcp/README.md) covers tools, storage and shared HTTP access.

## If setup fails

See [troubleshooting](troubleshooting.md) for import errors, missing native binaries,
server authentication, vector dimensions and locked stores.
