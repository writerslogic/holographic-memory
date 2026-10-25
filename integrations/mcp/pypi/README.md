<a href="https://holographic-memory.com/"><img src="https://raw.githubusercontent.com/writerslogic/holographic-memory/main/assets/crystal-halo/crystal-detailed-static.png" alt="Holographic Memory logo" width="96" align="left"></a>

# Holographic Memory MCP

<!-- mcp-name: io.github.writerslogic/holographic-memory-mcp -->

Persistent local memory for AI agents through the Model Context Protocol. The server stores memories on your machine and supports stdio and streamable HTTP transports.

[Website](https://holographic-memory.com/) · [Documentation](https://docs.holographic-memory.com/latest/integrations/mcp/) · [Source](https://github.com/writerslogic/holographic-memory) · [License](https://github.com/writerslogic/holographic-memory/blob/main/LICENSE)

<br clear="left">

## Run

Python 3.9 or newer is required for this launcher. Run it directly with `uvx`:

```sh
uvx holographic-memory-mcp
```

The default stdio transport works with MCP clients that launch a local command. To share one server between clients, use streamable HTTP:

```sh
uvx holographic-memory-mcp --transport http
```

The HTTP endpoint defaults to `http://127.0.0.1:8765/mcp`. See the [MCP guide](https://docs.holographic-memory.com/latest/integrations/mcp/) for client configuration, storage locations, encryption, authentication, and available flags.

## What the server provides

| Tool | Purpose |
| --- | --- |
| `remember` | Save or update a memory. |
| `recall` | Find memories relevant to a query. |
| `search` | Search with explicit filters and ranking options. |
| `get`, `list` | Read one memory or page through a store. |
| `forget` | Delete selected memories. |
| `stats`, `export` | Inspect and export a store. |

The server also exposes memory resources and prompts. Memories persist across client sessions. By default, recall uses lexical search; semantic recall requires a build configured with a local embedding model.

## Launcher and binary

The package installs a small Python launcher. On first run, it downloads the matching prebuilt `hms-mcp` binary from the [GitHub release](https://github.com/writerslogic/holographic-memory/releases) and verifies its SHA-256 against a checksum table shipped in the package. The server itself runs locally; the launcher needs network access for that first binary download.

Set `HMS_MCP_BINARY` to run your own `hms-mcp` build. `HMS_MCP_CACHE_DIR` changes the download cache. The [MCP guide](https://docs.holographic-memory.com/latest/integrations/mcp/) covers the server's `HMS_MCP_*` configuration variables.

The same launcher is [available on npm](https://www.npmjs.com/package/holographic-memory-mcp).
