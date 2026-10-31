<a href="https://writerslogic.com/warm/"><img src="https://raw.githubusercontent.com/writerslogic/warm/main/assets/crystal-halo/crystal-detailed-static.png" alt="WARM logo" width="96" align="left"></a>

# WARM MCP

<!-- mcp-name: io.github.writerslogic/warm-mcp -->

Persistent local memory for AI agents through the Model Context Protocol. The server stores memories on your machine and supports stdio and streamable HTTP transports.

[Website](https://writerslogic.com/warm/) · [Documentation](https://writerslogic.com/warm/docs/latest/integrations/mcp/) · [Source](https://github.com/writerslogic/warm) · [License](https://github.com/writerslogic/warm/blob/main/LICENSE)

<br clear="left">

## Run

Python 3.9 or newer is required for this launcher. Run it directly with `uvx`:

```sh
uvx warm-mcp
```

The default stdio transport works with MCP clients that launch a local command. To share one server between clients, use streamable HTTP:

```sh
uvx warm-mcp --transport http
```

The HTTP endpoint defaults to `http://127.0.0.1:8765/mcp`. See the [MCP guide](https://writerslogic.com/warm/docs/latest/integrations/mcp/) for client configuration, storage locations, encryption, authentication, and available flags.

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

The package installs a small Python launcher. On first run, it downloads the matching prebuilt `warm-mcp` binary from the [GitHub release](https://github.com/writerslogic/warm/releases) and verifies its SHA-256 against a checksum table shipped in the package. The server itself runs locally; the launcher needs network access for that first binary download.

Set `WARM_MCP_BINARY` to run your own `warm-mcp` build. `WARM_MCP_CACHE_DIR` changes the download cache. The [MCP guide](https://writerslogic.com/warm/docs/latest/integrations/mcp/) covers the server's `WARM_MCP_*` configuration variables.

The same launcher is [available on npm](https://www.npmjs.com/package/writerslogic-warm-mcp).
