# holographic-memory-mcp

MCP server for Holographic Memory: private, on-device persistent memory for AI agents. This package downloads the prebuilt `hms-mcp` binary for your platform (checked against a SHA-256 table shipped in the package) and runs it.

```sh
uvx holographic-memory-mcp                       # stdio, for MCP clients
uvx holographic-memory-mcp --transport http      # streamable HTTP on 127.0.0.1:8765/mcp
```

Tools: `remember`, `recall`, `search`, `get`, `list`, `forget`, `stats`, `export`. Everything is configured through flags or `HMS_MCP_*` environment variables; see `docs/integrations/mcp/README.md` in the repository.

Environment: `HMS_MCP_BINARY` runs a binary you built yourself, `HMS_MCP_CACHE_DIR` moves the download cache, `HMS_MCP_RELEASE_BASE` and `HMS_MCP_CHECKSUMS` point the downloader at a mirror and its checksum table.

Licensed Apache-2.0.
