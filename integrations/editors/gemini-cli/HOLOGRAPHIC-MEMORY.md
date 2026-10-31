# WARM

The `writerslogic-warm` MCP server keeps durable, private memory on this machine. Its tools appear as `mcp_writerslogic-warm_<tool>`: `remember`, `recall`, `search`, `get`, `list`, `forget`, `stats`, `export`.

- Before starting a task, call `recall` with a short description of it and use what comes back.
- Save durable facts, decisions and conventions with `remember`: one topic per memory and a stable `id` such as `decisions/<topic>`, so a later write with the same id replaces it.
- Do not store secrets or anything the code or git history already shows.
- Use `list` and `get` to review memories and `forget` to remove stale ones.
