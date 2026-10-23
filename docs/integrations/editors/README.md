# Holographic Memory in editors and agents

Every host below talks to `hms-mcp` (see `docs/integrations/mcp/README.md`). Ready-made configs live in `integrations/editors/<host>/`. The commands assume the launcher packages are published (`npx -y holographic-memory-mcp`, or `uvx holographic-memory-mcp`); until then substitute the path of a binary you built (`cargo build --release --features mcp,security --bin hms-mcp`) for `npx -y holographic-memory-mcp`.

Two ways to run it, for any host:

- **One process per client (stdio, the default).** The host starts the server. An engine store has one writer, so two hosts on the same store collide (`store_locked`); give each host its own `--data-dir`, or use the shared form.
- **One shared server (streamable HTTP).** Run `HMS_MCP_TOKEN=... hms-mcp --transport http` once and point every host at `http://127.0.0.1:8765/mcp` with `Authorization: Bearer $HMS_MCP_TOKEN`. The remote form of each host is under its heading.

On Windows, hosts that cannot run `npx` directly need `"command": "cmd", "args": ["/c", "npx", "-y", "holographic-memory-mcp"]`.

## Status of each integration

| Host | One-line install | Verified how |
|---|---|---|
| Claude Code | `claude plugin marketplace add ./integrations/editors/claude-code && claude plugin install holographic-memory@holographic-memory` (or `claude mcp add holographic-memory -- npx -y holographic-memory-mcp`) | `claude plugin validate` passes for the plugin and the marketplace; a headless `claude -p --plugin-dir` session loaded the plugin, ran the SessionStart hook (the model quoted its context), called `remember`, `recall` and `stats` through the plugin's MCP server, and resolved the `project` namespace from `${CLAUDE_PROJECT_DIR}` and the `userConfig` default. The session used the locally built binary in place of the unpublished npx package, and `plugin marketplace add` was not run because it edits the user's settings. |
| Codex CLI | `codex mcp add holographic-memory -- npx -y holographic-memory-mcp` | Verified with 0.163.0. In an isolated `CODEX_HOME`, `codex mcp add`, `list` and `get` accept the entry, and `codex mcp list` parses `integrations/editors/codex/config.toml`. No model session, so no tool call. |
| Gemini CLI | `gemini mcp add -s user holographic-memory npx -y holographic-memory-mcp`, or `gemini extensions install <repo url>` | Verified with 0.64.0 in an isolated `HOME`. `gemini mcp list` connected to the server (`Connected`); `gemini extensions link` loaded the extension manifest, its context file and its MCP server. No model session. |
| VS Code (Copilot) | `code --add-mcp '{"name":"holographic-memory","type":"stdio","command":"npx","args":["-y","holographic-memory-mcp"]}'` | Verified with 1.141.0. `code --add-mcp` in an isolated profile wrote a user `mcp.json` of exactly the shape in `integrations/editors/vscode/mcp.json`. Copilot chat not exercised. |
| Cursor | `mkdir -p .cursor && cp integrations/editors/cursor/mcp.json .cursor/mcp.json` | Config-validated only. |
| Windsurf | merge `integrations/editors/windsurf/mcp_config.json` into `~/.codeium/windsurf/mcp_config.json` | Config-validated only. |
| Zed | merge `integrations/editors/zed/settings.json` into `settings.json` | Config-validated only. |
| Cline | merge `integrations/editors/cline/cline_mcp_settings.json` into `cline_mcp_settings.json` | Config-validated only. |
| Continue | `mkdir -p .continue/mcpServers && cp integrations/editors/continue/holographic-memory.yaml .continue/mcpServers/` | Config-validated only. |
| Claude Desktop | merge `integrations/editors/claude-desktop/claude_desktop_config.json` into `claude_desktop_config.json` | Config-validated only. |

"Config-validated" means `integrations/editors/validate.py` checks each file's structure (required keys, field types, no unknown keys) against that host's current documentation, which was fetched while writing it; the applications were not driven. Run it with `uv run --with pyyaml python integrations/editors/validate.py`.

## Claude Code

The plugin (`integrations/editors/claude-code/plugin`) contains:

- `.claude-plugin/plugin.json`: manifest with one option, `namespace_strategy` (`project`, the default, or `global`).
- `.mcp.json`: the server, started with `HMS_MCP_NAMESPACE_STRATEGY=${user_config.namespace_strategy}` and `HMS_MCP_PROJECT=${CLAUDE_PROJECT_DIR}`. Tools appear as `mcp__plugin_holographic-memory_holographic-memory__<tool>`.
- `hooks/hooks.json` and `scripts/session-start.mjs`: a `SessionStart` hook (sources `startup`, `resume`, `clear`, `compact`) that adds the memory protocol to the session context: recall before work, save durable decisions with stable ids, recall again after compaction. The hook never opens the store, because the MCP server holds its lock.
- `skills/holographic-memory/SKILL.md`: what to remember, how to key and tag it, compare-and-set edits, review and forgetting.

Memories live in the default data directory, not in the plugin's data directory, so uninstalling the plugin does not delete them. Shared server: `claude mcp add --transport http holographic-memory http://127.0.0.1:8765/mcp --header "Authorization: Bearer $HMS_MCP_TOKEN"`.

## Cursor

`.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global), top-level `mcpServers`. The provided entry sets `type: "stdio"`, starts the server per project (`HMS_MCP_PROJECT=${workspaceFolder}`) and uses the `project` namespace strategy. Remote: `{"mcpServers": {"holographic-memory": {"url": "http://127.0.0.1:8765/mcp", "headers": {"Authorization": "Bearer ${env:HMS_MCP_TOKEN}"}}}}`.

## Continue

A block file in `.continue/mcpServers/` with `name`, `version` and `schema: v1`, and an `mcpServers` list; `type` is `stdio`, `sse` or `streamable-http`. MCP tools work in agent mode only. Remote: `type: streamable-http` and `url: http://127.0.0.1:8765/mcp`.

## Windsurf

`mcp_config.json` with `mcpServers`; remote servers use `serverUrl` (or `url`) and `headers`, and `${env:NAME}` is expanded in them. Windsurf's documentation now lives under Devin Desktop and names `~/.config/devin/mcp_config.json` (`%APPDATA%\devin\mcp_config.json` on Windows) for that product; the long-standing Windsurf path is `~/.codeium/windsurf/mcp_config.json` and is what the table uses. Cascade allows 100 tools in total; this server adds 8.

## Zed

`settings.json` under `context_servers`: local servers take `command`, `args`, `env`; remote servers take `url` and `headers`. Open it with the `zed: open settings file` action.

## VS Code

`.vscode/mcp.json` (workspace) or the user-profile `mcp.json`, top-level `servers` (not `mcpServers`). Remote: `{"servers": {"holographic-memory": {"type": "http", "url": "http://127.0.0.1:8765/mcp", "headers": {"Authorization": "Bearer ${input:hms-token}"}}}, "inputs": [{"type": "promptString", "id": "hms-token", "description": "hms-mcp token", "password": true}]}`.

## Cline

`cline_mcp_settings.json` with `mcpServers`: open it from the MCP Servers panel (Configure, then Configure MCP Servers); the CLI reads `~/.cline/data/settings/cline_mcp_settings.json` or `CLINE_MCP_SETTINGS_PATH`. The provided entry auto-approves only the read-only tools. Remote: `{"type": "streamableHttp", "url": "http://127.0.0.1:8765/mcp", "headers": {"Authorization": "Bearer <token>"}}` (without `type` Cline assumes the legacy SSE transport).

## Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS, `%APPDATA%\Claude\claude_desktop_config.json` on Windows, `mcpServers` for local stdio servers; restart the app after editing. Logs are in `~/Library/Logs/Claude/mcp-server-holographic-memory.log`. The config file is for local servers; add a shared HTTP server through the app's connector settings instead.

## Gemini CLI

`~/.gemini/settings.json` (`-s user`) or `.gemini/settings.json` (project, the default scope), `mcpServers`, with `httpUrl` for streamable HTTP. The extension in `integrations/editors/gemini-cli` bundles the server and a context file (`HOLOGRAPHIC-MEMORY.md`) that tells the model how to use it; `gemini extensions link <path>` installs it from a checkout. A folder must be trusted before project or user MCP servers run in it. Shared server: `gemini mcp add --transport http holographic-memory http://127.0.0.1:8765/mcp --header "Authorization: Bearer $HMS_MCP_TOKEN"`.

## Codex CLI

`~/.codex/config.toml`, or `.codex/config.toml` in a trusted project, `[mcp_servers.<name>]`. Remote: `url = "http://127.0.0.1:8765/mcp"` and `bearer_token_env_var = "HMS_MCP_TOKEN"`. `startup_timeout_sec = 60` in the provided entry covers the first-run download.
