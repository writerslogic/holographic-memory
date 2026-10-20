# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Structural validation of every host config in this directory against the documented schema.

Run: uv run --with pyyaml python integrations/editors/validate.py

This checks shape only (keys, types, required fields). Hosts that can be installed on the
machine are additionally exercised by their own CLI; see docs/integrations/editors/README.md.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
ENV_NAME = "HMS_MCP_"


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def check_stdio(name: str, entry: dict, *, allowed: set[str], where: str) -> None:
    assert isinstance(entry.get("command"), str) and entry["command"], f"{where}: command required"
    assert isinstance(entry.get("args", []), list) and all(isinstance(a, str) for a in entry.get("args", [])), f"{where}: args must be strings"
    env = entry.get("env", {})
    assert isinstance(env, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()), f"{where}: env must map strings"
    assert all(k.startswith(ENV_NAME) for k in env), f"{where}: env keys must be HMS_MCP_*"
    unknown = set(entry) - allowed
    assert not unknown, f"{where}: unknown keys {sorted(unknown)}"
    assert name == "holographic-memory", f"{where}: server name"


def mcp_servers(path: str, key: str, allowed: set[str]) -> None:
    servers = load_json(path)[key]
    assert list(servers) == ["holographic-memory"], f"{path}: one server named holographic-memory"
    check_stdio("holographic-memory", servers["holographic-memory"], allowed=allowed, where=path)


def main() -> int:
    base = {"command", "args", "env"}
    mcp_servers("cursor/mcp.json", "mcpServers", base | {"type", "envFile"})
    assert load_json("cursor/mcp.json")["mcpServers"]["holographic-memory"]["type"] == "stdio"
    mcp_servers("windsurf/mcp_config.json", "mcpServers", base | {"disabledTools"})
    mcp_servers("zed/settings.json", "context_servers", base)
    mcp_servers("vscode/mcp.json", "servers", base | {"type", "cwd", "envFile", "dev", "sandboxEnabled", "sandbox"})
    assert load_json("vscode/mcp.json")["servers"]["holographic-memory"]["type"] == "stdio"
    mcp_servers("cline/cline_mcp_settings.json", "mcpServers", base | {"disabled", "autoApprove", "timeout", "type"})
    mcp_servers("claude-desktop/claude_desktop_config.json", "mcpServers", base)
    mcp_servers("claude-code/plugin/.mcp.json", "mcpServers", base | {"type"})

    ext = load_json("gemini-cli/gemini-extension.json")
    assert ext["name"] == "holographic-memory" and ext["version"] and ext["description"]
    assert set(ext) <= {"name", "version", "description", "mcpServers", "contextFileName", "excludeTools", "settings", "themes", "migratedTo", "plan"}
    check_stdio("holographic-memory", ext["mcpServers"]["holographic-memory"], allowed=base | {"cwd", "timeout"}, where="gemini-extension.json")
    assert (ROOT / "gemini-cli" / ext["contextFileName"]).is_file()

    doc = yaml.safe_load((ROOT / "continue/holographic-memory.yaml").read_text())
    assert {"name", "version", "schema", "mcpServers"} <= set(doc) and doc["schema"] == "v1"
    (server,) = doc["mcpServers"]
    assert server["type"] == "stdio" and server["command"] and all(isinstance(a, str) for a in server["args"])

    toml = tomllib.loads((ROOT / "codex/config.toml").read_text())
    entry = toml["mcp_servers"]["holographic-memory"]
    assert entry["command"] == "npx" and isinstance(entry["args"], list)
    assert set(entry) <= {"command", "args", "env", "env_vars", "cwd", "enabled", "enabled_tools", "disabled_tools", "startup_timeout_sec", "tool_timeout_sec"}

    plugin = load_json("claude-code/plugin/.claude-plugin/plugin.json")
    assert plugin["name"] == "holographic-memory" and not plugin["name"].startswith(("claude", "anthropic"))
    market = load_json("claude-code/.claude-plugin/marketplace.json")
    assert market["plugins"][0]["name"] == plugin["name"] and market["plugins"][0]["source"] == "./plugin"
    assert (ROOT / "claude-code" / "plugin" / "skills/holographic-memory/SKILL.md").read_text().startswith("---\nname: holographic-memory\n")
    hooks = load_json("claude-code/plugin/hooks/hooks.json")["hooks"]
    assert set(hooks) == {"SessionStart"}
    for group in hooks["SessionStart"]:
        for hook in group["hooks"]:
            assert hook["type"] == "command" and hook["command"] == "node" and hook["args"][0].startswith("${CLAUDE_PLUGIN_ROOT}/scripts/")
            assert (ROOT / "claude-code" / "plugin" / hook["args"][0].replace("${CLAUDE_PLUGIN_ROOT}/", "")).is_file()
    print("all host configs are structurally valid")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as error:
        print(f"invalid: {error}", file=sys.stderr)
        sys.exit(1)
