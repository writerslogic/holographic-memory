# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Check the container's stdio handshake and inspectable tool definitions."""

import json
import subprocess
import sys


messages = [
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "container-smoke", "version": "1.0"},
        },
    },
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
]
request = "\n".join(json.dumps(message) for message in messages) + "\n"
process = subprocess.run(
    sys.argv[1:], input=request, text=True, capture_output=True, timeout=45, check=True
)
responses = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
assert any(response.get("id") == 1 and "result" in response for response in responses)
tools = next(response["result"]["tools"] for response in responses if response.get("id") == 2)
assert len(tools) == 8
for tool in tools:
    assert tool["outputSchema"]["type"] == "object", tool["name"]
    assert all(
        property_schema.get("description")
        for property_schema in tool["inputSchema"]["properties"].values()
    ), tool["name"]
    assert "readOnlyHint" in tool["annotations"], tool["name"]
print("MCP stdio inspection passed for", ", ".join(tool["name"] for tool in tools))
