# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""LiteLLM: OpenAI-format memory tools and a tool loop.

Needs a running hms-server (HMS_URL, default http://127.0.0.1:8080, default --input-dim 384).
LiteLLM's `mock_*` arguments stand in for the LLM, one scripted turn per call; drop the
wrapper and pass a real `model=` to use a provider.

    uv run --extra litellm python examples/litellm_memory.py
"""

import json
import os
import uuid

import litellm

from holographic_integrations.core import HmsConfig
from holographic_integrations.litellm import MemoryTools

config = HmsConfig(url=os.environ.get("HMS_URL", "http://127.0.0.1:8080"), tenant_id=f"demo{uuid.uuid4().hex[:8]}")
tools = MemoryTools(config, user_id="alice")

turns = iter([
    [{"id": "1", "type": "function", "function": {"name": "remember", "arguments": json.dumps({"text": "Alice prefers dark roast coffee"})}}],
    [{"id": "2", "type": "function", "function": {"name": "recall", "arguments": json.dumps({"query": "what coffee does alice like"})}}],
    "Alice prefers dark roast.",
])


def scripted(**kwargs):
    turn = next(turns)
    return litellm.completion(**kwargs, **({"mock_tool_calls": turn} if isinstance(turn, list) else {"mock_response": turn}))


messages = [{"role": "user", "content": "What coffee does Alice like? Remember it first."}]
print(tools.run(messages, model="gpt-4o", completion=scripted).choices[0].message.content)
print([m["role"] for m in messages])
tools.close()
