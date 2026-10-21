# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Pydantic AI: memory toolset plus persistent message history.

Needs a running hms-server (HMS_URL, default http://127.0.0.1:8080, default --input-dim 384).
A FunctionModel stands in for the LLM; pass a model name to `Agent(...)` to use a real one.

    uv run --extra pydantic-ai python examples/pydantic_ai_memory.py
"""

import os
import uuid

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from holographic_integrations.core import HmsConfig
from holographic_integrations.pydantic_ai import MessageStore, memory_toolset

config = HmsConfig(url=os.environ.get("HMS_URL", "http://127.0.0.1:8080"), tenant_id=f"demo{uuid.uuid4().hex[:8]}")

steps = iter([
    ToolCallPart("remember", {"text": "Release freeze starts on the 14th"}),
    TextPart("Stored."),
    ToolCallPart("recall", {"query": "when is the release freeze"}),
    TextPart("The freeze starts on the 14th."),
])


def model(messages, info):
    part = next(steps)
    return ModelResponse(parts=[part])


agent = Agent(FunctionModel(model), toolsets=[memory_toolset(config, user_id="alice", sync=True)])
store = MessageStore(config)

for prompt in ("Remember the release freeze date.", "When is the freeze?"):
    result = agent.run_sync(prompt, message_history=store.load("chat-1"))
    store.save("chat-1", result.all_messages())
    print(result.output)

print(f"{len(store.load('chat-1'))} messages persisted")
store.close()
