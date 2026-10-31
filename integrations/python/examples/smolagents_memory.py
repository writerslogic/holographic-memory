# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""smolagents: a CodeAgent that remembers and recalls through WARM.

Needs a running warm-server (WARM_URL, default http://127.0.0.1:8080, default --input-dim 384).
A scripted model stands in for the LLM; use `InferenceClientModel`/`LiteLLMModel` for a real one.

    uv run --extra smolagents python examples/smolagents_memory.py
"""

import os
import uuid

from smolagents import CodeAgent, Model
from smolagents.models import ChatMessage

from writerslogic_warm_integrations.core import WarmConfig
from writerslogic_warm_integrations.smolagents import memory_tools


class Script(Model):
    def __init__(self, *bodies):
        super().__init__()
        self.bodies = list(bodies)

    def generate(self, messages, **kwargs):
        return ChatMessage(role="assistant", content=f"Thought: next\n<code>\n{self.bodies.pop(0)}\n</code>")


config = WarmConfig(url=os.environ.get("WARM_URL", "http://127.0.0.1:8080"), tenant_id=f"demo{uuid.uuid4().hex[:8]}")
agent = CodeAgent(
    tools=memory_tools(config, user_id="alice"),
    model=Script(
        'print(remember(text="Standup moved to 09:30 on Tuesdays"))',
        'final_answer(recall(query="when is standup"))',
    ),
    add_base_tools=False,
)
print(agent.run("Remember the standup time, then look it up."))
