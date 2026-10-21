# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""AutoGen AssistantAgent with Holographic Memory.

    HMS_API_KEY=demo hms-server --input-dim 64 --dim 1024
    HMS_URL=http://127.0.0.1:8080 HMS_API_KEY=demo python autogen_memory.py

Offline: a replay model client stands in for the LLM and HashingEmbedder for the encoder.
"""

import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.models.replay import ReplayChatCompletionClient

from holographic_integrations.autogen import HolographicMemory
from holographic_integrations.core import HashingEmbedder, HmsConfig


async def main() -> None:
    hms = HmsConfig(
        url=os.environ.get("HMS_URL", "http://127.0.0.1:8080"),
        api_key=os.environ.get("HMS_API_KEY"),
        tenant_id="autogen_demo",
        embedder=HashingEmbedder(64),
    )
    memory = HolographicMemory("assistant_notes", hms=hms, k=3)
    await memory.clear()
    await memory.add(MemoryContent(content="The staging password rotates every Monday", mime_type=MemoryMimeType.TEXT))

    client = ReplayChatCompletionClient(["It rotates every Monday."])
    agent = AssistantAgent("helper", model_client=client, memory=[memory])
    result = await agent.run(task="How often does the staging password rotate?")
    print(result.messages[-1].content)
    prompt = " ".join(str(m.content) for m in client.create_calls[0]["messages"])
    print("memory reached the model:", "rotates every Monday" in prompt)
    await memory.close()


asyncio.run(main())
