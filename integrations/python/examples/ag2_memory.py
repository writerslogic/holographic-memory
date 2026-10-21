# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""AG2 agent with Holographic Memory tools.

    HMS_API_KEY=demo hms-server --input-dim 64 --dim 1024
    HMS_URL=http://127.0.0.1:8080 HMS_API_KEY=demo python ag2_memory.py

Offline: ag2's scripted TestConfig plays the model, calling `remember` then `recall`.
"""

import asyncio
import json
import os

from ag2 import Agent
from ag2.events import ToolCallEvent
from ag2.testing import TestConfig

from holographic_integrations.ag2 import HolographicMemoryToolkit
from holographic_integrations.core import HashingEmbedder, HmsConfig


async def main() -> None:
    hms = HmsConfig(
        url=os.environ.get("HMS_URL", "http://127.0.0.1:8080"),
        api_key=os.environ.get("HMS_API_KEY"),
        tenant_id="ag2_demo",
        embedder=HashingEmbedder(64),
    )
    memory = HolographicMemoryToolkit(hms)
    await memory.backend.forget_where(None)
    script = TestConfig(
        ToolCallEvent(name="remember", arguments=json.dumps({"text": "Deploys happen on Fridays", "metadata": {"team": "ops"}})),
        ToolCallEvent(name="recall", arguments=json.dumps({"query": "when do deploys happen", "filter": {"team": "ops"}})),
        "Deploys happen on Fridays.",
    )
    agent = Agent("assistant", config=script, tools=[memory])
    reply = await agent.ask("Note that deploys happen on Fridays, then tell me when deploys happen.")
    print(await reply.content())
    print([r.text for r in (await memory.backend.list()).records])
    await memory.aclose()


asyncio.run(main())
