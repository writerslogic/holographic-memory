# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Letta archival memory on Holographic Memory.

Use `ArchivalMemory` where you would use `client.agents.passages`. To give a Letta agent the same memory through
custom tools, call `register_tools(client)` and create the agent with `secrets=agent_secrets(url, ...)`.

Environment: HMS_URL, HMS_API_KEY, HMS_TENANT_ID, HMS_INPUT_DIM (default 384).
"""

import os

from holographic_integrations.core import HmsConfig
from holographic_integrations.letta import ArchivalMemory

AGENT_ID = "agent-11111111-1111-4111-8111-111111111111"

config = HmsConfig(
    url=os.environ.get("HMS_URL", "http://127.0.0.1:8080"),
    api_key=os.environ.get("HMS_API_KEY"),
    tenant_id=os.environ.get("HMS_TENANT_ID"),
    input_dim=int(os.environ.get("HMS_INPUT_DIM", "384")),
)

with ArchivalMemory(config) as passages:
    passages.create(AGENT_ID, text="Meeting 2024-03-15: decided to focus on performance work", tags=["meetings"])
    passages.create(AGENT_ID, text="Lunch order was pasta", tags=["food"])
    result = passages.search(AGENT_ID, query="what did the meeting decide about performance", tags=["meetings"], top_k=3)
    for hit in result.results:
        print(hit.id, hit.timestamp, hit.content, hit.tags)
    for passage in passages.list(AGENT_ID):
        passages.delete(passage.id, agent_id=AGENT_ID)
    print(passages.list(AGENT_ID))
