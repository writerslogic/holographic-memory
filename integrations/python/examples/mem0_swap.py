# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Swap mem0 for Holographic Memory with one import change.

    hms-server --input-dim 384 &            # the default embedder is a 384-dim hashing embedder
    python mem0_swap.py

Environment: HMS_URL (default http://127.0.0.1:8080), HMS_API_KEY, HMS_TENANT_ID, HMS_INPUT_DIM (default 384).
"""

import os
import warnings

from holographic_integrations.core import HmsConfig
from holographic_integrations.mem0 import LLMExtractor, Memory  # was: from mem0 import Memory

warnings.simplefilter("ignore", UserWarning)  # silences the "stored verbatim" notice for the infer=False style adds below

dim = int(os.environ.get("HMS_INPUT_DIM", "384"))
memory = Memory(
    hms=HmsConfig(
        url=os.environ.get("HMS_URL", "http://127.0.0.1:8080"),
        api_key=os.environ.get("HMS_API_KEY"),
        tenant_id=os.environ.get("HMS_TENANT_ID"),
        input_dim=dim,
    )
)

memory.add("I am vegetarian and I love hiking in the Alps", user_id="alice", metadata={"source": "chat"}, infer=False)
memory.add("My sister lives in Oslo", user_id="alice", metadata={"source": "chat"}, infer=False)

for hit in memory.search("what does alice eat", filters={"user_id": "alice"}, threshold=0.0, top_k=2)["results"]:
    print(round(hit["score"], 3), hit["memory"], hit.get("metadata"))

first = memory.get_all(filters={"user_id": "alice", "source": "chat"})["results"][0]
memory.update(first["id"], text=first["memory"] + " (updated)")
print([h["event"] for h in memory.history(first["id"])])
memory.delete_all(user_id="alice")
print(memory.get_all(filters={"user_id": "alice"}))

# Fact extraction needs a chat function of your choice (any provider, or a local model); this one is a stand-in.
extracting = Memory(
    hms=memory.settings.hms,
    extractor=LLMExtractor(lambda messages: '{"memory": [{"text": "Alice is vegetarian"}]}'),
)
print(extracting.add("I only eat vegetables", user_id="alice")["results"])
extracting.delete_all(user_id="alice")
