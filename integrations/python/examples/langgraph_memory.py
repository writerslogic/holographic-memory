# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""LangGraph: a graph that resumes from a WARM checkpoint, plus a long-term store.

Needs a running warm-server (`--input-dim 384`, the default). Set WARM_URL / WARM_API_KEY if not local.
"""

import operator
import os
import uuid
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from writerslogic_warm_integrations.core import HashingEmbedder
from writerslogic_warm_integrations.langchain import EmbedderAsLangChain
from writerslogic_warm_integrations.langgraph import WarmSaver, WarmStore

URL = os.environ.get("WARM_URL", "http://127.0.0.1:8080")
KEY = os.environ.get("WARM_API_KEY")
TENANT = "example-" + uuid.uuid4().hex[:8]


class State(TypedDict):
    log: Annotated[list, operator.add]


def build(saver, store):
    def remember(state, *, store):
        store.put(("facts", "ada"), "lang", {"text": "Ada prefers Rust for systems code"})
        return {"log": ["remembered"]}

    def recall(state, *, store):
        hits = store.search(("facts", "ada"), query="which language for systems work", limit=1)
        return {"log": [f"recalled: {hits[0].value['text']}"]}

    graph = StateGraph(State)
    graph.add_node("remember", remember)
    graph.add_node("recall", recall)
    graph.add_edge(START, "remember")
    graph.add_edge("remember", "recall")
    graph.add_edge("recall", END)
    return graph.compile(checkpointer=saver, store=store)


saver = WarmSaver(url=URL, api_key=KEY, tenant_id=TENANT)
store = WarmStore(
    index={"embed": EmbedderAsLangChain(HashingEmbedder(384)), "dims": 384, "fields": ["text"]},
    url=URL, api_key=KEY, tenant_id=TENANT,
)
config = {"configurable": {"thread_id": "demo"}}
print(build(saver, store).invoke({"log": []}, config))

# a fresh process would rebuild the same objects and read the same state
fresh = WarmSaver(url=URL, api_key=KEY, tenant_id=TENANT)
print(build(fresh, store).get_state(config).values)
print(store.list_namespaces())
