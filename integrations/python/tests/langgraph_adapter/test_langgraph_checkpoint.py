# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""langgraph-checkpoint-conformance against a live warm-server, plus real graphs."""

import operator
from typing import Annotated, TypedDict

import pytest
from langgraph.checkpoint.conformance import checkpointer_test, validate
from langgraph.checkpoint.conformance.capabilities import ALL_CAPABILITIES
from langgraph.checkpoint.conformance.report import ProgressCallbacks
from langgraph.graph import END, START, StateGraph

from writerslogic_warm_integrations.core import WarmConfig
from writerslogic_warm_integrations.langgraph import WarmSaver

DIM = 8


def saver_for(server, **kw):
    return WarmSaver(url=server.url, tenant_id=server.tenant(), input_dim=DIM, **kw)


async def test_checkpoint_conformance_all_capabilities(server_factory):
    server = server_factory(DIM)

    @checkpointer_test(name="WarmSaver")
    async def factory():
        saver = saver_for(server)
        yield saver
        await saver.aclose()

    report = await validate(factory, progress=ProgressCallbacks.quiet())
    failures = {cap: r.failures for cap, r in report.results.items() if r.failures}
    assert not failures, failures
    assert {c.value for c in ALL_CAPABILITIES} == {c for c, r in report.results.items() if r.detected}
    assert report.conformance_level() == "FULL"


class State(TypedDict):
    items: Annotated[list, operator.add]
    big: str


def build(saver):
    def a(state):
        return {"items": ["a"], "big": "x" * 200_000}

    def b(state):
        return {"items": ["b"]}

    graph = StateGraph(State)
    graph.add_node("a", a)
    graph.add_node("b", b)
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    return graph.compile(checkpointer=saver)


def test_graph_resumes_across_saver_instances_and_stores_large_values(server_factory):
    server = server_factory(DIM)
    tenant = server.tenant()
    first = WarmSaver(url=server.url, tenant_id=tenant, input_dim=DIM)
    cfg = {"configurable": {"thread_id": "t1"}}
    out = build(first).invoke({"items": [], "big": ""}, cfg)
    assert out["items"] == ["a", "b"] and len(out["big"]) == 200_000  # 200 kB blob spans several parts
    second = WarmSaver(url=server.url, tenant_id=tenant, input_dim=DIM)
    state = build(second).get_state(cfg)
    assert state.values["items"] == ["a", "b"] and len(state.values["big"]) == 200_000
    history = list(build(second).get_state_history(cfg))
    assert [h.metadata["step"] for h in history] == [2, 1, 0, -1]
    first.close()
    second.close()


async def test_async_graph_and_long_thread_lists_across_pages(server_factory):
    server = server_factory(DIM)
    saver = saver_for(server)
    cfg = {"configurable": {"thread_id": "long"}}

    class S(TypedDict):
        n: int

    def inc(state):
        return {"n": state["n"] + 1}

    graph = StateGraph(S)
    graph.add_node("inc", inc)
    graph.add_edge(START, "inc")
    graph.add_conditional_edges("inc", lambda s: END if s["n"] >= 60 else "inc")
    app = graph.compile(checkpointer=saver)
    out = await app.ainvoke({"n": 0}, {**cfg, "recursion_limit": 200})
    assert out["n"] == 60
    history = [t async for t in saver.alist(cfg)]
    assert len(history) == 62 and len(history) > 50  # spans two sequence pages
    ids = [t.checkpoint["id"] for t in history]
    assert ids == sorted(ids, reverse=True)
    assert [t.checkpoint["id"] for t in [x async for x in saver.alist(cfg, limit=3)]] == ids[:3]
    before = {"configurable": {**cfg["configurable"], "checkpoint_id": ids[10]}}
    assert [t.checkpoint["id"] async for t in saver.alist(cfg, before=before)] == ids[11:]
    await saver.aclose()


def test_checkpoint_page_limit_is_enforced(server_factory):
    server = server_factory(DIM)
    with pytest.raises(ValueError):
        WarmSaver(config=WarmConfig(url=server.url, tenant_id=server.tenant(), input_dim=DIM, max_fetch=10))
