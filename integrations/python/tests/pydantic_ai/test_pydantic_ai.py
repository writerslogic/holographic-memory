# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from holographic_integrations.core import HmsConfig
from holographic_integrations.pydantic_ai import AsyncMessageStore, MessageStore, memory_toolset


def cfg(server, **kw):
    return HmsConfig(url=server.url, tenant_id=server.tenant(), input_dim=server.dim, **kw)


def script(*steps):
    """A FunctionModel that plays one step per model request: a (tool, args) call or text."""
    queue = list(steps)

    def fn(messages, info: AgentInfo) -> ModelResponse:
        step = queue.pop(0)
        if isinstance(step, tuple):
            return ModelResponse(parts=[ToolCallPart(step[0], step[1])])
        return ModelResponse(parts=[TextPart(step)])

    return FunctionModel(fn)


def tool_returns(result):
    return [p.content for m in result.all_messages() if isinstance(m, ModelRequest)
            for p in m.parts if isinstance(p, ToolReturnPart)]


@pytest.mark.parametrize("sync", [False, True])
async def test_toolset_in_a_real_agent_run(server, sync):
    ts = memory_toolset(cfg(server), user_id="u1", sync=sync)
    agent = Agent(script(("remember", {"text": "the vault code word is plover"}),
                         ("recall", {"query": "vault code word plover"}), "done"), toolsets=[ts])
    r = await agent.run("go")
    assert r.output == "done"
    saved, found = tool_returns(r)
    assert saved in found and "plover" in found


async def test_tool_schemas_are_what_the_model_sees(server):
    seen = {}

    def fn(messages, info: AgentInfo) -> ModelResponse:
        seen.update({t.name: t.parameters_json_schema for t in info.function_tools})
        return ModelResponse(parts=[TextPart("ok")])

    await Agent(FunctionModel(fn), toolsets=[memory_toolset(cfg(server))]).run("x")
    assert set(seen) == {"remember", "recall", "forget"}
    assert seen["remember"]["required"] == ["text"] and "tags" in seen["remember"]["properties"]
    assert seen["recall"]["required"] == ["query"]


async def test_user_scoping_and_forget_ownership(server):
    c = cfg(server)
    mine, theirs = memory_toolset(c, user_id="u1"), memory_toolset(c, user_id="u2")
    agent_a = Agent(script(("remember", {"text": "private fact about badgers"}), "ok"), toolsets=[mine])
    mem_id = tool_returns(await agent_a.run("x"))[0]
    agent_b = Agent(script(("recall", {"query": "private fact about badgers"}), "ok"), toolsets=[theirs])
    out = tool_returns(await agent_b.run("x"))
    assert out == ["No matching memories."]
    agent_c = Agent(script(("forget", {"memory_id": mem_id}), "ok"), toolsets=[theirs])
    assert tool_returns(await agent_c.run("x")) == ["No such memory."]
    agent_d = Agent(script(("forget", {"memory_id": mem_id}), "ok"), toolsets=[mine])
    assert tool_returns(await agent_d.run("x")) == ["Deleted."]


async def test_testmodel_calls_every_tool_without_error(server):
    agent = Agent(TestModel(), toolsets=[memory_toolset(cfg(server))])
    r = await agent.run("exercise tools")
    assert len(tool_returns(r)) == 3


def test_message_store_roundtrip_and_resume(server):
    store = MessageStore(cfg(server))
    agent = Agent(TestModel(custom_output_text="reply"))
    r1 = agent.run_sync("first question", message_history=store.load("c1"))
    store.save("c1", r1.all_messages())
    r2 = agent.run_sync("second question", message_history=store.load("c1"))
    store.save("c1", r2.all_messages())
    stored = store.load("c1")
    assert stored == r2.all_messages() and len(stored) == 4
    assert store.load("c1", limit=1) == r2.all_messages()[-1:]
    assert store.load("other") == []
    store.delete("c1")
    assert store.load("c1") == []
    store.close()


def test_save_is_incremental_and_handles_compaction(server):
    store = MessageStore(cfg(server))
    msgs = [ModelRequest.user_text_prompt(f"m{i}") for i in range(7)]
    store.save("k", msgs[:3])
    store.save("k", msgs[:5])
    assert store.load("k") == msgs[:5]
    store.save("k", msgs[3:5])  # shorter than stored: rewritten
    assert store.load("k") == msgs[3:5]
    store.replace("k", msgs[:1])
    assert store.load("k") == msgs[:1]


@pytest.mark.parametrize("n", [49, 50, 51, 120])
async def test_async_store_beyond_one_page(server, n):
    store = AsyncMessageStore(cfg(server))
    msgs = [ModelRequest.user_text_prompt(f"m{i:03d}") for i in range(n)]
    await store.save("long", msgs)
    assert await store.load("long") == msgs
    assert await store.load("long", limit=2) == msgs[-2:]
    await store.aclose()


async def test_large_tool_return_survives_roundtrip(server):
    store = AsyncMessageStore(cfg(server))
    big = ModelRequest(parts=[ToolReturnPart("t", "é" * 30000, tool_call_id="c")])
    await store.save("big", [big])
    assert await store.load("big") == [big]
    await store.aclose()
