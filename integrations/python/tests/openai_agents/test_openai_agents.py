# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

pytest.importorskip("agents")

from agents import Agent, Runner
from agents.items import ModelResponse
from agents.tool_context import ToolContext
from agents.memory import Session
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

from writerslogic_warm_integrations.core import WarmConfig
from writerslogic_warm_integrations.openai_agents import WarmSession, memory_tools


def cfg(server, **kw):
    return WarmConfig(url=server.url, tenant_id=server.tenant(), input_dim=server.dim, **kw)


def message(text):
    return ResponseOutputMessage(
        id="m", type="message", role="assistant", status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def tool_call(name, args, call_id):
    return ResponseFunctionToolCall(type="function_call", call_id=call_id, name=name, arguments=json.dumps(args))


class ScriptedModel(Model):
    """Pops one scripted output list per model call and records the input it was given."""

    def __init__(self, script):
        self.script, self.inputs = list(script), []

    async def get_response(self, system_instructions, input, model_settings, tools, output_schema,
                           handoffs, tracing, *, previous_response_id, conversation_id, prompt):
        self.inputs.append(input)
        return ModelResponse(output=self.script.pop(0), usage=Usage(), response_id=None)

    def stream_response(self, *a, **k):
        raise NotImplementedError


async def invoke(tool, **args):
    raw = json.dumps(args)
    ctx = ToolContext(None, tool_name=tool.name, tool_call_id="x", tool_arguments=raw)
    return await tool.on_invoke_tool(ctx, raw)


def user(text):
    return {"role": "user", "content": text}


async def test_is_a_session_and_runner_replays_history(server):
    session = WarmSession("conv-1", cfg(server))
    assert isinstance(session, Session)
    model = ScriptedModel([[message("first reply")], [message("second reply")]])
    agent = Agent(name="a", model=model)
    r1 = await Runner.run(agent, "hello there", session=session)
    assert r1.final_output == "first reply"
    r2 = await Runner.run(agent, "and again", session=session)
    assert r2.final_output == "second reply"
    second_input = model.inputs[1]
    contents = [json.dumps(i) for i in second_input]
    assert any("hello there" in c for c in contents) and any("first reply" in c for c in contents)
    stored = await session.get_items()
    assert [i.get("role") for i in stored] == ["user", "assistant", "user", "assistant"]
    # another process: a fresh session object over the same tenant sees the same history
    again = WarmSession("conv-1", WarmConfig(url=server.url, tenant_id=session.backend.config.tenant_id, input_dim=server.dim))
    assert await again.get_items() == stored
    await session.aclose()
    await again.aclose()


async def test_sessions_are_isolated_by_id(server):
    c = cfg(server)
    a, b = WarmSession("a", c), WarmSession("b", c)
    await a.add_items([user("for a")])
    await b.add_items([user("for b")])
    assert await a.get_items() == [user("for a")]
    await a.clear_session()
    assert await a.get_items() == [] and await b.get_items() == [user("for b")]


@pytest.mark.parametrize("n", [49, 50, 51, 100, 101, 230])
async def test_order_and_limits_across_page_boundaries(server, n):
    s = WarmSession("big", cfg(server))
    items = [user(f"msg {i:04d}") for i in range(n)]
    for start in range(0, n, 37):  # several add_items calls, not aligned with pages
        await s.add_items(items[start : start + 37])
    assert await s.get_items() == items
    assert await s.get_items(limit=1) == items[-1:]
    assert await s.get_items(limit=n + 5) == items
    assert await s.get_items(limit=60) == items[-60:]
    assert await s.get_items(limit=0) == []
    assert await s.pop_item() == items[-1]
    assert await s.get_items() == items[:-1]
    await s.add_items([user("after pop")])
    assert (await s.get_items(limit=2))[-1] == user("after pop")
    await s.clear_session()
    assert await s.get_items() == [] and await s.pop_item() is None


async def test_default_limit_from_session_settings(server):
    s = WarmSession("lim", cfg(server), session_settings={"limit": 3})
    await s.add_items([user(str(i)) for i in range(10)])
    assert [i["content"] for i in await s.get_items()] == ["7", "8", "9"]
    assert len(await s.get_items(limit=5)) == 5


async def test_large_item_spans_parts_and_pop_removes_them(server):
    s = WarmSession("parts", cfg(server), chunk_chars=500, max_parts=4)
    big = {"type": "function_call_output", "call_id": "c", "output": "é" * 1200}
    await s.add_items([user("before"), big, user("after")])
    assert await s.get_items() == [user("before"), big, user("after")]
    await s.pop_item()
    assert await s.pop_item() == big
    assert s.backend.config.collection is None
    assert len((await s.backend.list()).records) == 1  # no orphaned part records
    with pytest.raises(ValueError):
        await s.add_items([{"role": "user", "content": "x" * 3000}])  # needs 7 parts > 4
    assert await s.get_items() == [user("before")]


async def test_page_size_must_fit_service_cap(server):
    with pytest.raises(ValueError):
        WarmSession("x", cfg(server, max_fetch=20), page_size=21)


async def test_memory_tools_drive_a_real_agent_loop(server):
    c = cfg(server)
    tools = memory_tools(c, user_id="u1")
    names = [t.name for t in tools]
    assert names == ["remember", "recall", "forget"]
    model = ScriptedModel([
        [tool_call("remember", {"text": "the launch code word is plover", "tags": ["secret"]}, "c1")],
        [message("noted")],
        [tool_call("recall", {"query": "launch code word plover"}, "c2")],
        [message("done")],
    ])
    agent = Agent(name="m", model=model, tools=tools)
    await Runner.run(agent, "remember the code word")
    res = await Runner.run(agent, "what was the code word?")
    outputs = [i for i in model.inputs[3] if isinstance(i, dict) and i.get("type") == "function_call_output"]
    assert outputs and "plover" in outputs[-1]["output"]
    assert res.final_output == "done"
    # memories written for u1 are invisible to u2
    other = memory_tools(c, user_id="u2")
    out = await invoke(other[1], **{"query": "launch code word plover"})
    assert out == "No matching memories."


async def test_forget_respects_ownership(server):
    c = cfg(server)
    mine = memory_tools(c, user_id="u1")
    theirs = memory_tools(c, user_id="u2")
    mem_id = await invoke(mine[0], **{"text": "private"})
    assert await invoke(theirs[2], **{"memory_id": mem_id}) == "No such memory."
    assert await invoke(mine[2], **{"memory_id": mem_id}) == "Deleted."


def test_sync_tools(server):
    tools = memory_tools(cfg(server), sync=True)
    import asyncio

    async def go():
        mid = await invoke(tools[0], **{"text": "sync note about badgers"})
        found = await invoke(tools[1], **{"query": "sync note about badgers"})
        assert mid in found and "badgers" in found

    asyncio.run(go())
