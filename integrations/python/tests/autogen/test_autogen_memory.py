# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("autogen_agentchat")

from autogen_agentchat.agents import AssistantAgent  # noqa: E402
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType  # noqa: E402
from autogen_core.model_context import BufferedChatCompletionContext  # noqa: E402
from autogen_core.models import UserMessage  # noqa: E402
from autogen_ext.models.replay import ReplayChatCompletionClient  # noqa: E402

from writerslogic_warm_integrations.autogen import WarmMemory  # noqa: E402
from writerslogic_warm_integrations.core import HashingEmbedder, WarmConfig  # noqa: E402

DIM = 64


def make(server, name="mem", **kw):
    warm = WarmConfig(url=server.url, tenant_id=server.tenant(), input_dim=DIM, embedder=HashingEmbedder(DIM))
    return WarmMemory(name, warm=warm, **kw)


def text(s, **meta):
    return MemoryContent(content=s, mime_type=MemoryMimeType.TEXT, metadata=meta or None)


async def test_is_autogen_memory_and_query_roundtrip(server):
    mem = make(server)
    assert isinstance(mem, Memory)
    await mem.add(text("The user prefers dark roast coffee", topic="food", id="m1"))
    await mem.add(text("The deploy runs on Fridays at noon", topic="ops"))
    res = await mem.query("dark roast coffee preference", k=1)
    assert [r.content for r in res.results] == ["The user prefers dark roast coffee"]
    meta = res.results[0].metadata
    assert meta["id"] == "m1" and meta["topic"] == "food" and 0 < meta["score"] <= 1
    assert res.results[0].mime_type == MemoryMimeType.TEXT
    await mem.close()


async def test_filter_threshold_and_json_content(server):
    mem = make(server)
    await mem.add(MemoryContent(content={"fact": "alpha", "n": 1}, mime_type=MemoryMimeType.JSON, metadata={"kind": "j", "rank": 3}))
    await mem.add(text("alpha plain text", kind="t", rank=1))
    by_kind = await mem.query("alpha", filter={"kind": "j"})
    assert [r.content for r in by_kind.results] == [{"fact": "alpha", "n": 1}]
    ranged = await mem.query("alpha", filter={"rank": {"$lt": 2}})
    assert [r.content for r in ranged.results] == ["alpha plain text"]
    assert (await mem.query("alpha", score_threshold=1.01)).results == []
    assert (await mem.query("   ")).results == []
    with pytest.raises(ValueError):
        await mem.add(MemoryContent(content=b"\x00\x01", mime_type=MemoryMimeType.BINARY))


async def test_names_isolate_and_clear(server):
    tenant = server.tenant()
    c1 = WarmMemory("one", warm=WarmConfig(url=server.url, tenant_id=tenant, input_dim=DIM))
    c2 = WarmMemory("two", warm=WarmConfig(url=server.url, tenant_id=tenant, input_dim=DIM))
    await c1.add(text("only in one"))
    await c2.add(text("only in two"))
    assert [r.content for r in (await c2.query("only in one")).results] == ["only in two"]
    assert len((await c1.query("only in one")).results) == 1
    await c1.clear()
    assert (await c1.query("only in one")).results == []
    assert len((await c2.query("only in two")).results) == 1


async def test_update_context_adds_system_message(server):
    mem = make(server)
    await mem.add(text("Project Bluebird launches in March"))
    ctx = BufferedChatCompletionContext(buffer_size=10)
    assert (await mem.update_context(ctx)).memories.results == []
    await ctx.add_message(UserMessage(content="When does Bluebird launch?", source="user"))
    out = await mem.update_context(ctx)
    assert len(out.memories.results) == 1
    msgs = await ctx.get_messages()
    assert "Relevant memory content" in msgs[-1].content and "Bluebird launches in March" in msgs[-1].content


async def test_assistant_agent_receives_memory_in_model_call(server):
    mem = make(server)
    await mem.add(text("The staging password rotates every Monday"))
    client = ReplayChatCompletionClient(["Every Monday."])
    agent = AssistantAgent("helper", model_client=client, memory=[mem])
    result = await agent.run(task="How often does the staging password rotate?")
    assert result.messages[-1].content == "Every Monday."
    prompt = " ".join(str(m.content) for m in client.create_calls[0]["messages"])
    assert "The staging password rotates every Monday" in prompt
    assert any(type(m).__name__ == "MemoryQueryEvent" for m in result.messages)


async def test_component_config_roundtrip(server):
    mem = make(server, "cfg", k=2, score_threshold=0.1)
    dumped = mem.dump_component()
    cfg = dumped.config
    assert cfg["name"] == "cfg" and cfg["k"] == 2 and "api_key" not in cfg
    loaded = WarmMemory.load_component(dumped)
    assert isinstance(loaded, WarmMemory) and loaded.name == "cfg"
