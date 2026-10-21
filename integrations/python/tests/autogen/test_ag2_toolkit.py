# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

pytest.importorskip("ag2")

from ag2 import Agent  # noqa: E402
from ag2.events import ToolCallEvent  # noqa: E402
from ag2.testing import TestConfig, TrackingConfig  # noqa: E402

from holographic_integrations.ag2 import HolographicMemoryToolkit  # noqa: E402
from holographic_integrations.core import HmsConfig  # noqa: E402

DIM = 64


def toolkit(server, **kw):
    return HolographicMemoryToolkit(HmsConfig(url=server.url, tenant_id=server.tenant(), input_dim=DIM), **kw)


def call(name, **args):
    return ToolCallEvent(name=name, arguments=json.dumps(args))


def test_tool_names_and_forget_is_optional(server):
    assert sorted(t.name for t in toolkit(server).tools) == ["forget", "recall", "remember"]
    assert sorted(t.name for t in toolkit(server, allow_forget=False).tools) == ["recall", "remember"]


async def test_agent_remembers_then_recalls_through_tools(server):
    kit = toolkit(server)
    cfg = TrackingConfig(TestConfig(
        call("remember", text="The vault code rotates on Mondays", metadata={"topic": "security"}),
        call("recall", query="when does the vault code rotate", filter={"topic": "security"}),
        "It rotates on Mondays.",
    ))
    agent = Agent("assistant", config=cfg, tools=[kit])
    reply = await agent.ask("Remember that the vault rotates on Mondays, then tell me when it rotates")
    assert await reply.content() == "It rotates on Mondays."
    stored = await kit.backend.list()
    assert [r.text for r in stored.records] == ["The vault code rotates on Mondays"]
    # the recall tool result reached the model on its last call
    assert "vault code rotates on Mondays" in str(cfg.mock.call_args_list[-1].args[0])


async def test_recall_and_forget_semantics(server):
    kit = toolkit(server)
    by_name = {t.name: t for t in kit.tools}
    assert by_name["remember"] and by_name["recall"] and by_name["forget"]
    backend = kit.backend
    from holographic_integrations.core import Record

    await backend.remember([Record("a", "alpha note", {"k": "x"}), Record("b", "alpha other", {"k": "y"})])
    cfg = TestConfig(call("recall", query="alpha", filter={"k": "y"}), call("forget", id="b"), "done")
    agent = Agent("assistant", config=cfg, tools=[kit])
    await (await agent.ask("go")).content()
    assert [r.id for r in (await backend.list()).records] == ["a"]
