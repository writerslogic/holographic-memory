# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

litellm = pytest.importorskip("litellm")

from holographic_integrations.core import HmsConfig
from holographic_integrations.litellm import MemoryTools


def cfg(server):
    return HmsConfig(url=server.url, tenant_id=server.tenant(), input_dim=server.dim)


def tc(name, args, i=1):
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class MockSteps:
    """LiteLLM's own mock completion path, one scripted turn per call (sync and async)."""

    def __init__(self, *turns):
        self.turns, self.calls = list(turns), []

    def _kwargs(self, kwargs):
        self.calls.append(kwargs)
        turn = self.turns.pop(0)
        out = {k: v for k, v in kwargs.items()}
        out.update({"mock_tool_calls": turn} if isinstance(turn, list) else {"mock_response": turn})
        return out

    def __call__(self, **kwargs):
        return litellm.completion(**self._kwargs(kwargs))

    async def acall(self, **kwargs):
        return await litellm.acompletion(**self._kwargs(kwargs))


def test_schema_is_valid_openai_format_and_a_copy():
    schema = MemoryTools.tools_schema()
    assert [t["function"]["name"] for t in schema] == ["remember", "recall", "forget"]
    for t in schema:
        assert t["type"] == "function" and t["function"]["parameters"]["type"] == "object"
    schema[0]["function"]["name"] = "mutated"
    assert MemoryTools.tools_schema()[0]["function"]["name"] == "remember"
    # LiteLLM accepts it as tools=
    r = litellm.completion(model="gpt-4o", messages=[{"role": "user", "content": "x"}], tools=schema, mock_response="ok")
    assert r.choices[0].message.content == "ok"


def test_sync_loop_with_litellm_mock_tool_calls(server):
    tools = MemoryTools(cfg(server), user_id="u1")
    step = MockSteps([tc("remember", {"text": "The vault code word is plover"}, 1)],
                     [tc("recall", {"query": "vault code word"}, 2)],
                     "the word is plover")
    messages = [{"role": "user", "content": "what is the code word?"}]
    resp = tools.run(messages, model="gpt-4o", completion=step)
    assert resp.choices[0].message.content == "the word is plover"
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "tool", "assistant", "tool"]
    assert "plover" in messages[-1]["content"] and messages[-1]["tool_call_id"] == "call_2"
    assert messages[2]["content"] in messages[-1]["content"]  # recall found what remember stored
    assert step.calls[1]["tools"][0]["function"]["name"] == "remember"
    tools.close()


async def test_async_loop(server):
    tools = MemoryTools(cfg(server))
    step = MockSteps([tc("remember", {"text": "Otters hold hands while sleeping"}, 1)],
                     [tc("recall", {"query": "otters sleeping"}, 2)], "done")
    messages = [{"role": "user", "content": "go"}]
    resp = await tools.arun(messages, model="gpt-4o", completion=step.acall)
    assert resp.choices[0].message.content == "done" and "Otters" in messages[-1]["content"]
    await tools.aclose()


def test_default_completion_is_litellm(server):
    tools = MemoryTools(cfg(server))
    resp = tools.run([{"role": "user", "content": "hi"}], model="gpt-4o", mock_response="plain answer")
    assert resp.choices[0].message.content == "plain answer"


def test_dispatch_accepts_dicts_litellm_and_openai_objects(server):
    from litellm.types.utils import ChatCompletionMessageToolCall, Function

    tools = MemoryTools(cfg(server))
    a = tools.handle_tool_call(tc("remember", {"text": "dict form"}, 1))
    b = tools.handle_tool_call(ChatCompletionMessageToolCall(
        id="x", type="function", function=Function(name="remember", arguments=json.dumps({"text": "object form"}))))
    c = tools.handle_tool_call({"id": "y", "function": {"name": "recall", "arguments": {"query": "object form"}}})
    assert a["role"] == "tool" and a["tool_call_id"] == "call_1" and b["tool_call_id"] == "x"
    assert b["content"] in c["content"] and c["tool_call_id"] == "y"


def test_bad_calls_return_error_messages_not_exceptions(server):
    tools = MemoryTools(cfg(server))
    for call, fragment in [
        (tc("nope", {}), "unknown tool"),
        ({"id": "1", "function": {"name": "remember", "arguments": "{not json"}}, "Error"),
        (tc("remember", {"text": ""}), "non-empty"),
        (tc("recall", {}), "query"),
        (tc("recall", {"query": "x", "k": "many"}), "Error"),
        (tc("forget", {}), "memory_id"),
    ]:
        out = tools.handle_tool_call(call)
        assert out["role"] == "tool" and fragment in out["content"], (call, out)


def test_k_is_clamped_and_scoping_holds(server):
    c = cfg(server)
    mine, theirs = MemoryTools(c, user_id="a", max_k=2), MemoryTools(c, user_id="b")
    ids = [mine.handle_tool_call(tc("remember", {"text": f"note {i} about owls"}, i))["content"] for i in range(5)]
    found = mine.handle_tool_call(tc("recall", {"query": "note about owls", "k": 99}))["content"]
    assert len(found.splitlines()) == 2
    assert theirs.handle_tool_call(tc("recall", {"query": "note about owls"}))["content"] == "No matching memories."
    assert theirs.handle_tool_call(tc("forget", {"memory_id": ids[0]}))["content"] == "No such memory."
    assert mine.handle_tool_call(tc("forget", {"memory_id": ids[0]}))["content"] == "Deleted."


def test_runaway_model_is_bounded(server):
    tools = MemoryTools(cfg(server))
    step = MockSteps(*[[tc("recall", {"query": "x"}, i)] for i in range(3)])
    with pytest.raises(RuntimeError):
        tools.run([{"role": "user", "content": "go"}], model="gpt-4o", completion=step, max_steps=3)
