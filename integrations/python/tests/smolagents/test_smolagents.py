# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

pytest.importorskip("smolagents")

from smolagents import CodeAgent, Model, ToolCallingAgent
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction

from holographic_integrations.core import HmsConfig
from holographic_integrations.smolagents import memory_tools


def cfg(server):
    return HmsConfig(url=server.url, tenant_id=server.tenant(), input_dim=server.dim)


class Scripted(Model):
    """Plays a fixed list of assistant messages and records what it was asked."""

    def __init__(self, script):
        super().__init__()
        self.script, self.seen = list(script), []

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        self.seen.append(messages)
        return self.script.pop(0)


def call(name, args, i):
    return ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[ChatMessageToolCall(id=f"c{i}", type="function",
                                        function=ChatMessageToolCallFunction(name=name, arguments=args))],
    )


def code(body):
    return ChatMessage(role="assistant", content=f"Thought: go\n<code>\n{body}\n</code>")


def test_tool_schema_matches_smolagents_contract(server):
    remember, recall, forget = memory_tools(cfg(server))
    assert [t.name for t in (remember, recall, forget)] == ["remember", "recall", "forget"]
    assert remember.inputs["tags"]["nullable"] is True and recall.output_type == "string"
    # smolagents' own prompt-facing rendering works
    assert "remember" in remember.to_code_prompt() and "query" in recall.to_code_prompt()


def test_tool_calling_agent_remember_then_recall(server):
    tools = memory_tools(cfg(server))
    model = Scripted([
        call("remember", {"text": "The vault code word is plover", "tags": ["secret"]}, 1),
        call("recall", {"query": "vault code word"}, 2),
        call("final_answer", {"answer": "done"}, 3),
    ])
    agent = ToolCallingAgent(tools=tools, model=model, add_base_tools=False)
    assert agent.run("remember then recall") == "done"
    observations = [m for m in model.seen[-1] if "plover" in json.dumps(m.content, default=str)]
    assert observations, "the recall observation must reach the model"


def test_code_agent_uses_tools_as_python_functions(server):
    tools = memory_tools(cfg(server))
    model = Scripted([
        code('mid = remember(text="Badgers dig setts under hedgerows")\nprint(mid)'),
        code('found = recall(query="badgers dig setts")\nfinal_answer(found)'),
    ])
    agent = CodeAgent(tools=tools, model=model, add_base_tools=False)
    answer = agent.run("go")
    assert "Badgers dig setts" in answer


def test_user_scoping_and_forget(server):
    c = cfg(server)
    a_remember, a_recall, a_forget = memory_tools(c, user_id="a")
    _, b_recall, b_forget = memory_tools(c, user_id="b")
    mid = a_remember(text="private note about otters")
    assert mid in a_recall(query="private note about otters")
    assert b_recall(query="private note about otters") == "No matching memories."
    assert b_forget(memory_id=mid) == "No such memory."
    assert a_forget(memory_id=mid) == "Deleted."
    assert a_recall(query="private note about otters") == "No matching memories."


def test_argument_validation_by_the_framework(server):
    _, recall, _ = memory_tools(cfg(server))
    with pytest.raises(Exception):
        recall(nonsense="x")
