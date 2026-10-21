# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Agents SDK: persistent session plus long-term memory tools.

Needs a running hms-server (HMS_URL, default http://127.0.0.1:8080; start it with the
default --input-dim 384). A scripted model stands in for the LLM so no API key is needed:
replace `model=` with a model name to use a real one.

    uv run --extra openai-agents python examples/openai_agents_memory.py
"""

import asyncio
import json
import os
import uuid

from agents import Agent, Runner
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

from holographic_integrations.core import HmsConfig
from holographic_integrations.openai_agents import HolographicSession, memory_tools


class Script(Model):
    def __init__(self, *outputs):
        self.outputs = list(outputs)

    async def get_response(self, *args, **kwargs):
        return ModelResponse(output=self.outputs.pop(0), usage=Usage(), response_id=None)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


def say(text):
    return [ResponseOutputMessage(id="m", type="message", role="assistant", status="completed",
                                  content=[ResponseOutputText(type="output_text", text=text, annotations=[])])]


def call(name, **args):
    return [ResponseFunctionToolCall(type="function_call", call_id=uuid.uuid4().hex, name=name, arguments=json.dumps(args))]


async def main() -> None:
    config = HmsConfig(url=os.environ.get("HMS_URL", "http://127.0.0.1:8080"), tenant_id=f"demo{uuid.uuid4().hex[:8]}")
    session = HolographicSession("demo-conversation", config)
    agent = Agent(
        name="assistant",
        model=Script(call("remember", text="The staging database password rotates on Fridays"), say("Noted."),
                     call("recall", query="when does the password rotate"), say("It rotates on Fridays.")),
        tools=memory_tools(config, user_id="alice"),
    )
    print((await Runner.run(agent, "Remember the rotation schedule.", session=session)).final_output)
    print((await Runner.run(agent, "When does the password rotate?", session=session)).final_output)
    print(f"{len(await session.get_items())} items stored in the session")
    await session.aclose()


asyncio.run(main())
