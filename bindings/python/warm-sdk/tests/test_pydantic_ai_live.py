"""Pydantic AI contracts with an offline model and a real WARM server."""
import uuid

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from writerslogic_warm_sdk import WarmClient, WarmError
from writerslogic_warm_sdk.pydantic_ai import WarmDependencies, get_warm_tool
from writerslogic_warm_sdk.pydantic_ai import tools
from test_live_server import API_KEY, BIN, embedding, server

pytestmark = pytest.mark.skipif(not BIN, reason="WARM_SERVER_BIN not set")


def run_tool(deps):
    steps = iter([
        ModelResponse(parts=[ToolCallPart("search_writerslogic_warm", {"query": "needle", "top_k": 1})]),
        ModelResponse(parts=[TextPart("done")]),
    ])
    agent = Agent(FunctionModel(lambda messages, info: next(steps)),
                  deps_type=WarmDependencies, tools=[get_warm_tool()])
    result = agent.run_sync("find the note", deps=deps)
    return [part.content for message in result.all_messages() if isinstance(message, ModelRequest)
            for part in message.parts if isinstance(part, ToolReturnPart)]


@pytest.mark.parametrize("outcome", ["found", "empty", "unauthorized", "embedder_error"])
def test_agent_search_preserves_scope_errors_and_closes_connections(server, monkeypatch, outcome):
    tenant = uuid.uuid4().hex
    with WarmClient(url=server, api_key=API_KEY, tenant_id=tenant) as writer:
        writer.add_documents([{"id": "note", "text": "private needle", "vector": embedding(1)}])
    clients = []

    def tracked_client(**kwargs):
        client = WarmClient(**kwargs)
        clients.append(client)
        return client

    def encode(query):
        assert query == "needle"
        if outcome == "embedder_error":
            raise ValueError("local encoder failed")
        return embedding(1)

    monkeypatch.setattr(tools, "WarmClient", tracked_client)
    deps = WarmDependencies(url=server, api_key="wrong" if outcome == "unauthorized" else API_KEY,
                            tenant_id=uuid.uuid4().hex if outcome == "empty" else tenant, embedder=encode)
    if outcome == "unauthorized":
        with pytest.raises(WarmError) as error:
            run_tool(deps)
        assert error.value.status_code == 401
    elif outcome == "embedder_error":
        with pytest.raises(ValueError, match="local encoder failed"):
            run_tool(deps)
    else:
        assert run_tool(deps) == ["private needle" if outcome == "found" else "No matches found."]
    assert len(clients) == 1
    assert clients[0].client.is_closed
