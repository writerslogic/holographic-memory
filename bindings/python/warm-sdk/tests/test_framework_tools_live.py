"""Native framework tool calls against the local WARM service."""
import uuid

import pytest

pytest.importorskip("crewai")
pytest.importorskip("smolagents")

from writerslogic_warm_sdk import WarmClient, WarmError
from writerslogic_warm_sdk.crewai import tools as crew_tools
from writerslogic_warm_sdk.smolagents import tools as smol_tools
from test_live_server import API_KEY, BIN, embedding, server

pytestmark = pytest.mark.skipif(not BIN, reason="WARM_SERVER_BIN not set")


@pytest.mark.parametrize("framework", ["crewai", "smolagents"])
@pytest.mark.parametrize("outcome", ["found", "empty", "unauthorized", "embedder_error"])
def test_native_tool_calls_close_clients(server, monkeypatch, framework, outcome):
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

    module = crew_tools if framework == "crewai" else smol_tools
    monkeypatch.setattr(module, "WarmClient", tracked_client)
    tool = module.WarmSearchTool(url=server, api_key="wrong" if outcome == "unauthorized" else API_KEY,
                                tenant_id=uuid.uuid4().hex if outcome == "empty" else tenant, embedder=encode)

    def invoke():
        if framework == "crewai":
            return tool.run(query="needle")
        with tool:
            return tool("needle")

    if outcome == "unauthorized":
        with pytest.raises(WarmError) as error:
            invoke()
        assert error.value.status_code == 401
    elif outcome == "embedder_error":
        with pytest.raises(ValueError, match="local encoder failed"):
            invoke()
    else:
        assert invoke() == ("private needle" if outcome == "found" else "No matches found.")
    assert len(clients) == 1
    assert clients[0].client.is_closed


@pytest.mark.parametrize("kwargs, count", [({}, 3), ({"top_k": None}, 3), ({"top_k": 0}, None), ({"top_k": 1}, 1), ({"top_k": 2}, 2)])
def test_smolagents_optional_top_k_matches_its_native_schema(server, kwargs, count):
    tenant = uuid.uuid4().hex
    with WarmClient(url=server, api_key=API_KEY, tenant_id=tenant) as writer:
        writer.add_documents([{"id": str(i), "text": f"note {i}", "vector": embedding(1)} for i in range(4)])
    with smol_tools.WarmSearchTool(url=server, api_key=API_KEY, tenant_id=tenant,
                                  embedder=lambda query: embedding(1)) as tool:
        if count is None:
            with pytest.raises(WarmError) as error:
                tool("needle", **kwargs)
            assert error.value.status_code == 422
        else:
            assert len(tool("needle", **kwargs).split("\n\n---\n\n")) == count
    tool.close()
