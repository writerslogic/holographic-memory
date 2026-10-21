# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import json
import time
import types

import httpx
import pytest

pytest.importorskip("letta_client")

from letta_client import Letta  # noqa: E402
from letta_client.types import Passage  # noqa: E402
from letta_client.types.agents.passage_search_response import PassageSearchResponse  # noqa: E402

from holographic_integrations.core import HmsConfig  # noqa: E402
from holographic_integrations.letta import ArchivalMemory, AsyncArchivalMemory, agent_secrets, register_tools  # noqa: E402
from holographic_integrations.letta import tools as letta_tools  # noqa: E402

AGENT = "agent-11111111-1111-4111-8111-111111111111"
OTHER = "agent-22222222-2222-4222-8222-222222222222"


def cfg(server, **kw):
    return HmsConfig(url=server.url, tenant_id=server.tenant(), input_dim=server.dim, **kw)


def test_passages_resource_surface_matches_letta_client():
    """Same method names, keyword-only arguments and defaults as client.agents.passages (letta-client 1.x)."""
    from letta_client.resources.agents.passages import PassagesResource

    for name in ("create", "list", "delete", "search"):
        theirs = inspect.signature(getattr(PassagesResource, name)).parameters
        ours = inspect.signature(getattr(ArchivalMemory, name)).parameters
        wanted = {k for k, p in theirs.items() if not k.startswith("extra_") and k not in ("self", "timeout")}
        assert wanted <= set(ours), (name, wanted - set(ours))


def test_create_search_list_delete_roundtrip(server):
    with ArchivalMemory(cfg(server)) as mem:
        (p,) = mem.create(AGENT, text="The quarterly roadmap focuses on performance work", tags=["meetings", "roadmap"])
        assert isinstance(p, Passage) and p.id.startswith("passage-") and p.tags == ["meetings", "roadmap"]
        mem.create(AGENT, text="Lunch order was pasta", tags=["food"])
        res = mem.search(AGENT, query="The quarterly roadmap focuses on performance work", top_k=1)
        assert isinstance(res, PassageSearchResponse) and res.count == 1
        assert res.results[0].id == p.id and res.results[0].content.startswith("The quarterly") and res.results[0].tags == ["meetings", "roadmap"]
        assert res.results[0].timestamp.endswith("+00:00")
        assert {x.text for x in mem.list(AGENT)} == {"The quarterly roadmap focuses on performance work", "Lunch order was pasta"}
        assert mem.delete(p.id, agent_id=AGENT) == {}
        assert [x.text for x in mem.list(AGENT)] == ["Lunch order was pasta"]
        with pytest.raises(KeyError):
            mem.delete(p.id, agent_id=AGENT)
        assert mem.search(AGENT, query="   ").count == 0


def test_agents_are_isolated(server):
    with ArchivalMemory(cfg(server)) as mem:
        mem.create(AGENT, text="secret of agent one")
        assert mem.list(OTHER) == []
        assert mem.search(OTHER, query="secret of agent one").count == 0


def test_tag_modes_and_date_window(server):
    with ArchivalMemory(cfg(server)) as mem:
        mem.create(AGENT, text="alpha note", tags=["a", "b"], created_at="2024-01-10T12:00:00+00:00")
        mem.create(AGENT, text="alpha memo", tags=["a"], created_at="2024-03-10T12:00:00+00:00")
        mem.create(AGENT, text="alpha item", tags=["c"], created_at="2024-06-10T12:00:00+00:00")

        def texts(**kw):
            return sorted(r.content for r in mem.search(AGENT, query="alpha", top_k=10, **kw).results)

        assert texts(tags=["a", "c"]) == ["alpha item", "alpha memo", "alpha note"]
        assert texts(tags=["a", "b"], tag_match_mode="all") == ["alpha note"]
        assert texts(start_datetime="2024-02-01") == ["alpha item", "alpha memo"]
        assert texts(start_datetime="2024-02-01", end_datetime="2024-03-10") == ["alpha memo"]  # bare end date = end of day
        assert texts(end_datetime="2024-01-10T11:00") == []
        with pytest.raises(ValueError, match="Invalid datetime"):
            mem.search(AGENT, query="alpha", start_datetime="yesterday")
        with pytest.raises(ValueError):
            mem.search(AGENT, query="alpha", tags=["a"], tag_match_mode="some")


def test_naive_datetimes_use_the_configured_timezone(server):
    with ArchivalMemory(cfg(server), timezone="America/New_York") as mem:
        mem.create(AGENT, text="beta event", created_at="2024-05-01T03:00:00+00:00")  # 23:00 on Apr 30 in New York
        assert mem.search(AGENT, query="beta event", end_datetime="2024-04-30T23:30").count == 1
        assert mem.search(AGENT, query="beta event", end_datetime="2024-04-30T22:30").count == 0
        assert mem.search(AGENT, query="beta event").results[0].timestamp.startswith("2024-04-30T23:00:00-04:00")


def test_builtin_tool_semantics(server):
    with ArchivalMemory(cfg(server)) as mem:
        assert mem.archival_memory_insert(AGENT, "Meeting on 2024-03-15 about the API redesign", ["meetings"]) is None
        rows = mem.archival_memory_search(AGENT, "Meeting on 2024-03-15 about the API redesign")
        assert list(rows[0]) == ["id", "timestamp", "content", "tags"] and rows[0]["tags"] == ["meetings"]
        assert mem.archival_memory_search(AGENT, "") == []
        for i in range(8):
            mem.archival_memory_insert(AGENT, f"filler fact {i}")
        assert len(mem.archival_memory_search(AGENT, "filler fact")) == 5  # letta's default page size
        assert len(mem.archival_memory_search(AGENT, "filler fact", top_k=7)) == 7


@pytest.mark.parametrize("total", [9, 10, 11, 25])
def test_list_beyond_the_service_cap(server_factory, total):
    """limit-1, limit, limit+1 and several pages with max_fetch=10."""
    srv = server_factory(64, max_top_k=10)
    c = HmsConfig(url=srv.url, tenant_id=srv.tenant(), input_dim=64, max_fetch=10)
    with ArchivalMemory(c, buckets=8) as mem:
        ids = [mem.create(AGENT, text=f"passage {i}", created_at=f"2024-01-{i + 1:02d}T00:00:00+00:00")[0].id for i in range(total)]
        got = mem.list(AGENT, limit=1000)
        assert [p.id for p in got] == ids  # oldest first
        assert [p.id for p in mem.list(AGENT, ascending=False, limit=3)] == ids[::-1][:3]
        assert [p.id for p in mem.list(AGENT, after=ids[1], limit=2)] == ids[2:4]
        assert [p.id for p in mem.list(AGENT, before=ids[3], limit=100)] == ids[:3]
        assert [p.text for p in mem.list(AGENT, search="PASSAGE 1", limit=100)] == [f"passage {i}" for i in range(total) if "passage 1" in f"passage {i}"]


async def test_async_surface(server):
    async with AsyncArchivalMemory(cfg(server)) as mem:
        await mem.archival_memory_insert(AGENT, "async memory about tulips", ["flowers"])
        (p,) = await mem.list(AGENT)
        res = await mem.search(AGENT, query="async memory about tulips", tags=["flowers"])
        assert res.results[0].id == p.id
        assert (await mem.archival_memory_search(AGENT, "async memory about tulips"))[0]["content"] == "async memory about tulips"
        await mem.delete(p.id, agent_id=AGENT)
        assert await mem.list(AGENT) == []


def _function(source):
    tree = ast.parse(source)
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef))


def test_uploaded_tools_match_letta_builtin_signatures():
    ins, search = _function(letta_tools.INSERT_SOURCE), _function(letta_tools.SEARCH_SOURCE)
    assert [a.arg for a in ins.args.args] == ["agent_state", "content", "tags"]
    assert [a.arg for a in search.args.args] == ["agent_state", "query", "tags", "tag_match_mode", "top_k", "start_datetime", "end_datetime"]
    assert ast.get_docstring(ins) and "Args:" in ast.get_docstring(search)


def test_uploaded_tool_source_runs_against_the_real_service(server, monkeypatch):
    for k, v in {"HMS_URL": server.url, "HMS_TENANT_ID": server.tenant(), "HMS_INPUT_DIM": str(server.dim)}.items():
        monkeypatch.setenv(k, v)
    namespace: dict = {}
    exec(letta_tools.INSERT_SOURCE, namespace)  # noqa: S102  (our own constant, as Letta's sandbox would)
    exec(letta_tools.SEARCH_SOURCE, namespace)
    state = types.SimpleNamespace(id=AGENT)
    assert namespace["hms_archival_insert"](state, "tool made memory about volcanoes", ["geo"]) is None
    out = json.loads(namespace["hms_archival_search"](state, "tool made memory about volcanoes", tags=["geo"]))
    assert out[0]["content"] == "tool made memory about volcanoes" and out[0]["tags"] == ["geo"]
    assert json.loads(namespace["hms_archival_search"](types.SimpleNamespace(id=OTHER), "volcanoes")) == []


def test_register_tools_through_letta_client_wire():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"id": f"tool-{len(seen)}", "name": f"t{len(seen)}", "tool_type": "custom"})

    client = Letta(base_url="http://letta.invalid", api_key="x", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    tools = register_tools(client, version="0.1.0")
    assert [t.id for t in tools] == ["tool-1", "tool-2"]
    assert [m for m, _, _ in seen] == ["PUT", "PUT"] and all(p == "/v1/tools/" for _, p, _ in seen)
    for _, _, body in seen:
        assert body["pip_requirements"] == [{"name": "holographic-integrations", "version": "0.1.0"}]
        compile(body["source_code"], "<tool>", "exec")
    assert "def hms_archival_insert(" in seen[0][2]["source_code"] and "def hms_archival_search(" in seen[1][2]["source_code"]
    assert agent_secrets("http://h:8080", api_key="k", input_dim=384) == {"HMS_URL": "http://h:8080", "HMS_API_KEY": "k", "HMS_INPUT_DIM": "384"}


def test_timestamps_are_monotone_for_default_created_at(server):
    with ArchivalMemory(cfg(server)) as mem:
        a = mem.create(AGENT, text="first")[0]
        time.sleep(0.01)
        b = mem.create(AGENT, text="second")[0]
        assert a.created_at < b.created_at
        assert [p.id for p in mem.list(AGENT)] == [a.id, b.id]
