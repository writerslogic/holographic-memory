# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""mem0 (OSS: REAL library and stores; platform: MOCK API), Zep (MOCK API) and Letta (MOCK API).

The mocks implement the documented wire behavior and are driven by the vendors' real SDKs
(`mem0.MemoryClient`, `zep_cloud.Zep`, `letta_client.Letta`). They prove protocol handling and
mapping, not that a live service behaves identically."""

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List

import pytest

from writerslogic_warm_sdk.importers import cli as cli_mod
from writerslogic_warm_sdk.importers import ImporterError

from .conftest import API_KEY, assert_faithful, assert_resumable, vec
from .helpers_embedder import DIM, embed
from .mockserver import MockApi

EMBEDDER = "importers.helpers_embedder:embed"
os.environ.setdefault("MEM0_DIR", tempfile.mkdtemp(prefix="mem0-dir-"))  # mem0 reads this at import time
os.environ.setdefault("MEM0_TELEMETRY", "false")
pytest.importorskip("mem0")
pytest.importorskip("zep_cloud")
pytest.importorskip("letta_client")


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("WARM_IMPORT_STATE_DIR", str(tmp_path / "state"))
    # the CLI builds a parser over every registered source; limit it to the ones this module owns
    mine = ["letta", "mem0", "mem0-platform", "zep"]
    monkeypatch.setattr(cli_mod, "source_names", lambda: mine)


def query_text(server, tenant, text, flt=None, top_k=5):
    sink = server.sink(tenant)
    try:
        return sink.query(vector=embed([text])[0], top_k=top_k, filter=flt)
    finally:
        sink.close()


# =============================================================================
# mem0 platform (MOCK of https://docs.mem0.ai/api-reference)
# =============================================================================

M0_KEY = "m0-secret-key-1234567890"


def mem0_dataset() -> List[Dict[str, Any]]:
    rows = []
    for u in ("alice", "bob"):
        for i in range(9):
            rows.append({
                "id": f"{u}-{i:02d}", "memory": f"{u} likes item {i}", "user_id": u, "agent_id": None, "app_id": None,
                "session_id": None, "metadata": {"topic": f"t{i % 3}", "n": i}, "categories": ["food", "likes"],
                "created_at": f"2025-03-{i + 1:02d}T10:00:00.000000-07:00", "updated_at": None, "immutable": False,
                "expiration_date": None,
            })
    # attached to a user AND an agent: must be imported exactly once
    rows.append({"id": "both-1", "memory": "shared by alice and agent a1", "user_id": "alice", "agent_id": "a1",
                 "app_id": None, "session_id": None, "metadata": None, "categories": [], "created_at": "2025-04-01T00:00:00Z",
                 "updated_at": "2025-04-02T00:00:00Z", "immutable": True, "expiration_date": None})
    rows.append({"id": "agent-1", "memory": "agent a1 own memory", "user_id": None, "agent_id": "a1", "app_id": None,
                 "session_id": None, "metadata": {"k": "v"}, "categories": None, "created_at": 1743500000,
                 "updated_at": None, "immutable": False, "expiration_date": None})
    rows.append({"id": "run-1", "memory": "a run scoped memory", "user_id": None, "agent_id": None, "app_id": None,
                 "session_id": "run42", "metadata": None, "categories": None, "created_at": "2025-05-01T00:00:00Z",
                 "updated_at": None, "immutable": False, "expiration_date": None})
    return rows


def make_mem0_platform_api(rows):
    api = MockApi(required_auth=("authorization", f"Token {M0_KEY}"))

    @api.route("GET", r"/v1/ping/")
    def ping(req):
        return {"status": "ok", "org_id": "org_1", "project_id": "proj_1", "user_email": "dev@example.com"}

    def entities():
        ents = {}
        for r in rows:
            for key, typ in (("user_id", "user"), ("agent_id", "agent"), ("app_id", "app"), ("session_id", "run")):
                if r.get(key):
                    ents[(typ, r[key])] = {"id": f"e-{typ}-{r[key]}", "name": r[key], "type": typ,
                                           "created_at": "2025-01-01T00:00:00Z", "updated_at": "2025-01-01T00:00:00Z",
                                           "owner": "x", "metadata": None}
        return sorted(ents.values(), key=lambda e: (e["type"], e["name"]))

    @api.route("GET", r"/v1/entities/")
    def entities_route(req):
        page = int(req.q("page", "1"))
        ents = entities()
        chunk = ents[(page - 1) * 3: page * 3]  # three per page: exercises `next`
        nxt = f"{api.url}/v1/entities/?page={page + 1}" if page * 3 < len(ents) else None
        return {"count": len(ents), "next": nxt, "previous": None, "results": chunk,
                "total_users": 2, "total_agents": 1, "total_apps": 0, "total_runs": 1}

    @api.route("POST", r"/v[23]/memories/")
    def list_memories(req):
        filters = (req.body or {}).get("filters") or {}
        assert len(filters) == 1, filters
        (key, value), = filters.items()
        field = "session_id" if key == "run_id" else key
        page, size = int(req.q("page", "1")), int(req.q("page_size", "100"))
        assert 1 <= size <= 200
        hits = [r for r in rows if r.get(field) == value]
        chunk = hits[(page - 1) * size: page * size]
        more = page * size < len(hits)
        return {"count": len(hits), "next": f"{api.url}/v2/memories/?page={page + 1}" if more else None,
                "previous": None, "results": chunk}

    return api


@pytest.fixture
def mem0_api():
    with make_mem0_platform_api(mem0_dataset()) as api:
        yield api


def mem0_platform_source(api, **kw):
    from writerslogic_warm_sdk.importers.sources.mem0 import Mem0PlatformSource

    return Mem0PlatformSource(M0_KEY, api.url, None, None, kw.pop("scopes", []), kw.pop("page_size", 4))


def test_mem0_platform_mock_roundtrip(mem0_api, run, server):
    report, tenant = run(mem0_platform_source(mem0_api), embedder=EMBEDDER, batch_size=7)
    assert_faithful(report, 21, vectors=False)  # 9+9 user memories, both-1, agent-1, run-1
    mem0_api.assert_clean()
    assert report.stats.namespaces == {"alice": 10, "bob": 9, "agent:a1": 1, "run:run42": 1}
    hit = query_text(server, tenant, "shared by alice and agent a1", {"warm_source_id": "both-1"})
    assert [m["id"] for m in hit] == ["alice::both-1"]
    meta = hit[0]["metadata"]
    assert meta["agent_id"] == "a1" and meta["immutable"] is True and meta["hms_namespace"] == "alice"
    assert meta["warm_created_at"] == "2025-04-01T00:00:00+00:00" and meta["warm_updated_at"] == "2025-04-02T00:00:00+00:00"
    hit = query_text(server, tenant, "bob likes item 4", {"warm_source_id": "bob-04"})
    assert hit[0]["metadata"]["topic"] == "t1" and hit[0]["metadata"]["n"] == 4
    assert hit[0]["metadata"]["categories"] == ["food", "likes"] and hit[0]["metadata"]["kind"] == "memory"
    hit = query_text(server, tenant, "agent a1 own memory", {"warm_source_id": "agent-1"})
    assert hit[0]["id"] == "agent:a1::agent-1" and hit[0]["metadata"]["warm_created_at"].startswith("2025-04-01")


@pytest.mark.parametrize("org_id,project_id", [("other-org", "proj_1"), ("org_1", "other-project")])
def test_mem0_platform_rejects_mismatched_key_scope(mem0_api, org_id, project_id):
    import inspect
    from mem0 import MemoryClient
    from writerslogic_warm_sdk.importers.sources.mem0 import Mem0PlatformSource

    if "org_id" in inspect.signature(MemoryClient).parameters:
        pytest.skip("legacy SDK accepts explicit project selection")
    source = Mem0PlatformSource(M0_KEY, mem0_api.url, org_id, project_id, [], 4)
    with pytest.raises(ImporterError, match="API key does not match"):
        list(source.records())
    mem0_api.assert_clean()


def test_mem0_platform_matching_key_scope(mem0_api):
    from writerslogic_warm_sdk.importers.sources.mem0 import Mem0PlatformSource

    source = Mem0PlatformSource(M0_KEY, mem0_api.url, "org_1", "proj_1", [("user_id", "bob")], 4)
    try:
        assert len(list(source.records())) == 9
    finally:
        source.close()
    mem0_api.assert_clean()


def test_mem0_platform_mock_scope_filter_and_page_size_limit(mem0_api):
    src = mem0_platform_source(mem0_api, scopes=[("user_id", "bob")])
    ids = [r.id for r in src.records()]
    assert ids == [f"bob-{i:02d}" for i in range(9)]
    from writerslogic_warm_sdk.importers.sources.mem0 import Mem0PlatformSource

    with pytest.raises(ImporterError, match="1..200"):
        Mem0PlatformSource(M0_KEY, mem0_api.url, None, None, [], 500)


def test_mem0_platform_mock_resume(mem0_api, server, tmp_path):
    assert_resumable(lambda: mem0_platform_source(mem0_api), server, tmp_path, 21, embedder=EMBEDDER)


def test_mem0_platform_mock_bad_key_and_credentials_never_leak(mem0_api, server, tmp_path, capsys):
    report_path = tmp_path / "rep.json"
    env = {"WARM_API_KEY": API_KEY, "MEM0_API_KEY": "wrong-key-should-not-print"}
    code = cli_mod.main(["mem0-platform", "--host", mem0_api.url, "--warm-url", server.url, "--embedder", EMBEDDER,
                         "--report", str(report_path)], env=env)
    cap = capsys.readouterr()
    assert code == 1 and "rejected the credentials" in cap.err
    assert "wrong-key-should-not-print" not in cap.out + cap.err
    # the good key through the CLI: not in output, report or checkpoint state either
    env["MEM0_API_KEY"] = M0_KEY
    mem0_api.auth_failures = 0
    code = cli_mod.main(["mem0-platform", "--host", mem0_api.url, "--warm-url", server.url, "--embedder", EMBEDDER,
                         "--tenant", server.tenant("m0cli"), "--report", str(report_path)], env=env)
    cap = capsys.readouterr()
    assert code == 0, cap.err
    blob = cap.out + cap.err + report_path.read_text()
    for root, _, files in os.walk(tmp_path):
        blob += "".join(Path(root, f).read_text(errors="replace") for f in files)
    assert M0_KEY not in blob and "wrong-key-should-not-print" not in blob


# =============================================================================
# mem0 open-source library (REAL: mem0ai Memory + local qdrant / chroma stores)
# =============================================================================


def _fake_embedder():
    from mem0.embeddings.base import EmbeddingBase

    class Fake(EmbeddingBase):
        """Stands in for a network embedder; mem0 only calls it when adding or searching."""

        def embed(self, text, memory_action=None):
            return embed([text])[0]

    return Fake()


def populate_mem0(config: Dict[str, Any], close) -> int:
    from mem0 import Memory

    mem = Memory.from_config(config)
    mem.embedding_model = _fake_embedder()
    n = 0
    for u in ("alice", "bob"):
        for i in range(8):
            mem.add(f"{u} remembers fact {i}", user_id=u, infer=False, metadata={"topic": f"t{i % 2}", "n": i})
            n += 1
    for i in range(3):
        mem.add(f"agent a1 note {i}", agent_id="a1", infer=False)
        n += 1
    mem.add("run scoped note", run_id="run7", infer=False)
    n += 1
    close(mem)
    return n


def oss_config(tmp_path, store: str) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "embedder": {"provider": "openai", "config": {"api_key": "sk-not-real", "embedding_dims": DIM}},
        "llm": {"provider": "openai", "config": {"api_key": "sk-not-real"}},
        "history_db_path": str(tmp_path / "history.db"),
    }
    if store == "qdrant":
        cfg["vector_store"] = {"provider": "qdrant", "config": {"path": str(tmp_path / "qdrant"), "collection_name": "mem",
                                                                 "embedding_model_dims": DIM, "on_disk": True}}
    else:
        cfg["vector_store"] = {"provider": "chroma", "config": {"path": str(tmp_path / "chroma"), "collection_name": "mem"}}
    return cfg


def _close_qdrant(mem):
    mem.vector_store.client.close()
    telemetry_store = getattr(mem, "_telemetry_vector_store", None)
    if telemetry_store is not None:
        telemetry_store.client.close()


def mem0_oss_source(cfg, **kw):
    from writerslogic_warm_sdk.importers.sources.mem0 import Mem0Source

    return Mem0Source(cfg, kw.pop("scopes", []), kw.pop("with_vectors", False), kw.pop("page_size", 5), 100000)


@pytest.mark.parametrize("store", ["qdrant", "chroma"])
def test_mem0_oss_real_store_roundtrip_text_encoded(tmp_path, run, server, store):
    cfg = oss_config(tmp_path, store)
    total = populate_mem0(cfg, _close_qdrant if store == "qdrant" else (lambda m: None))
    assert total == 20
    report, tenant = run(mem0_oss_source(cfg), embedder=EMBEDDER, batch_size=6)
    assert_faithful(report, 20, vectors=False)
    assert report.stats.namespaces == {"alice": 8, "bob": 8, "agent:a1": 3, "run:run7": 1}
    hit = query_text(server, tenant, "bob remembers fact 3", {"hms_namespace": "bob"})
    assert hit[0]["text"] == "bob remembers fact 3"
    meta = hit[0]["metadata"]
    assert meta["topic"] == "t1" and meta["n"] == 3 and meta["user_id"] == "bob" and meta["hash"]
    assert meta["warm_created_at"].startswith("2026-")  # mem0 stamps the add time with an offset; normalized to UTC


def test_mem0_oss_real_qdrant_carries_the_stored_embeddings(tmp_path, run):
    cfg = oss_config(tmp_path, "qdrant")
    populate_mem0(cfg, _close_qdrant)
    report, _ = run(mem0_oss_source(cfg, with_vectors=True))
    assert_faithful(report, 20, vectors=True)  # recall against mem0's own vectors is exact
    assert report.stats.counts["vectors_carried"] == 20


def test_mem0_oss_real_qdrant_resume_mid_page(tmp_path, server):
    cfg = oss_config(tmp_path, "qdrant")
    populate_mem0(cfg, _close_qdrant)
    assert_resumable(lambda: mem0_oss_source(cfg, page_size=7), server, tmp_path, 20, batch_size=3, embedder=EMBEDDER)


def test_mem0_oss_real_scope_listing_for_other_stores(tmp_path, run):
    cfg = oss_config(tmp_path, "qdrant")
    populate_mem0(cfg, _close_qdrant)
    src = mem0_oss_source(cfg, scopes=[("user_id", "alice"), ("agent_id", "a1"), ("run_id", "run7")])
    report, _ = run(src, embedder=EMBEDDER)
    assert_faithful(report, 8 + 3 + 1, vectors=False)


@pytest.mark.parametrize("limit", [7, 8, 9])
def test_mem0_oss_scope_limit_never_truncates_silently(tmp_path, limit):
    from writerslogic_warm_sdk.importers.sources.mem0 import Mem0Source

    cfg = oss_config(tmp_path, "qdrant")
    populate_mem0(cfg, _close_qdrant)
    src = Mem0Source(cfg, [("user_id", "alice")], False, 5, limit)
    try:
        if limit <= 8:
            with pytest.raises(ImporterError, match="raise --scope-limit"):
                list(src.records())
        else:
            assert len(list(src.records())) == 8
    finally:
        src.close()


def test_mem0_oss_config_secrets_are_masked(tmp_path, capsys):
    cfg = {"vector_store": {"provider": "qdrant", "config": {"path": str(tmp_path / "x"), "api_key": "qdrant-super-secret"}},
           "llm": {"provider": "openai", "config": {"api_key": "sk-hidden-llm-key"}},
           "embedder": {"provider": "openai", "config": {"api_key": "sk-hidden-emb-key"}}}
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    from writerslogic_warm_sdk.importers.common import scrub
    from writerslogic_warm_sdk.importers.sources.mem0 import Mem0Source

    import argparse

    p = argparse.ArgumentParser()
    Mem0Source.add_arguments(p)
    src = Mem0Source.from_args(p.parse_args(["--config", str(path)]), {})
    assert "qdrant-super-secret" not in json.dumps(src.describe())
    assert scrub("boom sk-hidden-llm-key and qdrant-super-secret") == "boom *** and ***"


# =============================================================================
# Zep (MOCK of the Zep v4 API served to the real zep-cloud SDK)
# =============================================================================

ZEP_KEY = "z_secret_key_abcdef"
ZEP = "/api/v4"


def _page(items: List[Dict[str, Any]], req, key: str = "items"):
    size = int(req.q("page_size", "3"))
    start = int(req.q("cursor") or 0)
    nxt = str(start + size) if start + size < len(items) else None
    out = {key: items[start:start + size], "total_size": len(items)}
    if nxt:
        out["next_cursor"] = nxt
    return out


def zep_dataset():
    users = [
        {"uuid": "uu1", "user_id": "alice", "graph_uuid": "gu1", "created_at": "2025-01-01T00:00:00Z"},
        {"uuid": "uu2", "user_id": "bob", "graph_uuid": "gu2", "created_at": "2025-01-02T00:00:00Z"},
    ]
    threads = [
        {"uuid": "th1", "thread_id": "chat-1", "user_uuid": "uu1", "created_at": "2025-02-01T00:00:00Z"},
        {"uuid": "th2", "thread_id": "chat-2", "user_uuid": "uu2", "created_at": "2025-02-02T00:00:00Z"},
    ]
    msgs = {
        "th1": [{"uuid": f"m1-{i}", "role": "user" if i % 2 == 0 else "assistant", "name": "Alice" if i % 2 == 0 else None,
                 "content": f"alice thread message {i}", "created_at": f"2025-02-01T10:0{i}:00Z",
                 "metadata": {"seq": i} if i == 0 else None, "processed": True, "thread_uuid": "th1"} for i in range(7)],
        "th2": [{"uuid": f"m2-{i}", "role": "user", "content": f"bob thread message {i}",
                 "created_at": f"2025-02-02T10:0{i}:00Z", "thread_uuid": "th2"} for i in range(6)],
    }
    graphs = [{"uuid": "gs1", "graph_id": "kb", "created_at": "2025-01-03T00:00:00Z"},
              {"uuid": "gu1", "graph_id": "alice-graph", "user_uuid": "uu1", "created_at": "2025-01-01T00:00:00Z"}]

    def episodes(prefix, n, thread=None):
        return [{"uuid": f"{prefix}-ep{i}", "content": f"{prefix} episode text {i}", "source": "text", "source_description": "notes",
                 "created_at": f"2025-03-0{i + 1}T00:00:00Z", "processed": True, "metadata": {"i": i}} for i in range(n)]

    graph_data = {
        "gu1": {"episodes": episodes("gu1", 2) + [{"uuid": "gu1-msg", "content": "alice thread message 0", "source": "message",
                                                   "thread_uuid": "th1", "created_at": "2025-02-01T10:00:00Z"}],
                "edges": [{"uuid": f"gu1-e{i}", "name": "LIKES", "fact": f"Alice likes thing {i}", "source_node_uuid": "n1",
                           "target_node_uuid": f"n{i + 2}", "source_node_name": "Alice", "target_node_name": f"Thing{i}",
                           "created_at": "2025-03-10T00:00:00Z", "valid_at": "2025-03-01T00:00:00Z",
                           "invalid_at": "2025-06-01T00:00:00Z" if i == 0 else None, "expired_at": None,
                           "episode_uuids": ["gu1-ep0"], "attributes": {"w": i}} for i in range(3)],
                "nodes": [{"uuid": f"gu1-n{i}", "name": f"Entity{i}", "summary": f"summary of entity {i}", "labels": ["Entity"],
                           "created_at": "2025-03-11T00:00:00Z", "episode_uuids": [], "episode_uuids_truncated": False,
                           "attributes": {"a": i}} for i in range(2)]},
        "gu2": {"episodes": episodes("gu2", 1), "edges": [{"uuid": f"gu2-e{i}", "name": "KNOWS", "fact": f"Bob knows person {i}",
                                                           "created_at": "2025-03-12T00:00:00Z"} for i in range(2)],
                "nodes": [{"uuid": f"gu2-n{i}", "name": f"Person{i}", "summary": None, "labels": ["Person"],
                           "created_at": "2025-03-13T00:00:00Z", "episode_uuids": [], "episode_uuids_truncated": False}
                          for i in range(2)]},
        "gs1": {"episodes": episodes("gs1", 2), "edges": [{"uuid": f"gs1-e{i}", "name": "REL", "fact": f"kb fact {i}",
                                                           "created_at": "2025-03-14T00:00:00Z"} for i in range(2)],
                "nodes": [{"uuid": "gs1-n0", "name": "KbThing", "summary": "a thing in the kb", "labels": [],
                           "created_at": "2025-03-15T00:00:00Z", "episode_uuids": [], "episode_uuids_truncated": False}]},
    }
    return users, threads, msgs, graphs, graph_data


ZEP_EXPECTED = 13 + (2 + 3 + 2) + (1 + 2 + 2) + (2 + 2 + 1)  # messages + gu1 + gu2 + gs1 (thread episode skipped)


def make_zep_api():
    users, threads, msgs, graphs, gd = zep_dataset()
    api = MockApi(required_auth=("authorization", f"Api-Key {ZEP_KEY}"))
    api.route("POST", ZEP + r"/users/list")(lambda req: _page(users, req))
    api.route("POST", ZEP + r"/graphs/list")(lambda req: _page(graphs, req))
    api.route("GET", ZEP + r"/threads")(lambda req: _page(threads, req))
    api.route("GET", ZEP + r"/threads/(?P<t>[^/]+)/messages")(lambda req: _page(msgs[req.match["t"]], req))
    for kind in ("episodes", "edges", "nodes"):
        api.route("POST", ZEP + rf"/graphs/(?P<g>[^/]+)/{kind}/list")(
            lambda req, kind=kind: _page(gd.get(req.match["g"], {}).get(kind, []), req))
    return api


def zep_source(api, **kw):
    from writerslogic_warm_sdk.importers.sources.zep import ZepSource

    return ZepSource(ZEP_KEY, api.url + ZEP, kw.pop("kinds", ["messages", "episodes", "facts", "entities"]), 3)


def test_zep_mock_roundtrip_all_kinds(run, server):
    with make_zep_api() as api:
        report, tenant = run(zep_source(api), embedder=EMBEDDER, batch_size=8)
        api.assert_clean()
    assert_faithful(report, ZEP_EXPECTED, vectors=False)
    assert report.stats.namespaces == {"alice": 14, "bob": 11, "graph:kb": 5}
    hit = query_text(server, tenant, "Alice likes thing 0", {"warm_source_id": "fact:gu1-e0"})
    meta = hit[0]["metadata"]
    assert hit[0]["id"] == "alice::fact:gu1-e0" and meta["kind"] == "fact" and meta["relation"] == "LIKES"
    assert meta["valid_at"] == "2025-03-01T00:00:00Z" and meta["invalid_at"] == "2025-06-01T00:00:00Z"
    assert meta["attributes"] == {"w": 0} and meta["warm_created_at"] == "2025-03-10T00:00:00+00:00"
    hit = query_text(server, tenant, "alice thread message 0", {"warm_source_id": "message:m1-0"})
    assert hit[0]["metadata"]["role"] == "user" and hit[0]["metadata"]["zep_metadata"] == {"seq": 0}
    assert hit[0]["metadata"]["thread_id"] == "chat-1" and hit[0]["metadata"]["hms_namespace"] == "alice"
    hit = query_text(server, tenant, "Entity1: summary of entity 1", {"warm_source_id": "entity:gu1-n1"})
    assert hit[0]["metadata"]["labels"] == ["Entity"]
    hit = query_text(server, tenant, "kb fact 1", {"hms_namespace": "graph:kb"})
    assert hit[0]["text"] == "kb fact 1"


def test_zep_mock_thread_messages_are_not_duplicated_by_their_episodes():
    with make_zep_api() as api:
        ids = [r.id for r in zep_source(api).records()]
    assert "episode:gu1-msg" not in ids and len(ids) == len(set(ids)) == ZEP_EXPECTED


def test_zep_mock_kinds_subset_and_validation():
    with make_zep_api() as api:
        ids = [r.id for r in zep_source(api, kinds=["facts"]).records()]
    assert len(ids) == 3 + 2 + 2 and all(i.startswith("fact:") for i in ids)
    from writerslogic_warm_sdk.importers.sources.zep import ZepSource

    with pytest.raises(ImporterError, match="unknown --kinds"):
        ZepSource(ZEP_KEY, None, ["bogus"], 10)
    with pytest.raises(ImporterError, match="ZEP_API_KEY"):
        ZepSource(None, None, ["facts"], 10)


def test_zep_mock_resume(server, tmp_path):
    with make_zep_api() as api:
        assert_resumable(lambda: zep_source(api), server, tmp_path, ZEP_EXPECTED, embedder=EMBEDDER)


def test_zep_mock_api_key_never_leaks_through_the_cli(server, tmp_path, capsys):
    with make_zep_api() as api:
        report_path = tmp_path / "rep.json"
        code = cli_mod.main(["zep", "--base-url", api.url + ZEP, "--warm-url", server.url, "--embedder", EMBEDDER,
                             "--tenant", server.tenant("zcli"), "--report", str(report_path), "--page-size", "4"],
                            env={"WARM_API_KEY": API_KEY, "ZEP_API_KEY": ZEP_KEY})
        api.assert_clean()
    cap = capsys.readouterr()
    assert code == 0, cap.err
    blob = cap.out + cap.err
    for root, _, files in os.walk(tmp_path):
        blob += "".join(Path(root, f).read_text(errors="replace") for f in files)
    assert ZEP_KEY not in blob
    assert json.loads(report_path.read_text())["counts"]["docs_acked"] == ZEP_EXPECTED


def test_zep_mock_wrong_key_fails_without_echoing_it(server, capsys):
    with make_zep_api() as api:
        code = cli_mod.main(["zep", "--base-url", api.url + ZEP, "--warm-url", server.url, "--embedder", EMBEDDER],
                            env={"WARM_API_KEY": API_KEY, "ZEP_API_KEY": "wrong-zep-key-0000"})
    cap = capsys.readouterr()
    assert code == 1 and "wrong-zep-key-0000" not in cap.out + cap.err


# =============================================================================
# Letta (MOCK of the Letta REST API served to the real letta-client SDK)
# =============================================================================

LETTA_KEY = "letta-secret-key-xyz789"


def letta_dataset():
    agents = [{"id": f"agent-{i}", "name": f"bot{i}", "created_at": f"2025-01-0{i + 1}T00:00:00Z"} for i in range(3)]
    passages = {
        a["id"]: [{"id": f"passage-{a['id'][-1]}-{j:02d}", "text": f"{a['name']} archival memory {j}", "tags": ["t", f"j{j}"],
                   "metadata": {"j": j}, "archive_id": "archive-1", "created_at": f"2025-02-{j + 1:02d}T00:00:00Z",
                   "embedding": vec(100 * int(a["id"][-1]) + j) if j % 2 == 0 else None} for j in range(5 if a["id"] != "agent-2" else 0)]
        for a in agents}
    blocks = {a["id"]: [{"id": f"block-{a['id'][-1]}-{lab}", "label": lab, "value": f"{a['name']} {lab} block value",
                         "limit": 2000, "description": f"the {lab}", "read_only": lab == "persona", "metadata": {}}
                        for lab in ("human", "persona")] for a in agents}
    messages = {a["id"]: [{"id": f"message-{a['id'][-1]}-{k}", "message_type": "user_message" if k % 2 == 0 else "assistant_message",
                           "content": f"{a['name']} chat line {k}", "date": f"2025-04-01T00:0{k}:00Z"} for k in range(3)]
                for a in agents[:1]}
    messages["agent-0"].append({"id": "message-0-r", "message_type": "reasoning_message", "reasoning": "thinking about tea",
                                "date": "2025-04-01T00:05:00Z"})
    messages["agent-0"].append({"id": "message-0-x", "message_type": "tool_call_message", "date": "2025-04-01T00:06:00Z"})
    return agents, passages, blocks, messages


def _after(items, req):
    after, limit = req.q("after"), int(req.q("limit", "50"))
    if req.q("ascending") == "false" or req.q("order") == "desc":
        items = list(reversed(items))
    if after:
        ids = [i["id"] for i in items]
        items = items[ids.index(after) + 1:]
    return items[:limit]


def make_letta_api():
    agents, passages, blocks, messages = letta_dataset()
    api = MockApi(required_auth=("authorization", f"Bearer {LETTA_KEY}"))
    api.route("GET", r"/v1/agents/")(lambda req: _after(agents, req))
    api.route("GET", r"/v1/agents/(?P<a>[^/]+)")(lambda req: next(a for a in agents if a["id"] == req.match["a"]))
    api.route("GET", r"/v1/agents/(?P<a>[^/]+)/archival-memory")(lambda req: _after(passages[req.match["a"]], req))
    api.route("GET", r"/v1/agents/(?P<a>[^/]+)/core-memory/blocks")(lambda req: _after(blocks[req.match["a"]], req))
    api.route("GET", r"/v1/agents/(?P<a>[^/]+)/messages")(lambda req: _after(messages.get(req.match["a"], []), req))
    return api


def letta_source(api, **kw):
    from writerslogic_warm_sdk.importers.sources.letta import LettaSource

    return LettaSource(LETTA_KEY, api.url, kw.pop("kinds", ["passages", "blocks"]), kw.pop("agents", []),
                       kw.pop("with_vectors", False), kw.pop("page_size", 2))


LETTA_PASSAGES, LETTA_BLOCKS = 10, 6


def test_letta_mock_roundtrip_passages_and_blocks(run, server):
    with make_letta_api() as api:
        report, tenant = run(letta_source(api), embedder=EMBEDDER, batch_size=4)
        api.assert_clean()
    assert_faithful(report, LETTA_PASSAGES + LETTA_BLOCKS, vectors=False)
    assert report.stats.namespaces == {"agent-0": 7, "agent-1": 7, "agent-2": 2}
    hit = query_text(server, tenant, "bot1 archival memory 3", {"hms_namespace": "agent-1"})
    meta = hit[0]["metadata"]
    assert hit[0]["id"] == "agent-1::passage:passage-1-03" and meta["tags"] == ["t", "j3"] and meta["letta_metadata"] == {"j": 3}
    assert meta["agent_name"] == "bot1" and meta["kind"] == "archival_passage" and meta["warm_created_at"] == "2025-02-04T00:00:00+00:00"
    hit = query_text(server, tenant, "bot2 persona block value".replace("block value", "block value"), {"hms_namespace": "agent-2"})
    blk = [m for m in hit if m["metadata"]["label"] == "persona"][0]
    assert blk["metadata"]["read_only"] is True and blk["metadata"]["kind"] == "core_memory_block"


def test_letta_mock_carries_passage_embeddings_when_asked(run):
    with make_letta_api() as api:
        report, _ = run(letta_source(api, kinds=["passages"], with_vectors=True), embedder=EMBEDDER)
    # vector-bearing passages carry theirs; the rest would need encoding, which would mix spaces: rejected, not mixed
    assert report.stats.counts["vectors_carried"] == 3 + 3
    assert report.stats.rejects == {"vector_missing_in_carry_run": 4}


def test_letta_mock_messages_including_non_text(run):
    with make_letta_api() as api:
        report, _ = run(letta_source(api, kinds=["messages"], agents=["agent-0"]), embedder=EMBEDDER)
    assert report.stats.counts["docs_acked"] == 4  # 3 chat lines + reasoning; the textless tool call is skipped, listed
    assert report.stats.skips == {"empty": 1} and report.stats.skip_samples["empty"] == ["message:message-0-x"]


def test_letta_mock_resume(server, tmp_path):
    with make_letta_api() as api:
        assert_resumable(lambda: letta_source(api), server, tmp_path, LETTA_PASSAGES + LETTA_BLOCKS, embedder=EMBEDDER)


def test_letta_mock_key_never_leaks_through_the_cli(server, tmp_path, capsys):
    with make_letta_api() as api:
        report_path = tmp_path / "rep.json"
        code = cli_mod.main(["letta", "--base-url", api.url, "--warm-url", server.url, "--embedder", EMBEDDER,
                             "--tenant", server.tenant("lcli"), "--report", str(report_path)],
                            env={"WARM_API_KEY": API_KEY, "LETTA_API_KEY": LETTA_KEY})
        api.assert_clean()
        capsys.readouterr()
        bad = cli_mod.main(["letta", "--base-url", api.url, "--warm-url", server.url, "--embedder", EMBEDDER],
                           env={"WARM_API_KEY": API_KEY, "LETTA_API_KEY": "letta-wrong-key-1111"})
    cap = capsys.readouterr()
    assert code == 0 and bad == 1 and "letta-wrong-key-1111" not in cap.out + cap.err
    blob = ""
    for root, _, files in os.walk(tmp_path):
        blob += "".join(Path(root, f).read_text(errors="replace") for f in files)
    assert LETTA_KEY not in blob


def test_missing_client_libraries_are_reported_per_source(monkeypatch):
    import builtins

    from writerslogic_warm_sdk.importers.model import MissingDependency
    from writerslogic_warm_sdk.importers.sources.letta import LettaSource

    real = builtins.__import__

    def deny(name, *a, **k):
        if name.split(".")[0] == "letta_client":
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", deny)
    with pytest.raises(MissingDependency, match="letta-client"):
        list(LettaSource("k", "http://x", ["blocks"], [], False, 5).records())
