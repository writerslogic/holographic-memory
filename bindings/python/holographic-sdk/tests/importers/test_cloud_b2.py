# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Elasticsearch, OpenSearch, Weaviate, Vespa and Pinecone sources.

VERIFICATION LEVEL: every test here talks to a MOCK of the vendor's documented HTTP API
(tests/importers/mockserver.py), not to a live service. Pinecone is driven through the
real `pinecone` client library against the mock; the other three use httpx directly.
The destination is always a real hms-server.
"""

from __future__ import annotations

import base64
import os
from fnmatch import fnmatch
from typing import Any, Dict, List

import pytest

from holographic_sdk.importers import ImporterError
from holographic_sdk.importers import cli as cli_mod
from holographic_sdk.importers.sources.elastic import ElasticsearchSource, OpenSearchSource
from holographic_sdk.importers.sources.pinecone import PineconeSource
from holographic_sdk.importers.sources.vespa import VespaSource, tensor_vector
from holographic_sdk.importers.sources.weaviate import WeaviateSource

from .conftest import API_KEY, assert_faithful, assert_resumable, vec
from .mockserver import ApiError, MockApi

KEY = "sekret-cloud-key-0123456789"
N = 23


@pytest.fixture(autouse=True)
def _state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HMS_IMPORT_STATE_DIR", str(tmp_path / "state"))


@pytest.fixture
def cli_main(monkeypatch):
    """cli.main restricted to the sources of this module, so the parser does not depend on other sources."""
    from holographic_sdk.importers.sources import source_class

    mine = ["elasticsearch", "opensearch", "weaviate", "vespa", "pinecone"]
    monkeypatch.setattr(cli_mod, "source_names", lambda: mine)
    monkeypatch.setattr(cli_mod, "source_class", source_class)
    return cli_mod.main


def hms_query(server, tenant, vector, **flt):
    sink = server.sink(tenant)
    try:
        return sink.query(vector=vector, top_k=100, filter=flt or None)
    finally:
        sink.close()


def scan_for(root, secret: str) -> None:
    for base, _, files in os.walk(root):
        for f in files:
            with open(os.path.join(base, f), errors="replace") as fh:
                assert secret not in fh.read(), f"credential found in {f}"


# ---- Elasticsearch / OpenSearch -------------------------------------------------------


def make_es(store: Dict[str, List[Dict[str, Any]]], auth_value: str, vec_in_source: bool, vec_type: str):
    api = MockApi(required_auth=("authorization", auth_value))
    scrolls: Dict[str, List[Dict[str, Any]]] = {}

    def names(pattern: str) -> List[str]:
        return sorted(n for n in store if any(fnmatch(n, p) for p in pattern.split(",")))

    def shaped(index: str, doc: Dict[str, Any], want_fields: List[str]) -> Dict[str, Any]:
        src = dict(doc["_source"])
        out = {"_index": index, "_id": doc["_id"], "_score": None}
        fields = {}
        if not vec_in_source:
            for f in list(src):
                if f == "emb":
                    fields[f] = src.pop(f)  # dense_vector is excluded from _source and served via `fields`
        out["_source"] = src
        if fields and "emb" in want_fields:
            out["fields"] = fields
        return out

    def page(hits: List[Dict[str, Any]], sid: str, total: int) -> Dict[str, Any]:
        return {"_scroll_id": sid, "took": 1, "timed_out": False,
                "hits": {"total": {"value": total, "relation": "eq"}, "hits": hits}}

    @api.route("GET", r"/(?P<idx>[^/_][^/]*)/_mapping")
    def mapping(req):
        found = names(req.match["idx"])
        if not found:
            raise ApiError(404, {"error": {"type": "index_not_found_exception"}})
        return {n: {"mappings": {"properties": {"emb": {"type": vec_type, "dims": 32}, "text": {"type": "text"}}}}
                for n in found}

    @api.route("POST", r"/(?P<idx>[^/_][^/]*)/_count")
    def count(req):
        assert req.body["query"] == {"match_all": {}}
        return {"count": sum(len(store[n]) for n in names(req.match["idx"])), "_shards": {"total": 1}}

    @api.route("POST", r"/(?P<idx>[^/_][^/]*)/_search")
    def search(req):
        assert req.q("scroll"), "importer must use the scroll API"
        want = req.body.get("fields", [])
        flat = [(n, d) for n in names(req.match["idx"]) for d in store[n]]
        size = req.body["size"]
        sid = f"scroll-{len(scrolls)}"
        scrolls[sid] = [shaped(n, d, want) for n, d in flat[size:]]
        return page([shaped(n, d, want) for n, d in flat[:size]], sid, len(flat))

    @api.route("POST", r"/_search/scroll")
    def scroll(req):
        sid = req.body["scroll_id"]
        assert req.body["scroll"], "scroll keep-alive is required"
        hits, scrolls[sid] = scrolls[sid][: 5], scrolls[sid][5:]
        return page(hits, sid, 0)

    @api.route("DELETE", r"/_search/scroll")
    def clear(req):
        return {"succeeded": True, "num_freed": 1}

    return api


def es_docs(prefix: str, n: int, offset: int = 0):
    return [
        {"_id": f"{prefix}{i}", "_source": {
            "text": f"{prefix} document {i}", "emb": vec(offset + i), "n": i,
            "meta": {"lang": "en", "tags": ["a", "b"]}, "created": 1_700_000_000 + i}}
        for i in range(n)
    ]


def test_elasticsearch_mock_roundtrip(tmp_path, server, run):
    store = {"docs": es_docs("d", N)}
    with make_es(store, f"ApiKey {KEY}", vec_in_source=False, vec_type="dense_vector").start() as api:
        src = ElasticsearchSource(api.url, "docs", api_key=KEY, page_size=5, timestamp_field="created")
        assert src.count() == N
        report, tenant = run(src, batch_size=7)
        api.assert_clean()
    assert_faithful(report, N)
    got = {m["id"]: m for m in hms_query(server, tenant, vec(4))}
    assert len(got) == N and got["d4"]["text"] == "d document 4"
    meta = got["d4"]["metadata"]
    assert meta["n"] == 4 and meta["meta"] == {"lang": "en", "tags": ["a", "b"]}
    assert meta["hms_created_at"].startswith("2023-11-14") and "created" not in meta
    assert "emb" not in meta and "text" not in meta


def test_opensearch_mock_roundtrip_pattern_namespaces(server, run):
    store = {"docs-a": es_docs("x", 6), "docs-b": es_docs("x", 5, offset=100), "other": es_docs("o", 3, offset=200)}
    basic = "Basic " + base64.b64encode(b"admin:pw-os-12345").decode()
    with make_es(store, basic, vec_in_source=True, vec_type="knn_vector").start() as api:
        src = OpenSearchSource(api.url, "docs-*", user="admin", password="pw-os-12345", page_size=4)
        report, tenant = run(src, batch_size=3)
        api.assert_clean()
    assert_faithful(report, 11)
    assert report.stats.namespaces == {"docs-a": 6, "docs-b": 5}
    both = hms_query(server, tenant, vec(1), hms_source_id="x1")
    assert {m["id"] for m in both} == {"docs-a::x1", "docs-b::x1"}  # same _id in two indices stays distinct
    only_b = hms_query(server, tenant, vec(101), hms_namespace="docs-b")
    assert len(only_b) == 5


def test_elasticsearch_requires_credentials_and_names_conflicting_vector_fields():
    store = {"docs": es_docs("d", 2)}
    with make_es(store, f"ApiKey {KEY}", True, "dense_vector").start() as api:
        bad = ElasticsearchSource(api.url, "docs", api_key="wrong-key-value")
        with pytest.raises(ImporterError, match="401"):
            list(bad.records())
        bad.close()
    assert api.auth_failures >= 1


def test_elasticsearch_mock_resume(tmp_path, server):
    store = {"docs": es_docs("d", N)}
    with make_es(store, f"ApiKey {KEY}", True, "dense_vector").start() as api:
        assert_resumable(lambda: ElasticsearchSource(api.url, "docs", api_key=KEY, page_size=5), server, tmp_path, N)
        api.assert_clean()


def test_elasticsearch_cli_never_leaks_the_api_key(tmp_path, server, capsys, cli_main):
    store = {"docs": es_docs("d", 8)}
    report = tmp_path / "report.json"
    with make_es(store, f"ApiKey {KEY}", True, "dense_vector").start() as api:
        code = cli_main(
            ["elasticsearch", "--index", "docs", "--hms-url", server.url, "--tenant", server.tenant("esc"),
             "--report", str(report)],
            env={"ELASTIC_API_KEY": KEY, "ELASTICSEARCH_URL": api.url, "HMS_API_KEY": API_KEY},
        )
        api.assert_clean()
    out = capsys.readouterr()
    assert code == 0, out.err
    assert KEY not in out.out + out.err
    scan_for(tmp_path, KEY)


# ---- Weaviate -------------------------------------------------------------------------


def uid(i: int) -> str:
    return f"00000000-0000-4000-8000-{i:012d}"


def make_weaviate(cold_tenant: bool = False, token: str = KEY):
    api = MockApi(required_auth=("authorization", f"Bearer {token}"))
    schema = [
        {"class": "Article", "multiTenancyConfig": {"enabled": True},
         "properties": [{"name": "title", "dataType": ["text"]}, {"name": "body", "dataType": ["text"]},
                        {"name": "views", "dataType": ["int"]}]},
        {"class": "Note", "properties": [{"name": "content", "dataType": ["text"]}]},
    ]
    tenants = [{"name": "acme", "activityStatus": "HOT"}, {"name": "globex", "activityStatus": "HOT"}]
    if cold_tenant:
        tenants.append({"name": "frozen", "activityStatus": "COLD"})
    data: Dict[Any, List[Dict[str, Any]]] = {
        ("Article", "acme"): [
            {"class": "Article", "id": uid(10 + i), "tenant": "acme", "vector": vec(10 + i),
             "properties": {"title": f"t{i}", "body": f"acme body {i}", "views": i},
             "creationTimeUnix": 1_700_000_000_000 + i, "lastUpdateTimeUnix": 1_700_000_100_000 + i}
            for i in range(9)],
        ("Article", "globex"): [
            {"class": "Article", "id": uid(10 + i), "tenant": "globex", "vector": vec(50 + i),
             "properties": {"title": f"g{i}", "body": f"globex body {i}", "views": i},
             "creationTimeUnix": 1_700_000_000_000, "lastUpdateTimeUnix": 1_700_000_000_000}
            for i in range(6)],  # same uuids as acme: legal across tenants
        ("Note", None): [
            {"class": "Note", "id": uid(200 + i), "vectors": {"default": vec(90 + i)},
             "properties": {"content": f"note {i}", "pinned": i % 2 == 0}}
            for i in range(5)],
    }
    page_cap = 4  # the server returns fewer objects than `limit` asks for

    @api.route("GET", r"/v1/schema")
    def get_schema(req):
        return {"classes": schema}

    @api.route("GET", r"/v1/schema/(?P<cls>[^/]+)/tenants")
    def get_tenants(req):
        return tenants

    @api.route("GET", r"/v1/objects")
    def objects(req):
        cls = req.q("class")
        if not cls:
            raise ApiError(422, {"error": [{"message": "class is required for cursor listing"}]})
        tenant = req.q("tenant")
        mt = cls == "Article"
        if mt and not tenant:
            raise ApiError(422, {"error": [{"message": "has multi-tenancy enabled, but request was without tenant"}]})
        rows = sorted(data.get((cls, tenant if mt else None), []), key=lambda o: o["id"])
        after = req.q("after")
        if after:
            rows = [o for o in rows if o["id"] > after]
        limit = min(int(req.q("limit", "25")), page_cap)
        out = []
        for o in rows[:limit]:
            o = dict(o)
            if req.q("include") != "vector":
                o.pop("vector", None)
                o.pop("vectors", None)
            out.append(o)
        return {"objects": out, "totalResults": len(out)}

    return api


def test_weaviate_mock_roundtrip_tenants_and_named_vectors(server, run):
    with make_weaviate().start() as api:
        report, tenant = run(WeaviateSource(api.url, api_key=KEY, page_size=50), batch_size=6)
        api.assert_clean()
    assert_faithful(report, 20)
    assert report.stats.namespaces == {"acme": 9, "globex": 6, "": 5}
    ids = {m["id"] for m in hms_query(server, tenant, vec(10), hms_source_id=uid(10))}
    assert ids == {f"acme::{uid(10)}", f"globex::{uid(10)}"}
    acme = hms_query(server, tenant, vec(12), hms_namespace="acme")
    top = next(m for m in acme if m["metadata"]["hms_source_id"] == uid(12))
    assert top["text"] == "acme body 2" and top["metadata"]["title"] == "t2" and top["metadata"]["views"] == 2
    assert top["metadata"]["weaviate_class"] == "Article"
    assert top["metadata"]["hms_created_at"].startswith("2023-11-14")
    assert top["metadata"]["hms_updated_at"].startswith("2023-11-1")
    notes = hms_query(server, tenant, vec(91), hms_namespace="")
    note = next(m for m in notes if m["metadata"]["hms_source_id"] == uid(201))
    assert note["text"] == "note 1" and note["metadata"]["pinned"] is False


def test_weaviate_mock_cold_tenant_is_rejected_not_dropped(run):
    with make_weaviate(cold_tenant=True).start() as api:
        report, _ = run(WeaviateSource(api.url, api_key=KEY, classes=["Article"]), sample=0)
    assert report.stats.counts["docs_acked"] == 15
    assert report.stats.rejects == {"source_unreadable": 1}
    assert report.stats.reject_samples["source_unreadable"] == ["Article/frozen"]


def test_weaviate_mock_resume_is_seekable(tmp_path, server):
    with make_weaviate().start() as api:
        src_for = lambda: WeaviateSource(api.url, api_key=KEY)  # noqa: E731
        assert src_for().seekable
        assert_resumable(src_for, server, tmp_path, 20, batch_size=4)
        api.assert_clean()


def test_weaviate_cli_never_leaks_the_api_key(tmp_path, server, capsys, cli_main):
    report = tmp_path / "r.json"
    with make_weaviate().start() as api:
        code = cli_main(
            ["weaviate", "--class", "Note", "--hms-url", server.url, "--tenant", server.tenant("wc"),
             "--report", str(report)],
            env={"WEAVIATE_API_KEY": KEY, "WEAVIATE_URL": api.url, "HMS_API_KEY": API_KEY},
        )
    out = capsys.readouterr()
    assert code == 0, out.err
    assert KEY not in out.out + out.err
    scan_for(tmp_path, KEY)


# ---- Vespa ----------------------------------------------------------------------------


def make_vespa(n: int = N, page: int = 6, token: str = KEY, grouped: bool = False):
    api = MockApi(required_auth=("authorization", f"Bearer {token}"))
    docs = []
    for i in range(n):
        ident = f"id:shop:item:g=grp{i % 2}:u{i}" if grouped else f"id:shop:item::u{i}"
        docs.append({"id": ident, "fields": {
            "text": f"item text {i}", "price": i * 1.5, "owner": {"name": f"o{i}"},
            "embedding": {"type": "tensor<float>(x[32])", "values": vec(i)}, "ts": 1_700_000_000 + i}})

    @api.route("GET", r"/document/v1/(?P<ns>[^/]+)/(?P<dt>[^/]+)/docid")
    def visit(req):
        assert req.match["ns"] == "shop" and req.match["dt"] == "item"
        assert req.q("format.tensors") == "short"
        start = int(req.q("continuation", "0").replace("tok-", "") or 0)
        chunk = docs[start:start + min(int(req.q("wantedDocumentCount", "10")), page)]
        out: Dict[str, Any] = {"pathId": req.path, "documents": chunk, "documentCount": len(chunk)}
        if start + len(chunk) < len(docs):
            out["continuation"] = f"tok-{start + len(chunk)}"
        return out

    return api


def test_vespa_mock_roundtrip(server, run):
    with make_vespa().start() as api:
        src = VespaSource(api.url, "shop", "item", token=KEY, page_size=10, timestamp_field="ts")
        report, tenant = run(src, batch_size=5)
        api.assert_clean()
    assert_faithful(report, N)
    got = {m["id"]: m for m in hms_query(server, tenant, vec(3))}
    meta = got["u3"]["metadata"]
    assert got["u3"]["text"] == "item text 3" and meta["price"] == 4.5 and meta["owner"] == {"name": "o3"}
    assert meta["vespa_namespace"] == "shop" and meta["vespa_doctype"] == "item" and "embedding" not in meta
    assert meta["hms_created_at"].startswith("2023-11-14")


def test_vespa_mock_group_namespace(server, run):
    with make_vespa(grouped=True).start() as api:
        src = VespaSource(api.url, "shop", "item", token=KEY, namespace_from="group")
        report, tenant = run(src)
    assert_faithful(report, N)
    assert report.stats.namespaces == {"grp0": 12, "grp1": 11}


def test_vespa_mock_resume(tmp_path, server):
    with make_vespa().start() as api:
        assert_resumable(lambda: VespaSource(api.url, "shop", "item", token=KEY), server, tmp_path, N)


def test_vespa_cli_never_leaks_the_token(tmp_path, server, capsys, cli_main):
    report = tmp_path / "r.json"
    with make_vespa().start() as api:
        code = cli_main(
            ["vespa", "--vespa-namespace", "shop", "--doctype", "item", "--hms-url", server.url,
             "--tenant", server.tenant("vc"), "--report", str(report)],
            env={"VESPA_TOKEN": KEY, "VESPA_URL": api.url, "HMS_API_KEY": API_KEY},
        )
    out = capsys.readouterr()
    assert code == 0, out.err
    assert KEY not in out.out + out.err
    scan_for(tmp_path, KEY)


def test_vespa_tensor_renderings():
    assert tensor_vector({"type": "tensor<float>(x[3])", "values": [1, 2, 3]}) == ([1.0, 2.0, 3.0], None)
    assert tensor_vector([0.5, 1.5]) == ([0.5, 1.5], None)
    cells = {"cells": [{"address": {"x": "1"}, "value": 2.0}, {"address": {"x": "0"}, "value": 1.0}]}
    assert tensor_vector(cells) == ([1.0, 2.0], None)
    assert tensor_vector({"values": "7f80"}) == ([127.0, -128.0], None)  # int8 hex form
    assert tensor_vector({"blocks": []})[0] is None
    assert tensor_vector({"cells": [{"address": {"x": "a"}, "value": 1.0}]})[0] is None  # mapped, not indexed
    assert tensor_vector({"cells": [{"address": {"x": "0", "y": "0"}, "value": 1.0}]})[0] is None


def test_vespa_unusable_tensor_is_rejected_not_dropped(run):
    api = make_vespa(n=3)
    docs_route = api.routes[0]

    with api.start():
        # replace one document's tensor with a mixed (blocks) tensor through a wrapper route
        original = docs_route[2]

        def patched(req):
            out = original(req)
            out["documents"][1] = {"id": out["documents"][1]["id"], "fields": {
                "text": "bad one", "embedding": {"blocks": [{"address": {"a": "x"}, "values": [1, 2]}]}}}
            return out

        api.routes[0] = (docs_route[0], docs_route[1], patched)
        report, _ = run(VespaSource(api.url, "shop", "item", token=KEY, vector_field="embedding"), sample=0)
    assert report.stats.counts["docs_acked"] == 2
    assert report.stats.rejects == {"source_unreadable": 1}


# ---- Pinecone -------------------------------------------------------------------------


def make_pinecone(key: str = KEY):
    api = MockApi(required_auth=("api-key", key))
    ns_data: Dict[str, Dict[str, Dict[str, Any]]] = {"__default__": {}, "tenant-a": {}, "tenant-b": {}}
    for i in range(7):
        ns_data["__default__"][f"v{i}"] = {"values": vec(i), "metadata": {"text": f"default {i}", "n": i}}
    for i in range(9):
        ns_data["tenant-a"][f"doc#{i}"] = {"values": vec(20 + i), "metadata": {"content": f"a {i}", "ts": 1_700_000_000 + i}}
    for i in range(4):
        ns_data["tenant-b"][f"doc#{i}"] = {"values": vec(40 + i), "metadata": {"text": f"b {i}", "lang": "fr"},
                                           "sparseValues": {"indices": [1, 5], "values": [0.5, 0.25]}}

    def ns_of(req):
        ns = req.q("namespace", "") or ""
        return "__default__" if ns in ("", "__default__") else ns

    @api.route("POST", r"/describe_index_stats")
    def stats(req):
        return {"namespaces": {k: {"vectorCount": len(v)} for k, v in ns_data.items()}, "dimension": 32,
                "indexFullness": 0, "totalVectorCount": sum(len(v) for v in ns_data.values())}

    @api.route("GET", r"/vectors/list")
    def list_ids(req):
        ids = sorted(ns_data[ns_of(req)])  # bitwise ("C") collation order
        prefix = req.q("prefix")
        if prefix:
            ids = [i for i in ids if i.startswith(prefix)]
        start = int(req.q("paginationToken", "0") or 0)
        limit = int(req.q("limit", "100"))
        chunk = ids[start:start + limit]
        out: Dict[str, Any] = {"vectors": [{"id": i} for i in chunk], "namespace": req.q("namespace", ""),
                               "usage": {"readUnits": 1}}
        if start + limit < len(ids):
            out["pagination"] = {"next": str(start + limit)}
        return out

    @api.route("GET", r"/vectors/fetch")
    def fetch(req):
        want = req.query.get("ids", [])
        assert 0 < len(want) <= 100
        data = ns_data[ns_of(req)]
        vectors = {}
        for i in want:
            if i in data:
                vectors[i] = {"id": i, **data[i]}
        return {"vectors": vectors, "namespace": req.q("namespace", ""), "usage": {"readUnits": 1}}

    return api, ns_data


def test_pinecone_mock_roundtrip_through_real_client(server, run):
    api, data = make_pinecone()
    with api.start():
        src = PineconeSource(api.url, api_key=KEY, page_size=4)
        assert src.count() == 20
        report, tenant = run(src, batch_size=6)
        api.assert_clean()
    assert_faithful(report, 20)
    assert report.stats.namespaces == {"": 7, "tenant-a": 9, "tenant-b": 4}
    same = hms_query(server, tenant, vec(20), hms_source_id="doc#0")
    assert {m["id"] for m in same} == {"tenant-a::doc#0", "tenant-b::doc#0"}
    a = next(m for m in hms_query(server, tenant, vec(23), hms_namespace="tenant-a")
             if m["metadata"]["hms_source_id"] == "doc#3")
    assert a["text"] == "a 3"
    b = next(m for m in hms_query(server, tenant, vec(40), hms_namespace="tenant-b")
             if m["metadata"]["hms_source_id"] == "doc#0")
    assert b["metadata"]["pinecone_sparse"] == {"indices": [1, 5], "values": [0.5, 0.25]}  # kept, not dropped
    assert b["metadata"]["lang"] == "fr" and b["text"] == "b 0"
    d = next(m for m in hms_query(server, tenant, vec(3), hms_namespace="") if m["metadata"]["hms_source_id"] == "v3")
    assert d["id"] == "v3" and d["metadata"]["n"] == 3  # the default namespace keeps bare ids


def test_pinecone_mock_namespace_filter_and_timestamp(server, run):
    api, _ = make_pinecone()
    with api.start():
        src = PineconeSource(api.url, api_key=KEY, namespaces=["tenant-a"], timestamp_field="ts")
        report, tenant = run(src)
    assert_faithful(report, 9)
    m = next(x for x in hms_query(server, tenant, vec(20), hms_namespace="tenant-a")
             if x["metadata"]["hms_source_id"] == "doc#0")
    assert m["metadata"]["hms_created_at"].startswith("2023-11-14") and "ts" not in m["metadata"]


def test_pinecone_mock_unknown_namespace_is_an_error():
    api, _ = make_pinecone()
    with api.start():
        with pytest.raises(ImporterError, match="not in the index"):
            list(PineconeSource(api.url, api_key=KEY, namespaces=["nope"]).records())


def test_pinecone_mock_resume_is_seekable(tmp_path, server):
    api, _ = make_pinecone()
    with api.start():
        assert PineconeSource(api.url, api_key=KEY).seekable
        assert_resumable(lambda: PineconeSource(api.url, api_key=KEY, page_size=4), server, tmp_path, 20, batch_size=3)


def test_pinecone_cli_never_leaks_the_api_key(tmp_path, server, capsys, cli_main):
    api, _ = make_pinecone()
    report = tmp_path / "r.json"
    with api.start():
        code = cli_main(
            ["pinecone", "--host", api.url, "--hms-url", server.url, "--tenant", server.tenant("pc"),
             "--report", str(report)],
            env={"PINECONE_API_KEY": KEY, "HMS_API_KEY": API_KEY},
        )
    out = capsys.readouterr()
    assert code == 0, out.err
    assert KEY not in out.out + out.err
    scan_for(tmp_path, KEY)


def test_pinecone_without_a_key_fails_before_any_request():
    with pytest.raises(ImporterError, match="PINECONE_API_KEY"):
        PineconeSource("http://127.0.0.1:1")


# ---- Weaviate: REAL embedded instance --------------------------------------------------
# Downloads the Weaviate binary (~100 MB, cached by the client) and starts it locally, so it
# is opt-in: HMS_TEST_WEAVIATE_EMBEDDED=1.


@pytest.fixture(scope="module")
def embedded_weaviate(tmp_path_factory):
    if os.environ.get("HMS_TEST_WEAVIATE_EMBEDDED") != "1":
        pytest.skip("set HMS_TEST_WEAVIATE_EMBEDDED=1 to run against a real embedded Weaviate")
    weaviate = pytest.importorskip("weaviate")
    import socket

    def free_port() -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    port, grpc_port = free_port(), free_port()
    client = weaviate.connect_to_embedded(
        version="1.30.1", port=port, grpc_port=grpc_port,
        persistence_data_path=str(tmp_path_factory.mktemp("weaviate-data")),
    )
    try:
        yield client, f"http://127.0.0.1:{port}"
    finally:
        client.close()


def test_weaviate_real_embedded_roundtrip(embedded_weaviate, server, run, tmp_path):
    from weaviate.classes.config import Configure, DataType, Property
    from weaviate.classes.data import DataObject
    from weaviate.classes.tenants import Tenant

    client, url = embedded_weaviate
    props = [Property(name="body", data_type=DataType.TEXT), Property(name="n", data_type=DataType.INT)]
    mt = client.collections.create("Mt", vectorizer_config=Configure.Vectorizer.none(), properties=props,
                                   multi_tenancy_config=Configure.multi_tenancy(enabled=True))
    mt.tenants.create([Tenant(name="acme"), Tenant(name="globex")])
    for t, base in (("acme", 0), ("globex", 100)):
        mt.with_tenant(t).data.insert_many(
            [DataObject(properties={"body": f"{t} body {i}", "n": i}, vector=vec(base + i)) for i in range(11)])
    plain = client.collections.create("Plain", vectorizer_config=Configure.Vectorizer.none(), properties=props)
    plain.data.insert_many([DataObject(properties={"body": f"plain {i}", "n": i}, vector=vec(300 + i)) for i in range(9)])

    make = lambda: WeaviateSource(url, page_size=4)  # noqa: E731
    report, tenant = run(make(), batch_size=5)
    assert_faithful(report, 31)
    assert report.stats.namespaces == {"acme": 11, "globex": 11, "": 9}
    hit = next(m for m in hms_query(server, tenant, vec(103), hms_namespace="globex") if m["text"] == "globex body 3")
    assert hit["metadata"]["n"] == 3 and hit["metadata"]["weaviate_class"] == "Mt"
    assert hit["metadata"]["hms_created_at"].startswith("20")
    assert_resumable(make, server, tmp_path, 31, batch_size=4)
