# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""OpenAI vector stores via the real `openai` SDK against a MOCK of the documented API.

Docs: https://platform.openai.com/docs/api-reference/vector-stores,
https://platform.openai.com/docs/api-reference/vector-stores-files
(list files, retrieve file content). The mock is not a live OpenAI service."""

import json
import sys
import types

import pytest

pytest.importorskip("openai")

from holographic_sdk.importers.sources.openai_vs import OpenAiVectorStoreSource

from .conftest import assert_faithful, assert_resumable, vec
from .mockserver import ApiError, MockApi

KEY = "sk-test-mock-key-123456"


def _store(i):
    return {"id": f"vs_{i}", "object": "vector_store", "created_at": 1_700_000_000 + i, "name": f"Store {i}",
            "status": "completed", "metadata": {"team": "t"}, "usage_bytes": 1, "file_counts": {"completed": 3},
            "last_active_at": 1_700_000_000}


def _vsfile(store, i, status="completed"):
    return {"id": f"file-{store}-{i}", "object": "vector_store.file", "created_at": 1_710_000_000 + i,
            "vector_store_id": f"vs_{store}", "status": status, "usage_bytes": 10, "last_error": None,
            "chunking_strategy": {"type": "static"}, "attributes": {"author": f"a{i}", "n": i}}


def _page(items, req):
    after, limit = req.q("after"), int(req.q("limit", "20"))
    ids = [x["id"] for x in items]
    start = ids.index(after) + 1 if after in ids else 0
    chunk = items[start:start + limit]
    return {"object": "list", "data": chunk, "first_id": chunk[0]["id"] if chunk else None,
            "last_id": chunk[-1]["id"] if chunk else None, "has_more": start + limit < len(items)}


@pytest.fixture
def api():
    stores = [_store(0), _store(1)]
    files = {0: [_vsfile(0, i) for i in range(3)], 1: [_vsfile(1, i) for i in range(3)]}
    files[1].append(_vsfile(1, 9, status="failed"))
    mock = MockApi(required_auth=("authorization", f"Bearer {KEY}"))

    @mock.route("GET", r"/v1/vector_stores")
    def list_stores(req):
        return _page(stores, req)

    @mock.route("GET", r"/v1/vector_stores/vs_(?P<s>\d+)")
    def get_store(req):
        return stores[int(req.match["s"])]

    @mock.route("GET", r"/v1/vector_stores/vs_(?P<s>\d+)/files")
    def list_files(req):
        return _page(files[int(req.match["s"])], req)

    @mock.route("GET", r"/v1/vector_stores/vs_(?P<s>\d+)/files/(?P<f>[^/]+)/content")
    def content(req):
        f = req.match["f"]
        if f.endswith("-9"):
            raise ApiError(404)
        return {"object": "vector_store.file_content.page", "has_more": False, "next_page": None,
                "data": [{"type": "text", "text": f"{f} chunk {k} body"} for k in range(2)]}

    @mock.route("GET", r"/v1/files/(?P<f>[^/]+)")
    def file(req):
        return {"id": req.match["f"], "object": "file", "filename": f"{req.match['f']}.pdf", "bytes": 1,
                "created_at": 1, "purpose": "assistants", "status": "processed"}

    with mock:
        yield mock


@pytest.fixture
def embedder(monkeypatch):
    mod = types.ModuleType("oa_test_embedder")
    mod.embed = lambda texts: [vec(sum(map(ord, t)) % 10007) for t in texts]
    monkeypatch.setitem(sys.modules, "oa_test_embedder", mod)
    return "oa_test_embedder:embed"


def make(api, **kw):
    return OpenAiVectorStoreSource(KEY, api.url + "/v1", kw.pop("stores", []), kw.pop("namespace_by", "id"),
                                   page_size=kw.pop("page_size", 2))


def test_openai_vector_store_mock_roundtrip(api, run, server, embedder):
    report, tenant = run(make(api), embedder=embedder, allow_rejects=True)
    api.assert_clean()
    # 3 + 3 completed files x 2 chunks; the failed file is rejected visibly, not dropped
    assert report.stats.counts["docs_acked"] == 12
    assert report.stats.rejects == {"source_unreadable": 1}
    assert_faithful_ignoring_rejects(report)
    sink = server.sink(tenant)
    try:
        hits = sink.query(vector=vec(0), top_k=100, filter={"hms_source_id": "file-1-2#1"})
    finally:
        sink.close()
    m = hits[0]["metadata"]
    assert m["filename"] == "file-1-2.pdf" and m["author"] == "a2" and m["vector_store_id"] == "vs_1"
    assert m["hms_namespace"] == "vs_1" and m["hms_created_at"].startswith("2024-03-")


def assert_faithful_ignoring_rejects(report):
    v = report.verification
    assert v["performed"] and v["roundtrip_mismatches"] == 0 and v["self_hit_rate"] == 1.0


def test_openai_vector_store_mock_resume(api, server, tmp_path, embedder):
    assert_resumable(lambda: make(api, stores=["vs_0"]), server, tmp_path, expected_docs=6, batch_size=2,
                     embedder=embedder)
    api.assert_clean()


def test_openai_vector_store_mock_credential_not_persisted(api, run, tmp_path, embedder):
    report, _ = run(make(api, stores=["vs_0"]), embedder=embedder, checkpoint_path=str(tmp_path / "ck.json"),
                    report_path=str(tmp_path / "rep.json"))
    assert KEY not in (tmp_path / "ck.json").read_text() + (tmp_path / "rep.json").read_text()
    assert KEY not in json.dumps(report.descriptor)
