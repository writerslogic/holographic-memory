# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest

pytest.importorskip("haystack")

from haystack import Document, Pipeline  # noqa: E402
from haystack.dataclasses import ByteStream, SparseEmbedding  # noqa: E402
from haystack.document_stores.types import DuplicatePolicy, FilterPolicy  # noqa: E402
from haystack.errors import FilterError  # noqa: E402
from haystack.utils import Secret  # noqa: E402

from holographic_integrations.core import FunctionEmbedder, HashingEmbedder, HmsError  # noqa: E402
from holographic_integrations.haystack import (  # noqa: E402
    HolographicDocumentStore, HolographicEmbeddingRetriever, HolographicTextRetriever,
)
from holographic_integrations.haystack import _filters  # noqa: E402

DIM = 64
TOPICS = ["astronomy telescope galaxy", "cooking pasta tomato", "football goal striker", "python code compiler"]


def make_store(server, **kw):
    kw.setdefault("embedding_dim", DIM)
    return HolographicDocumentStore(url=server.url, tenant_id=server.tenant(), **kw)


def docs(n=8):
    return [
        Document(content=f"{TOPICS[i % 4]} item{i}", meta={"topic": TOPICS[i % 4].split()[0], "n": i, "year": 2000 + i})
        for i in range(n)
    ]


def test_roundtrip_preserves_blob_sparse_none_content_and_clashing_meta(server):
    store = make_store(server)
    d = Document(
        content=None,
        blob=ByteStream(b"\x00\x01binary", mime_type="application/x-test"),
        meta={"content": "meta key that clashes with a field name", "nested": {"a": [1, 2]}},
        embedding=HashingEmbedder(DIM).embed_query("x"),
        sparse_embedding=SparseEmbedding(indices=[1, 5], values=[0.5, 0.25]),
    )
    store.write_documents([d])
    [got] = store.filter_documents()
    assert got == d
    assert store.filter_documents({"field": "meta.nested.a", "operator": "in", "value": [[1, 2]]}) == [d]


def test_documents_without_embedding_are_embedded_and_retrievable(server):
    store = make_store(server)
    store.write_documents(docs())
    hits = store.text_retrieval("cooking pasta tomato item1", top_k=2)
    assert hits[0].content.startswith("cooking pasta")
    assert hits[0].score is not None and hits[0].embedding is None


def test_filter_pushdown_and_residual(server):
    store = make_store(server)
    store.write_documents(docs(12))
    flt = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.topic", "operator": "==", "value": "python"},
            {"field": "meta.year", "operator": ">=", "value": 2007},
        ],
    }
    assert {d.meta["n"] for d in store.filter_documents(flt)} == {7, 11}
    pushed, complete = _filters.pushdown(flt, ())
    assert pushed == {"topic": "python"} and not complete
    # OR and NOT are evaluated locally by Haystack's own evaluator
    either = {"operator": "OR", "conditions": [{"field": "meta.n", "operator": "==", "value": 1}, {"field": "meta.n", "operator": "==", "value": 2}]}
    assert {d.meta["n"] for d in store.filter_documents(either)} == {1, 2}
    # retrieval with a residual filter over-fetches so the filter does not starve the result
    hits = store.text_retrieval("item", filters={"field": "meta.n", "operator": ">", "value": 9}, top_k=2)
    assert {d.meta["n"] for d in hits} == {10, 11}


def test_numeric_equality_is_not_pushed_so_1_equals_1_0():
    pushed, _ = _filters.pushdown({"field": "meta.n", "operator": "==", "value": 1}, ())
    assert pushed == {}
    pushed, _ = _filters.pushdown({"field": "meta.d", "operator": "==", "value": "2020-01-01"}, ())
    assert pushed == {}


def test_invalid_filters_raise_even_on_empty_store(server):
    store = make_store(server)
    for bad in (
        {"conditions": []},
        {"operator": "AND"},
        {"field": "meta.a", "operator": "in", "value": 3},
        {"field": "meta.a", "operator": "~=", "value": 3},
    ):
        with pytest.raises(FilterError):
            store.filter_documents(bad)


@pytest.mark.parametrize("count", [9, 10, 11])
def test_enumeration_limit_boundary(server_factory, count):
    srv = server_factory(DIM, max_top_k=10, max_tenants=100000)
    store = HolographicDocumentStore(url=srv.url, tenant_id=srv.tenant(), embedding_dim=DIM, list_buckets=1, max_fetch=10)
    store.write_documents(docs(count))
    if count < 10:
        assert store.count_documents() == count
    else:
        with pytest.raises(HmsError, match="list_buckets"):
            store.count_documents()


def test_buckets_extend_enumeration_past_max_fetch(server_factory):
    srv = server_factory(DIM, max_top_k=10, max_tenants=100000)
    store = HolographicDocumentStore(url=srv.url, tenant_id=srv.tenant(), embedding_dim=DIM, list_buckets=16, max_fetch=10)
    store.write_documents(docs(40))
    assert store.count_documents() == 40
    assert len(store.filter_documents({"field": "meta.n", "operator": ">=", "value": 20})) == 20
    store.delete_all_documents()
    assert store.count_documents() == 0


def test_collections_share_a_tenant_without_mixing(server):
    tenant = server.tenant()
    a = HolographicDocumentStore(url=server.url, tenant_id=tenant, collection="a", embedding_dim=DIM)
    b = HolographicDocumentStore(url=server.url, tenant_id=tenant, collection="b", embedding_dim=DIM)
    d = Document(id="same", content="shared id")
    a.write_documents([d])
    b.write_documents([Document(id="same", content="other")], DuplicatePolicy.FAIL)
    assert a.filter_documents()[0].content == "shared id" and b.filter_documents()[0].content == "other"


def test_reserved_meta_rejected(server):
    with pytest.raises(ValueError, match="reserved"):
        make_store(server).write_documents([Document(content="x", meta={"hs_bucket": 1})])


def test_masked_store_still_retrieves(server):
    store = make_store(server, zero_trust_key=Secret.from_token("a long enough passphrase"))
    store.write_documents(docs())
    assert store.text_retrieval("football goal striker item2", top_k=1)[0].meta["n"] == 2


def test_serialization_roundtrip(server, monkeypatch):
    monkeypatch.setenv("HMS_API_KEY", "secret-from-env")
    store = HolographicDocumentStore(
        url=server.url, tenant_id="abc", collection="c", embedding_dim=DIM, embedder=HashingEmbedder(DIM, bigrams=False),
        list_buckets=8, return_embedding=False,
    )
    data = store.to_dict()
    assert "secret-from-env" not in json.dumps(data)
    again = HolographicDocumentStore.from_dict(data)
    assert (again.tenant_id, again.collection, again.list_buckets, again.return_embedding) == ("abc", "c", 8, False)
    assert again.embedder.bigrams is False
    retriever = HolographicTextRetriever(store, filters={"field": "meta.a", "operator": "==", "value": "b"}, top_k=3)
    back = HolographicTextRetriever.from_dict(retriever.to_dict())
    assert back.top_k == 3 and back.filters == retriever.filters and back.document_store.collection == "c"
    emb = HolographicEmbeddingRetriever(store, scale_score=True, filter_policy=FilterPolicy.MERGE)
    assert HolographicEmbeddingRetriever.from_dict(emb.to_dict()).filter_policy == FilterPolicy.MERGE
    with pytest.raises(ValueError):
        HolographicDocumentStore(embedder=FunctionEmbedder(lambda texts: [[1.0] * DIM for _ in texts], DIM), embedding_dim=DIM).to_dict()


def test_pipeline_with_retrievers_and_filter_policy(server):
    store = make_store(server)
    store.write_documents(docs(12))
    pipe = Pipeline()
    pipe.add_component("retriever", HolographicTextRetriever(store, top_k=3, filter_policy=FilterPolicy.MERGE,
                                                             filters={"field": "meta.topic", "operator": "==", "value": "cooking"}))
    out = pipe.run({"retriever": {"query": "pasta tomato", "filters": {"field": "meta.n", "operator": ">", "value": 4}}})
    got = out["retriever"]["documents"]
    assert got and all(d.meta["topic"] == "cooking" and d.meta["n"] > 4 for d in got)
    # pipeline round trip through YAML-able dict
    again = Pipeline.from_dict(pipe.to_dict())
    assert again.get_component("retriever").document_store.tenant_id == store.tenant_id


def test_embedding_retriever_cosine_and_scale(server):
    store = make_store(server)
    emb = HashingEmbedder(DIM)
    store.write_documents([Document(content=t, embedding=emb.embed_query(t)) for t in TOPICS])
    r = HolographicEmbeddingRetriever(store, top_k=2)
    out = r.run(emb.embed_query(TOPICS[2]))["documents"]
    assert out[0].content == TOPICS[2] and out[0].score == pytest.approx(1.0, abs=1e-6)
    scaled = r.run(emb.embed_query(TOPICS[2]), scale_score=True, return_embedding=True)["documents"]
    assert scaled[0].score == pytest.approx(1.0, abs=1e-6) and scaled[0].embedding is not None
    assert r.run(emb.embed_query("x"), top_k=0) == {"documents": []}
    with pytest.raises(ValueError):
        HolographicEmbeddingRetriever(store, top_k=0)


async def test_async_store_and_components(server):
    store = make_store(server)
    try:
        assert await store.write_documents_async(docs(8)) == 8
        assert await store.count_documents_async() == 8
        assert await store.write_documents_async(docs(8), DuplicatePolicy.SKIP) == 0
        r = HolographicTextRetriever(store, top_k=2)
        out = (await r.run_async("python code compiler item3"))["documents"]
        assert out[0].meta["n"] == 3
        assert await store.delete_by_filter_async({"field": "meta.topic", "operator": "==", "value": "python"}) == 2
        assert await store.update_by_filter_async({"field": "meta.topic", "operator": "==", "value": "cooking"}, {"tag": "x"}) == 2
        assert len(await store.filter_documents_async({"field": "meta.tag", "operator": "==", "value": "x"})) == 2
        await store.delete_all_documents_async()
        assert await store.count_documents_async() == 0
    finally:
        await store.aclose()


def test_text_queries_mode_sends_text_to_service():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        if request.url.path.endswith("/query"):
            return httpx.Response(200, json={"matches": [
                {"id": "d1", "text": "hello", "metadata": {"_hms_id": "d1", "_hms_collection": "", "k": "v", "hs_bucket": 0}, "score": 0.03}]})
        return httpx.Response(200, json={"added": len(body)})

    store = HolographicDocumentStore(url="http://svc", embedding_dim=8, text_queries=True, transport=httpx.MockTransport(handler))
    store.write_documents([Document(id="d1", content="hello")], DuplicatePolicy.OVERWRITE)
    [doc] = store.text_retrieval("hello there", top_k=1)
    batch = next(b for p, b in seen if p.endswith("batch"))
    assert "vector" not in batch[0]
    query = next(b for p, b in seen if p.endswith("/query"))
    assert query["query"] == "hello there" and "query_vector" not in query
    assert doc.score == 0.03 and doc.meta == {"k": "v"}
