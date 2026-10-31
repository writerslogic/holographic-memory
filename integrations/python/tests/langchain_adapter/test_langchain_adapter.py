# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from writerslogic_warm_integrations.core import WarmConfig
from writerslogic_warm_integrations.langchain import (
    WarmChatMessageHistory,
    WarmRetriever,
    WarmVectorStore,
)

DIM = 16


def store(server, **kw):
    return WarmVectorStore(DeterministicFakeEmbedding(size=server.dim), url=server.url, tenant_id=server.tenant(), **kw)


@pytest.fixture
def srv(server_factory):
    return server_factory(DIM)


def docs(n):
    return [
        Document(page_content=f"item {i} about {'cats' if i % 2 else 'dogs'}", metadata={"n": i, "kind": "cat" if i % 2 else "dog", "tags": ["a", "b"] if i < 3 else ["c"]})
        for i in range(n)
    ]


def test_filters_equality_pushed_and_operators_client_side(srv):
    s = store(srv)
    s.add_documents(docs(12))
    kinds = {d.metadata["kind"] for d in s.similarity_search("item", k=6, filter={"kind": "cat"})}
    assert kinds == {"cat"}
    got = s.similarity_search("item", k=12, filter={"n": {"$gte": 9}})
    assert {d.metadata["n"] for d in got} == {9, 10, 11}
    got = s.similarity_search("item", k=12, filter={"$or": [{"n": 0}, {"n": 11}], "kind": {"$in": ["cat", "dog"]}})
    assert {d.metadata["n"] for d in got} == {0, 11}
    got = s.similarity_search("item", k=12, filter={"tags": {"$contains": "b"}})
    assert {d.metadata["n"] for d in got} == {0, 1, 2}


def test_scores_relevance_and_by_vector(srv):
    emb = DeterministicFakeEmbedding(size=DIM)
    s = WarmVectorStore(emb, url=srv.url, tenant_id=srv.tenant())
    s.add_documents(docs(5))
    (doc, score), *_ = s.similarity_search_with_score("item 3 about cats", k=3)
    assert doc.page_content == "item 3 about cats" and score == pytest.approx(1.0, abs=1e-6)
    rel = s.similarity_search_with_relevance_scores("item 3 about cats", k=5)
    assert all(0.0 <= r <= 1.0 for _, r in rel) and rel[0][1] == pytest.approx(1.0, abs=1e-6)
    assert s.similarity_search_with_relevance_scores("item 3 about cats", k=5, score_threshold=0.999)[0][0].page_content == "item 3 about cats"
    by_vec = s.similarity_search_by_vector(emb.embed_query("item 1 about cats"), k=1)
    assert by_vec[0].page_content == "item 1 about cats"


def test_mmr_diversifies(srv):
    s = store(srv)
    texts = ["alpha one", "alpha one", "alpha one", "beta two"]
    s.add_texts(texts, ids=["a", "b", "c", "d"])
    plain = s.similarity_search("alpha one", k=2)
    assert [d.page_content for d in plain] == ["alpha one", "alpha one"]
    mmr = s.max_marginal_relevance_search("alpha one", k=2, fetch_k=4, lambda_mult=0.0)
    assert sorted(d.page_content for d in mmr) != ["alpha one", "alpha one"]
    # fetch_k above the cap of 100 is clamped rather than rejected
    assert len(s.max_marginal_relevance_search("alpha one", k=2, fetch_k=10_000)) == 2


async def test_async_paths_and_from_texts(srv):
    emb = DeterministicFakeEmbedding(size=DIM)
    s = await WarmVectorStore.afrom_texts(
        ["x one", "y two"], emb, metadatas=[{"m": 1}, {"m": 2}], ids=["1", "2"], url=srv.url, tenant_id=srv.tenant()
    )
    try:
        assert [d.id for d in await s.asimilarity_search("y two", k=1)] == ["2"]
        assert (await s.aget_by_ids(["1", "9"]))[0].metadata == {"m": 1}
        mmr = await s.amax_marginal_relevance_search("x one", k=1)
        assert mmr[0].id == "1"
        await s.adelete(filter={"m": 1})
        assert await s.aget_by_ids(["1"]) == []
        with pytest.raises(ValueError):
            await s.adelete()
        assert await s.adelete_all() == 1
    finally:
        await s.aclose()


def test_retriever_modes_and_as_retriever(srv):
    s = store(srv)
    s.add_documents(docs(6))
    r = WarmRetriever(vectorstore=s, k=3, include_score=True)
    out = r.invoke("item 2 about dogs", k=2, filter={"kind": "dog"})
    assert len(out) == 2 and all(d.metadata["kind"] == "dog" and "warm_score" in d.metadata for d in out)
    thr = WarmRetriever(vectorstore=s, search_type="similarity_score_threshold", score_threshold=0.99).invoke("item 2 about dogs")
    assert [d.page_content for d in thr] == ["item 2 about dogs"]
    assert len(WarmRetriever(vectorstore=s, search_type="mmr", k=2).invoke("item")) == 2
    assert len(s.as_retriever(search_kwargs={"k": 2, "filter": {"kind": "cat"}}).invoke("item")) == 2
    with pytest.raises(ValueError):
        WarmRetriever(vectorstore=s, search_type="similarity_score_threshold").invoke("x")


def test_collection_and_reserved_keys(srv):
    a = store(srv, collection="a")
    a.add_texts(["only in a"], ids=["1"])
    cfg = WarmConfig(url=srv.url, tenant_id=a.config.tenant_id, collection="b", embedder=a.config.embedder, input_dim=DIM)
    b = WarmVectorStore(config=cfg)
    assert b.similarity_search("only in a") == []
    with pytest.raises(ValueError):
        a.add_texts(["x"], [{"_warm_id": 1}])
    with pytest.raises(ValueError):
        WarmVectorStore(DeterministicFakeEmbedding(size=DIM), config=cfg, url="http://x")


def test_default_embedder_without_langchain_embeddings(server_factory):
    srv = server_factory(384)
    s = WarmVectorStore(url=srv.url, tenant_id=srv.tenant())
    s.add_texts(["quarterly revenue grew", "the cat sat"], ids=["1", "2"])
    assert s.similarity_search("revenue grew quarterly", k=1)[0].id == "1"


def test_chat_history_orders_across_pages_sync(srv):
    cfg = WarmConfig(url=srv.url, tenant_id=srv.tenant(), input_dim=DIM)
    h = WarmChatMessageHistory("s1", config=cfg, page_size=10)
    other = WarmChatMessageHistory("s2", config=cfg, page_size=10)
    other.add_user_message("not mine")
    msgs = [HumanMessage(f"q{i}") if i % 2 == 0 else AIMessage(f"a{i}") for i in range(25)]
    h.add_messages(msgs[:7])
    h.add_messages(msgs[7:])
    got = h.messages
    assert [m.content for m in got] == [m.content for m in msgs]
    assert isinstance(got[1], AIMessage) and isinstance(got[0], HumanMessage)
    fresh = WarmChatMessageHistory("s1", config=cfg, page_size=10)  # new writer continues the sequence
    fresh.add_message(SystemMessage("tail"))
    assert [m.content for m in h.messages][-2:] == ["q24", "tail"]
    assert [m.content for m in other.messages] == ["not mine"]
    h.clear()
    assert h.messages == [] and len(other.messages) == 1


@pytest.mark.parametrize("n", [9, 10, 11, 20, 21])
async def test_chat_history_page_boundaries_async(srv, n):
    cfg = WarmConfig(url=srv.url, tenant_id=srv.tenant(), input_dim=DIM)
    h = WarmChatMessageHistory("b", config=cfg, page_size=10)
    await h.aadd_messages([HumanMessage(f"m{i}") for i in range(n)])
    assert [m.content for m in await h.aget_messages()] == [f"m{i}" for i in range(n)]
    await h.aclear()
    assert await h.aget_messages() == []
    await h.aclose()


def test_text_query_mode_wire_format():
    """Text mode is exercised at the wire level only: it needs a server built with local-models."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        if request.url.path.endswith("/query"):
            return httpx.Response(200, json={"matches": [{"id": "1", "text": "hit", "metadata": {"_warm_id": "1", "_warm_collection": "", "k": "v"}, "score": 0.5}]})
        return httpx.Response(200, json={"added": len(body)})

    cfg = WarmConfig(url="http://warm.test", input_dim=8, text_queries=True, transport=httpx.MockTransport(handler))
    s = WarmVectorStore(config=cfg)
    s.add_texts(["some text"], ids=["1"])
    docs_ = s.similarity_search("a question", k=2, filter={"k": "v"})
    assert docs_ == [Document(id="1", page_content="hit", metadata={"k": "v"})]
    assert "vector" not in seen[0][1][0]
    assert seen[1][1] == {"top_k": 2, "query": "a question", "filter": {"k": "v", "_warm_collection": ""}}
