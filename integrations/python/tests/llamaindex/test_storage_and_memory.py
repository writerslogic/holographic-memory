# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.memory import Memory
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.storage.chat_store.base_db import MessageStatus
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.schema import TextNode

from holographic_integrations.core import HmsError
from holographic_integrations.llamaindex import (
    HolographicAsyncChatStore,
    HolographicChatStore,
    HolographicDocumentStore,
    HolographicIndexStore,
    HolographicKVStore,
    HolographicMemoryBlock,
    HolographicVectorStore,
)

DIM = 16


def kw(server):
    return dict(url=server.url, tenant_id=server.tenant(), input_dim=DIM)


@pytest.fixture
def srv(server_factory):
    return server_factory(DIM)


# --- KV store -------------------------------------------------------------------------


def test_kv_store_contract(srv):
    kv = HolographicKVStore(**kw(srv))
    kv.put("a", {"x": 1})
    kv.put_all([("b", {"x": 2}), ("c", {"x": 3})], collection="other")
    assert kv.get("a") == {"x": 1} and kv.get("a", collection="other") is None
    assert kv.get_all() == {"a": {"x": 1}}
    assert kv.get_all("other") == {"b": {"x": 2}, "c": {"x": 3}}
    kv.put("a", {"x": 9})
    assert kv.get("a") == {"x": 9}
    assert kv.delete("a") is True and kv.delete("a") is False
    long_key = "k" * 400
    kv.put(long_key, {"v": 1})
    assert kv.get(long_key) == {"v": 1} and long_key in kv.get_all()
    with pytest.raises(HmsError):
        kv.put("big", {"v": "w " * 40000})
    kv.close()


async def test_kv_store_async(srv):
    kv = HolographicKVStore(**kw(srv))
    await kv.aput_all([(f"k{i}", {"i": i}) for i in range(20)])
    assert len(await kv.aget_all()) == 20
    assert await kv.aget("k7") == {"i": 7}
    assert await kv.adelete("k7") and await kv.aget("k7") is None
    await kv.aclose()


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_kv_bucket_capacity_boundary(server_factory, delta):
    srv = server_factory(DIM, max_top_k=10)
    kv = HolographicKVStore(url=srv.url, tenant_id=srv.tenant(), input_dim=DIM, max_fetch=10, buckets=1)
    n = 10 + delta
    kv.put_all([(f"k{i}", {"i": i}) for i in range(n)])
    if n >= 10:
        with pytest.raises(HmsError):
            kv.get_all()
    else:
        assert len(kv.get_all()) == n
    kv.close()


# --- docstore / index store -----------------------------------------------------------


def test_full_storage_context_persists_in_hms(srv):
    base = kw(srv)
    embed = MockEmbedding(embed_dim=DIM)
    ctx = StorageContext.from_defaults(
        vector_store=HolographicVectorStore(**base, collection="vec"),
        docstore=HolographicDocumentStore(**base, collection="docs"),
        index_store=HolographicIndexStore(**base, collection="idx"),
    )
    docs = [Document(text=f"fact {i}", doc_id=f"d{i}") for i in range(5)]
    index = VectorStoreIndex.from_documents(docs, storage_context=ctx, embed_model=embed)
    index_id = index.index_id

    fresh = StorageContext.from_defaults(
        vector_store=HolographicVectorStore(**base, collection="vec"),
        docstore=HolographicDocumentStore(**base, collection="docs"),
        index_store=HolographicIndexStore(**base, collection="idx"),
    )
    loaded = load_index_from_storage(fresh, index_id=index_id, embed_model=embed)
    assert len(loaded.as_retriever(similarity_top_k=5).retrieve("fact")) == 5
    assert {d.doc_id for d in fresh.docstore.docs.values()} == set()  # nodes live in the vector store
    assert sorted(fresh.docstore.get_all_ref_doc_info() or {}) == []


def test_docstore_roundtrip_matches_simple_docstore(srv):
    hms = HolographicDocumentStore(**kw(srv))
    ref = SimpleDocumentStore()
    nodes = [TextNode(id_=f"n{i}", text=f"text {i}", metadata={"i": i}) for i in range(6)]
    for store in (hms, ref):
        store.add_documents(nodes)
    assert set(hms.docs) == set(ref.docs)
    assert hms.get_document("n3").text == "text 3" and hms.document_exists("n3")
    assert hms.get_document_hash("n3") == ref.get_document_hash("n3")
    hms.delete_document("n3")
    ref.delete_document("n3")
    assert set(hms.docs) == set(ref.docs) and not hms.document_exists("n3")
    assert hms.get_all_document_hashes() == ref.get_all_document_hashes()


async def test_docstore_async(srv):
    store = HolographicDocumentStore(**kw(srv))
    await store.async_add_documents([TextNode(id_=f"a{i}", text=f"t{i}") for i in range(4)])
    assert len(await store.aget_all_document_hashes()) == 4
    assert (await store.aget_document("a2")).text == "t2"
    await store.adelete_document("a2")
    assert not await store.adocument_exists("a2")


# --- chat stores ----------------------------------------------------------------------


def msgs(n, start=0):
    return [ChatMessage(role="user" if i % 2 == 0 else "assistant", content=f"m{i}") for i in range(start, start + n)]


def test_chat_store_matches_simple_store(srv):
    from llama_index.core.storage.chat_store import SimpleChatStore

    hms, ref = HolographicChatStore(**kw(srv)), SimpleChatStore()
    for s in (hms, ref):
        s.set_messages("k1", msgs(4))
        s.add_message("k1", msgs(1, 4)[0])
        s.set_messages("k2", msgs(2))
    text = lambda ms: [m.content for m in ms]
    assert text(hms.get_messages("k1")) == text(ref.get_messages("k1"))
    assert hms.get_keys() == sorted(ref.get_keys())
    assert hms.delete_last_message("k1").content == ref.delete_last_message("k1").content
    assert hms.delete_message("k1", 0).content == "m0"  # sequence number
    assert text(hms.get_messages("k1")) == ["m1", "m2", "m3"]
    assert text(hms.delete_messages("k2")) == ["m0", "m1"] and "k2" not in hms.get_keys()
    assert hms.delete_messages("nope") is None
    hms.set_messages("k1", msgs(2))
    assert text(hms.get_messages("k1")) == ["m0", "m1"]


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_long_history_crosses_page_boundaries(server_factory, delta):
    srv = server_factory(DIM, max_top_k=10)
    store = HolographicChatStore(url=srv.url, tenant_id=srv.tenant(), input_dim=DIM, max_fetch=10, page_size=10)
    n = 20 + delta  # one page = 10; the history spans the 2-page boundary at 20
    for batch in range(0, n, 7):
        for m in msgs(min(7, n - batch), batch):
            store.add_message("long", m)
    assert [m.content for m in store.get_messages("long")] == [f"m{i}" for i in range(n)]
    with pytest.raises(ValueError):
        HolographicChatStore(url=srv.url, tenant_id=srv.tenant(), input_dim=DIM, max_fetch=10, page_size=11)


def test_chat_memory_buffer_uses_the_store(srv):
    store = HolographicChatStore(**kw(srv))
    mem = ChatMemoryBuffer.from_defaults(chat_store=store, chat_store_key="u1", token_limit=3000)
    mem.put(ChatMessage(role="user", content="hello there"))
    mem.put(ChatMessage(role="assistant", content="hi"))
    again = ChatMemoryBuffer.from_defaults(chat_store=HolographicChatStore(**kw(srv) | {"tenant_id": store._log.config.tenant_id}), chat_store_key="u1")
    assert [m.content for m in again.get()] == ["hello there", "hi"]


async def test_async_chat_store_status_semantics(srv):
    s = HolographicAsyncChatStore(**kw(srv))
    await s.add_messages("k", msgs(6))
    assert await s.count_messages("k") == 6
    archived = await s.archive_oldest_messages("k", 2)
    assert [m.content for m in archived] == ["m0", "m1"]
    assert await s.count_messages("k") == 4 and await s.count_messages("k", MessageStatus.ARCHIVED) == 2
    assert [m.content for m in await s.get_messages("k", limit=2, offset=1)] == ["m3", "m4"]
    assert [m.content for m in await s.delete_oldest_messages("k", 1)] == ["m2"]
    await s.delete_messages("k", MessageStatus.ARCHIVED)
    assert [m.content for m in await s.get_messages("k", status=None)] == ["m3", "m4", "m5"]
    assert await s.get_keys() == ["k"]
    await s.delete_messages("k")
    assert await s.get_keys() == []
    await s.aclose()


# --- Memory + memory block ------------------------------------------------------------


async def test_memory_with_hms_chat_store_and_block(server_factory):
    srv = server_factory(256)
    base = dict(url=srv.url, tenant_id=srv.tenant(), input_dim=256)
    block = HolographicMemoryBlock(**base, similarity_top_k=2, retrieval_context_window=1)
    store = HolographicAsyncChatStore(**base)
    memory = Memory.from_defaults(
        session_id="s1", token_limit=120, token_flush_size=60, chat_store=store, memory_blocks=[block], chat_history_token_ratio=0.5
    )
    facts = [
        "My favorite fruit is the mango and I eat one daily",
        "I live in Lisbon near the river with my two cats",
        "I work as a marine biologist studying coral reefs",
        "My sister plays the cello in an orchestra in Vienna",
    ]
    for f in facts:
        await memory.aput(ChatMessage(role="user", content=f))
        await memory.aput(ChatMessage(role="assistant", content="Noted, thank you for sharing that detail with me."))
    recalled = await block.aget([ChatMessage(role="user", content="what is my favorite fruit mango")], session_id="s1")
    assert "mango" in recalled
    other = await block.aget([ChatMessage(role="user", content="what is my favorite fruit mango")], session_id="other-session")
    assert other == ""
    got = await memory.aget(input="Tell me about the mango fruit I like")
    joined = " ".join(m.content for m in got)
    assert "mango" in joined
    await block.aclose()
    await store.aclose()


async def test_stock_vector_memory_block_works_on_the_hms_store(srv):
    from llama_index.core.memory.memory_blocks import VectorMemoryBlock

    store = HolographicVectorStore(**kw(srv))
    block = VectorMemoryBlock(vector_store=store, embed_model=MockEmbedding(embed_dim=DIM), similarity_top_k=1)
    await block.aput([ChatMessage(role="user", content="I adopted a greyhound named Biscuit")], session_id="s9")
    assert "Biscuit" in await block.aget([ChatMessage(role="user", content="dog")], session_id="s9")
    # the stock block caches its first session filter in query_kwargs, so use a fresh block per session
    other = VectorMemoryBlock(vector_store=store, embed_model=MockEmbedding(embed_dim=DIM), similarity_top_k=1)
    assert await other.aget([ChatMessage(role="user", content="dog")], session_id="s10") == ""
    await store.aclose()
