"""LangChain contracts against WARM, using deterministic local embeddings."""

import uuid

import pytest

pytest.importorskip("langchain_core")
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from writerslogic_warm_sdk.langchain import WarmVectorStore
from test_live_server import API_KEY, BIN, embedding, server

pytestmark = pytest.mark.skipif(not BIN, reason="WARM_SERVER_BIN not set")


class LocalEmbeddings(Embeddings):
    def __init__(self, count=None):
        self.count = count
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        if self.count is not None:
            return [embedding(i) for i in range(self.count)]
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return embedding(sum(text.encode("utf-8")))


@pytest.fixture
def store(server):
    value = WarmVectorStore(
        embedding=LocalEmbeddings(), url=server, api_key=API_KEY,
        tenant_id=uuid.uuid4().hex,
    )
    try:
        yield value
    finally:
        value.close()


def test_supplied_ids_survive_upserts_and_search(store):
    assert store.add_texts(iter(["first", "second"]), ids=["one", "two"]) == ["one", "two"]
    assert store.add_texts(["replacement"], ids=["one"]) == ["one"]
    found = store.similarity_search("replacement", k=10)
    assert {doc.id: doc.page_content for doc in found} == {"one": "replacement", "two": "second"}


def test_document_ids_and_explicit_override_are_preserved(store):
    assert store.add_documents([Document(id="source", page_content="one")]) == ["source"]
    assert store.add_documents([Document(id="ignored", page_content="two")], ids=["override"]) == ["override"]
    ids = store.add_documents([Document(id="retained", page_content="three"), Document(page_content="four")])
    assert ids[0] == "retained"
    assert len(ids[1]) > 0 and ids[1] != "retained"


def test_retriever_forwards_metadata_filters(store):
    store.add_texts(["public", "private"], metadatas=[{"scope": "public"}, {"scope": "private"}])
    retriever = store.as_retriever(search_kwargs={"k": 10, "filter": {"scope": "public"}})
    found = retriever.invoke("private")
    assert [doc.page_content for doc in found] == ["public"]
    assert found[0].metadata == {"scope": "public"}


@pytest.mark.parametrize("field", ["metadatas", "ids"])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_mismatched_input_counts_fail_before_embedding_or_storage(store, field, count):
    values = [{} for _ in range(count)] if field == "metadatas" else [str(i) for i in range(count)]
    with pytest.raises(ValueError):
        store.add_texts(["one", "two"], **{field: values})
    assert store._embedding.calls == 0
    assert store._client.query(embedding(0), top_k=10)["matches"] == []


@pytest.mark.parametrize("count", [0, 1, 3])
def test_mismatched_embedding_counts_never_partially_store(store, count):
    store._embedding = LocalEmbeddings(count=count)
    with pytest.raises(ValueError):
        store.add_texts(["one", "two"])
    assert store._client.query(embedding(0), top_k=10)["matches"] == []


def test_empty_input_skips_embedding(store):
    assert store.add_texts(iter([])) == []
    assert store._embedding.calls == 0


def test_from_texts_accepts_ids_without_forwarding_them_to_constructor(server):
    store = WarmVectorStore.from_texts(
        ["factory"], LocalEmbeddings(), ids=["factory-id"],
        metadatas=[{"source": "factory"}], url=server, api_key=API_KEY,
        tenant_id=uuid.uuid4().hex,
    )
    try:
        found = store.similarity_search("factory")
        assert [(doc.id, doc.page_content, doc.metadata) for doc in found] == [
            ("factory-id", "factory", {"source": "factory"})
        ]
    finally:
        store.close()


@pytest.mark.parametrize("invalid", ["", 1, b"bytes"])
def test_invalid_ids_fail_before_embedding_or_storage(store, invalid):
    with pytest.raises(ValueError):
        store.add_texts(["one"], ids=[invalid])
    assert store._embedding.calls == 0
    assert store._client.query(embedding(0), top_k=10)["matches"] == []
