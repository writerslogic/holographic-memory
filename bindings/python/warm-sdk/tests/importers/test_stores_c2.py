# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""LlamaIndex and LangChain persisted stores, built with the real libraries (no mocks)."""

import json
import os
import pickle

import pytest

from writerslogic_warm_sdk.importers import ImporterError
from writerslogic_warm_sdk.importers.sources.langchain import LangChainSource
from writerslogic_warm_sdk.importers.sources.llamaindex import LlamaIndexSource

from .conftest import DIM, assert_faithful, assert_resumable, vec


def _vecs(n):
    return [vec(i) for i in range(n)]


@pytest.fixture
def llama_dir(tmp_path):
    pytest.importorskip("llama_index.core")
    from llama_index.core import VectorStoreIndex
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

    nodes = []
    for i in range(30):
        n = TextNode(id_=f"node-{i:02d}", text=f"llama node {i}", metadata={"k": i % 4, "creation_date": "2024-03-01"},
                     embedding=vec(i))
        n.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=f"doc-{i // 10}")
        nodes.append(n)
    index = VectorStoreIndex(nodes, embed_model=MockEmbedding(embed_dim=DIM), show_progress=False)
    index.storage_context.persist(str(tmp_path / "persist"))
    return str(tmp_path / "persist")


def test_llamaindex_real_roundtrip(llama_dir, run, server):
    report, tenant = run(LlamaIndexSource(llama_dir, "default__vector_store.json", "docstore.json", "", "auto"))
    assert_faithful(report, 30)
    sink = server.sink(tenant)
    try:
        hit = sink.query(vector=vec(7), top_k=1)[0]
    finally:
        sink.close()
    assert hit["id"] == "node-07" and hit["text"] == "llama node 7"
    meta = hit["metadata"]
    assert meta["k"] == 3 and meta["ref_doc_id"] == "doc-0" and meta["llamaindex_source"] == "doc-0"
    assert meta["warm_created_at"].startswith("2024-03-01")


def test_llamaindex_resume(llama_dir, server, tmp_path):
    assert_resumable(lambda: LlamaIndexSource(llama_dir, "default__vector_store.json", "docstore.json", "", "auto"),
                     server, tmp_path, expected_docs=30)


@pytest.fixture
def lc_faiss(tmp_path):
    pytest.importorskip("langchain_community")
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

    class E(Embeddings):
        def embed_documents(self, texts):
            return [vec(int(t.split()[-1])) for t in texts]

        def embed_query(self, text):
            return vec(int(text.split()[-1]))

    docs = [Document(page_content=f"lc doc {i}", metadata={"k": i, "tag": "x"}, id=f"id{i}") for i in range(25)]
    FAISS.from_documents(docs, E()).save_local(str(tmp_path / "faiss"))
    return str(tmp_path / "faiss"), docs


def test_langchain_faiss_real_roundtrip(lc_faiss, run, server):
    path, _ = lc_faiss
    report, tenant = run(LangChainSource(path, "auto", "index", "lc"))
    assert_faithful(report, 25)
    sink = server.sink(tenant)
    try:
        hit = sink.query(vector=vec(4), top_k=1)[0]
    finally:
        sink.close()
    assert hit["id"] == "lc::id4" and hit["text"] == "lc doc 4" and hit["metadata"]["k"] == 4
    assert hit["metadata"]["hms_namespace"] == "lc"


def test_langchain_faiss_resume(lc_faiss, server, tmp_path):
    path, _ = lc_faiss
    assert_resumable(lambda: LangChainSource(path, "auto", "index", ""), server, tmp_path, expected_docs=25)


def test_langchain_pickle_with_arbitrary_code_is_refused(tmp_path):
    pytest.importorskip("faiss")
    import faiss
    import numpy as np

    folder = tmp_path / "evil"
    folder.mkdir()
    index = faiss.IndexFlatL2(4)
    index.add(np.zeros((1, 4), dtype="float32"))
    faiss.write_index(index, str(folder / "index.faiss"))
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    (folder / "index.pkl").write_bytes(pickle.dumps(Evil()))
    with pytest.raises(ImporterError, match="refusing to unpickle"):
        list(LangChainSource(str(folder), "faiss", "index", "").records())
    assert not marker.exists()


def test_langchain_inmemory_jsonl_and_filestore_real(tmp_path, run, server):
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.load import dumpd, dumps
    from langchain_core.vectorstores import InMemoryVectorStore

    class E(Embeddings):
        def embed_documents(self, texts):
            return [vec(int(t.split()[-1])) for t in texts]

        def embed_query(self, text):
            return vec(int(text.split()[-1]))

    docs = [Document(page_content=f"mem doc {i}", metadata={"k": i}, id=f"m{i}") for i in range(10)]
    store = InMemoryVectorStore(E())
    store.add_documents(docs)
    dump = tmp_path / "store.json"
    store.dump(str(dump))
    report, _ = run(LangChainSource(str(dump), "auto", "index", ""))
    assert_faithful(report, 10)

    jl = tmp_path / "docs.jsonl"
    jl.write_text("".join(json.dumps(dumpd(d)) + "\n" for d in docs) + "not json\n")
    src = LangChainSource(str(jl), "auto", "index", "")
    recs = list(src.records())
    assert [r.id for r in recs[:10]] == [f"m{i}" for i in range(10)] and recs[10].error

    fs = tmp_path / "kv"
    fs.mkdir()
    for d in docs:
        (fs / d.id).write_bytes(dumps(d).encode())
    got = list(LangChainSource(str(fs), "auto", "index", "").records())
    assert sorted(r.id for r in got) == sorted(d.id for d in docs) and got[0].text.startswith("mem doc")
