# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""HolographicVectorStore against a real hms-server.

`SimpleVectorStore` (llama-index-core's reference implementation) is the oracle for filter
semantics: every filter case must select the same node ids from both stores.
"""

import dataclasses
import math
import random

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

from holographic_integrations.llamaindex import HolographicVectorStore

DIM = 16


def vec(seed):
    rng = random.Random(seed)
    v = [rng.uniform(-1, 1) for _ in range(DIM)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def nodes(n=12):
    out = []
    for i in range(n):
        out.append(
            TextNode(
                id_=f"n{i}",
                text=f"node text {i}",
                embedding=vec(i),
                metadata={
                    "idx": i,
                    "grp": "even" if i % 2 == 0 else "odd",
                    "score": i / 2,
                    "tags": ["a", "b"] if i % 3 == 0 else ["b", "c"],
                    "name": f"Alpha{i}" if i < 6 else f"beta{i}",
                    **({"opt": "x"} if i % 4 == 0 else {}),
                    **({"none_list": []} if i % 5 == 0 else {}),
                },
            )
        )
    return out


@pytest.fixture
def store(server_factory):
    srv = server_factory(DIM)
    s = HolographicVectorStore(url=srv.url, tenant_id=srv.tenant(), input_dim=DIM)
    yield s
    s.close()


def ids(result):
    return sorted(result.ids)


def F(key, value, op=FilterOperator.EQ):
    return MetadataFilter(key=key, value=value, operator=op)


CASES = {
    "eq": MetadataFilters(filters=[F("grp", "even")]),
    "eq_int": MetadataFilters(filters=[F("idx", 3)]),
    "eq_float": MetadataFilters(filters=[F("score", 2.5)]),
    "ne": MetadataFilters(filters=[F("grp", "even", FilterOperator.NE)]),
    "ne_missing_key": MetadataFilters(filters=[F("opt", "x", FilterOperator.NE)]),
    "gt": MetadataFilters(filters=[F("idx", 8, FilterOperator.GT)]),
    "gte": MetadataFilters(filters=[F("idx", 8, FilterOperator.GTE)]),
    "lt": MetadataFilters(filters=[F("score", 2.0, FilterOperator.LT)]),
    "lte": MetadataFilters(filters=[F("score", 2.0, FilterOperator.LTE)]),
    "in": MetadataFilters(filters=[F("idx", [1, 4, 9], FilterOperator.IN)]),
    "nin": MetadataFilters(filters=[F("grp", ["even"], FilterOperator.NIN)]),
    "any": MetadataFilters(filters=[F("tags", ["a", "z"], FilterOperator.ANY)]),
    "all": MetadataFilters(filters=[F("tags", ["b", "c"], FilterOperator.ALL)]),
    "contains": MetadataFilters(filters=[F("tags", "a", FilterOperator.CONTAINS)]),
    "text_match": MetadataFilters(filters=[F("name", "lpha", FilterOperator.TEXT_MATCH)]),
    "text_match_ci": MetadataFilters(filters=[F("name", "ALPHA", FilterOperator.TEXT_MATCH_INSENSITIVE)]),
    "is_empty": MetadataFilters(filters=[F("none_list", None, FilterOperator.IS_EMPTY)]),
    "is_empty_missing": MetadataFilters(filters=[F("opt", None, FilterOperator.IS_EMPTY)]),
    "and": MetadataFilters(filters=[F("grp", "odd"), F("idx", 5, FilterOperator.GT)]),
    "or": MetadataFilters(filters=[F("idx", 1), F("idx", 7)], condition=FilterCondition.OR),
    "not": MetadataFilters(filters=[F("grp", "even"), F("idx", 1)], condition=FilterCondition.NOT),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_filters_match_the_reference_store(store, name):
    ref = SimpleVectorStore()
    ns = nodes()
    ref.add(ns)
    store.add(ns)
    q = VectorStoreQuery(query_embedding=vec(100), similarity_top_k=len(ns), filters=CASES[name])
    expected = ids(ref.query(q))
    assert expected, f"case {name} selects nothing; the test would prove nothing"
    assert ids(store.query(q)) == expected


def test_nested_filter_groups(store):
    store.add(nodes())
    nested = MetadataFilters(
        filters=[
            F("grp", "even"),
            MetadataFilters(filters=[F("idx", 2), F("idx", 10)], condition=FilterCondition.OR),
        ],
        condition=FilterCondition.AND,
    )
    q = VectorStoreQuery(query_embedding=vec(0), similarity_top_k=12, filters=nested)
    assert ids(store.query(q)) == ["n10", "n2"]


def test_query_returns_nodes_text_and_cosine(store):
    ns = nodes()
    store.add(ns)
    res = store.query(VectorStoreQuery(query_embedding=vec(3), similarity_top_k=3))
    assert res.ids[0] == "n3" and res.similarities[0] == pytest.approx(1.0, abs=1e-6)
    assert res.nodes[0].text == "node text 3" and res.nodes[0].metadata["idx"] == 3
    assert res.similarities == sorted(res.similarities, reverse=True)


def test_node_and_doc_id_restrictions(store):
    ns = nodes()
    for n in ns[:4]:
        n.relationships = {}
    store.add(ns)
    res = store.query(VectorStoreQuery(query_embedding=vec(0), similarity_top_k=12, node_ids=["n1", "n2"]))
    assert ids(res) == ["n1", "n2"]


def test_get_delete_clear(store):
    store.add(nodes())
    assert sorted(n.node_id for n in store.get_nodes(node_ids=["n1", "n2", "ghost"])) == ["n1", "n2"]
    assert sorted(n.node_id for n in store.get_nodes(filters=MetadataFilters(filters=[F("idx", 9, FilterOperator.GTE)]))) == ["n10", "n11", "n9"]
    store.delete_nodes(node_ids=["n1"])
    store.delete_nodes(filters=MetadataFilters(filters=[F("grp", "even")]))
    assert sorted(n.node_id for n in store.get_nodes()) == ["n11", "n3", "n5", "n7", "n9"]
    with pytest.raises(ValueError):
        store.delete_nodes()
    store.clear()
    assert store.get_nodes() == []


def test_delete_by_ref_doc_id(store):
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo

    ns = nodes(6)
    for i, n in enumerate(ns):
        n.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=f"doc{i % 2}")
    store.add(ns)
    store.delete("doc0")
    assert sorted(n.node_id for n in store.get_nodes()) == ["n1", "n3", "n5"]


def test_unsupported_modes_are_rejected(store):
    store.add(nodes(3))
    for mode in (VectorStoreQueryMode.HYBRID, VectorStoreQueryMode.TEXT_SEARCH, VectorStoreQueryMode.MMR):
        with pytest.raises(ValueError):
            store.query(VectorStoreQuery(query_embedding=vec(0), query_str="x", mode=mode))


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_similarity_top_k_at_server_cap(server_factory, delta):
    srv = server_factory(DIM, max_top_k=10)
    s = HolographicVectorStore(url=srv.url, tenant_id=srv.tenant(), input_dim=DIM, max_fetch=10)
    s.add(nodes(12))
    k = 10 + delta
    if k > 10:
        with pytest.raises(ValueError):
            s.query(VectorStoreQuery(query_embedding=vec(0), similarity_top_k=k))
    else:
        assert len(s.query(VectorStoreQuery(query_embedding=vec(0), similarity_top_k=k)).ids) == k
    s.close()


def test_residual_filter_can_return_fewer_than_k(server_factory):
    srv = server_factory(DIM, max_top_k=10)
    s = HolographicVectorStore(url=srv.url, tenant_id=srv.tenant(), input_dim=DIM, max_fetch=10, overfetch=1)
    s.add(nodes(12))
    f = MetadataFilters(filters=[F("idx", 11, FilterOperator.GTE)])
    # overfetch=1: only the 3 nearest candidates are examined, so matches outside them are lost
    short = s.query(VectorStoreQuery(query_embedding=vec(0), similarity_top_k=3, filters=f))
    wide = dataclasses.replace(s._config, overfetch=4)
    s2 = HolographicVectorStore(config=wide)
    full = s2.query(VectorStoreQuery(query_embedding=vec(0), similarity_top_k=3, filters=f))
    assert len(short.ids) <= len(full.ids) == 1
    s.close(); s2.close()


async def test_async_paths(store):
    await store.async_add(nodes())
    res = await store.aquery(VectorStoreQuery(query_embedding=vec(5), similarity_top_k=1))
    assert res.ids == ["n5"]
    assert len(await store.aget_nodes(node_ids=["n1", "n2"])) == 2
    await store.adelete_nodes(filters=MetadataFilters(filters=[F("grp", "odd")]))
    assert len(await store.aget_nodes()) == 6
    await store.aclear()
    assert await store.aget_nodes() == []
    await store.aclose()


def test_index_roundtrip_with_retriever_and_query_engine(server_factory):
    srv = server_factory(8)
    embed = MockEmbedding(embed_dim=8)
    store = HolographicVectorStore(url=srv.url, tenant_id=srv.tenant(), input_dim=8)
    ctx = StorageContext.from_defaults(vector_store=store)
    docs = [Document(text=f"document number {i} about topic {i % 3}", doc_id=f"doc{i}", metadata={"grp": i % 2}) for i in range(6)]
    index = VectorStoreIndex.from_documents(docs, storage_context=ctx, embed_model=embed)
    retriever = index.as_retriever(similarity_top_k=2, filters=MetadataFilters(filters=[F("grp", 1)]))
    got = retriever.retrieve("document number 3")
    assert got and all(n.node.metadata["grp"] == 1 for n in got)
    reopened = VectorStoreIndex.from_vector_store(store, embed_model=embed)
    assert len(reopened.as_retriever(similarity_top_k=3).retrieve("anything")) == 3
    index.delete_ref_doc("doc1", delete_from_docstore=True)
    assert "doc1" not in {n.ref_doc_id for n in store.get_nodes()}
    store.close()
