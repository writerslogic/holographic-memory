# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np
import pytest

pytest.importorskip("semantic_kernel")

from semantic_kernel import Kernel  # noqa: E402
from semantic_kernel.connectors.ai.embedding_generator_base import EmbeddingGeneratorBase  # noqa: E402
from semantic_kernel.data.vector import (  # noqa: E402
    DistanceFunction,
    VectorStoreField,
    vectorstoremodel,
)
from semantic_kernel.exceptions import VectorStoreModelValidationError  # noqa: E402
from semantic_kernel.exceptions.vector_store_exceptions import VectorStoreOperationException  # noqa: E402

from holographic_integrations.core import HashingEmbedder, HmsConfig  # noqa: E402
from holographic_integrations.semantic_kernel import HolographicCollection, HolographicStore  # noqa: E402

DIM = 64


@vectorstoremodel
@dataclass
class Doc:
    id: Annotated[str, VectorStoreField("key")]
    text: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    team: Annotated[str, VectorStoreField("data", is_indexed=True)]
    rank: Annotated[int, VectorStoreField("data", is_indexed=True)]
    embedding: Annotated[list[float] | str | None, VectorStoreField("vector", dimensions=DIM)] = None


@vectorstoremodel
@dataclass
class DistDoc:
    id: Annotated[str, VectorStoreField("key")]
    text: Annotated[str, VectorStoreField("data")]
    embedding: Annotated[
        list[float] | None,
        VectorStoreField("vector", dimensions=DIM, distance_function=DistanceFunction.COSINE_SIMILARITY),
    ] = None


class SkEmbedder(EmbeddingGeneratorBase):
    """A Semantic Kernel embedding service backed by the offline hashing encoder."""

    async def generate_embeddings(self, texts: list[str], settings: Any = None, **kwargs: Any) -> np.ndarray:
        return np.array(HashingEmbedder(DIM).embed_documents(texts))


def store(server, **kw):
    return HolographicStore(HmsConfig(url=server.url, tenant_id=server.tenant(), input_dim=DIM, embedder=HashingEmbedder(DIM), **kw))


def docs(n=6):
    return [
        Doc(id=f"d{i}", text=f"alpha beta gamma number{i} topic{i % 3}", team="a" if i % 2 == 0 else "b", rank=i)
        for i in range(n)
    ]


async def collect(results):
    return [r async for r in results.results]


async def test_upsert_get_delete(server):
    col = store(server).get_collection(Doc, collection_name="docs")
    keys = await col.upsert(docs())
    assert keys == [f"d{i}" for i in range(6)]
    one = await col.get("d3")
    assert isinstance(one, Doc) and (one.text, one.team, one.rank) == ("alpha beta gamma number3 topic0", "b", 3)
    many = await col.get(["d1", "nope", "d2"])
    assert sorted(d.id for d in many) == ["d1", "d2"]
    assert await col.get("nope") is None
    assert await col.upsert(docs(1)[0]) == "d0"  # single record in, single key out
    await col.delete(["d0", "ghost"])
    assert await col.get("d0") is None
    await col.delete("d1")
    assert await col.get(["d1"]) is None


async def test_get_without_keys_orders_and_pages(server):
    col = store(server).get_collection(Doc, collection_name="docs")
    await col.upsert(docs())
    top = await col.get(top=3, order_by={"rank": False})
    assert [d.rank for d in top] == [5, 4, 3]
    page = await col.get(top=2, skip=2, order_by="rank")
    assert [d.rank for d in page] == [2, 3]


async def test_vector_search_with_values_and_scores(server):
    col = store(server).get_collection(Doc, collection_name="docs")
    await col.upsert(docs())
    res = await collect(await col.search(values="alpha beta gamma number4 topic1", top=2))
    assert res[0].record.id == "d4" and len(res) == 2
    assert res[0].score <= res[1].score  # DEFAULT is a distance: lower is better
    assert 0 <= res[0].score < 0.5
    paged = await collect(await col.search(values="alpha beta gamma number4 topic1", top=1, skip=1))
    assert paged[0].record.id == res[1].record.id


async def test_cosine_similarity_orientation(server):
    col = store(server).get_collection(DistDoc, collection_name="sim")
    await col.upsert([DistDoc(id="x", text="uniquewordone only"), DistDoc(id="y", text="different content here")])
    res = await collect(await col.search(values="uniquewordone only", top=2))
    assert res[0].record.id == "x" and res[0].score > res[1].score
    assert res[0].score > 0.99


async def test_filters_lambda_and_string(server):
    col = store(server).get_collection(Doc, collection_name="docs")
    await col.upsert(docs(12))
    q = "alpha beta gamma"
    ids = lambda rs: {r.record.id for r in rs}  # noqa: E731
    r = await collect(await col.search(values=q, top=12, filter="lambda x: x.team == 'a'"))
    assert ids(r) == {f"d{i}" for i in range(0, 12, 2)}
    r = await collect(await col.search(values=q, top=12, filter="lambda x: x.rank >= 9"))
    assert ids(r) == {"d9", "d10", "d11"}
    r = await collect(await col.search(values=q, top=12, filter="lambda x: x.team == 'b' and x.rank < 6"))
    assert ids(r) == {"d1", "d3", "d5"}
    r = await collect(await col.search(values=q, top=12, filter="lambda x: x.rank in [1, 2] or not x.team == 'a'"))
    assert ids(r) == {"d1", "d2", "d3", "d5", "d7", "d9", "d11"}
    r = await collect(await col.search(values=q, top=12, filter=["lambda x: x.team == 'a'", "lambda x: 4 <= x.rank"]))
    assert ids(r) == {"d4", "d6", "d8", "d10"}
    r = await collect(await col.search(values=q, top=12, filter=lambda x: x.team == "b"))
    assert ids(r) == {"d1", "d3", "d5", "d7", "d9", "d11"}


async def test_filter_errors(server):
    col = store(server).get_collection(Doc, collection_name="docs")
    for bad in ("lambda x: x.text == 'a'", "lambda x: x.nothing == 1", "lambda x: 1 < x.rank < 3", "lambda x: x.team.upper() == 'A'"):
        with pytest.raises(VectorStoreOperationException):
            await col.search(values="q", filter=bad)


async def test_filter_applies_before_ranking_past_the_cap(server_factory):
    srv = server_factory(DIM, max_top_k=5)
    cfg = HmsConfig(url=srv.url, tenant_id=srv.tenant(), input_dim=DIM, embedder=HashingEmbedder(DIM), max_fetch=5)
    col = HolographicStore(cfg).get_collection(Doc, collection_name="docs")
    await col.upsert([Doc(id=f"n{i}", text=f"alpha match {i}", team="no", rank=i) for i in range(12)])
    await col.upsert([Doc(id="needle", text="zzz rare needle", team="yes", rank=99)])
    res = await collect(await col.search(values="alpha match", top=3, filter="lambda x: x.team == 'yes'"))
    assert [r.record.id for r in res] == ["needle"]
    with pytest.raises(Exception, match="max_fetch"):
        await col.search(values="alpha", top=4, skip=2)


async def test_collections_are_isolated_in_a_tenant(server):
    st = store(server)
    a, b = st.get_collection(Doc, collection_name="a"), st.get_collection(Doc, collection_name="b")
    await a.upsert([Doc(id="same", text="from a", team="x", rank=1)])
    await b.upsert([Doc(id="same", text="from b", team="x", rank=2)])
    assert (await a.get("same")).text == "from a" and (await b.get("same")).text == "from b"
    assert await st.list_collection_names() == ["a", "b"]
    await st.ensure_collection_deleted("a")
    assert await a.get("same") is None and (await b.get("same")).rank == 2
    assert await st.list_collection_names() == ["b"]


async def test_sk_embedding_generator_and_vector_values(server):
    st = HolographicStore(
        HmsConfig(url=server.url, tenant_id=server.tenant(), input_dim=DIM, embedder=HashingEmbedder(DIM)),
        embedding_generator=SkEmbedder(service_id="e", ai_model_id="hash"),
    )
    col = st.get_collection(Doc, collection_name="g")
    await col.upsert([Doc(id=f"g{i}", text=f"alpha number{i}", team="t", rank=i, embedding=f"alpha number{i}") for i in range(4)])
    res = await collect(await col.search(values="alpha number2", top=1))
    assert res[0].record.id == "g2"
    vec = HashingEmbedder(DIM).embed_query("alpha number3")
    res = await collect(await col.search(vector=vec, top=1))
    assert res[0].record.id == "g3"


async def test_kernel_search_function_end_to_end(server):
    col = store(server).get_collection(Doc, collection_name="kernel")
    await col.upsert(docs())
    kernel = Kernel()
    fn = col.create_search_function(
        function_name="search_docs", description="Search docs", top=2,
        string_mapper=lambda r: f"{r.record.id}:{r.record.text}",
    )
    kernel.add_function(plugin_name="memory", function=fn)
    out = await kernel.invoke(plugin_name="memory", function_name="search_docs", query="alpha beta gamma number5 topic2")
    assert str(out.value[0]).startswith("d5:") and len(out.value) == 2


async def test_model_validation_and_hybrid_gate(server):
    @vectorstoremodel
    @dataclass
    class TwoVec:
        id: Annotated[str, VectorStoreField("key")]
        text: Annotated[str, VectorStoreField("data")]
        v1: Annotated[list[float] | None, VectorStoreField("vector", dimensions=DIM)] = None
        v2: Annotated[list[float] | None, VectorStoreField("vector", dimensions=DIM)] = None

    @vectorstoremodel
    @dataclass
    class Dot:
        id: Annotated[str, VectorStoreField("key")]
        text: Annotated[str, VectorStoreField("data")]
        v: Annotated[list[float] | None, VectorStoreField("vector", dimensions=DIM, distance_function=DistanceFunction.DOT_PROD)] = None

    st = store(server)
    with pytest.raises(VectorStoreModelValidationError):
        st.get_collection(TwoVec, collection_name="t")
    with pytest.raises(VectorStoreModelValidationError):
        st.get_collection(Dot, collection_name="t")
    col = st.get_collection(Doc, collection_name="h")
    with pytest.raises(Exception, match="text_queries"):
        await col.hybrid_search(values="alpha")


async def test_isinstance_of_sk_types(server):
    from semantic_kernel.data.vector import VectorSearchProtocol, VectorStore, VectorStoreCollection

    st = store(server)
    col = st.get_collection(Doc, collection_name="p")
    assert isinstance(st, VectorStore) and isinstance(col, VectorStoreCollection)
    assert isinstance(col, HolographicCollection) and isinstance(col, VectorSearchProtocol)
