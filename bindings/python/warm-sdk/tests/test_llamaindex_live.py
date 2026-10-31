"""LlamaIndex query contracts against the real WARM server."""

import uuid

import pytest

pytest.importorskip("llama_index.core")
from llama_index.core.bridge.pydantic import ValidationError
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition, FilterOperator, MetadataFilter, MetadataFilters,
    VectorStoreQuery, VectorStoreQueryMode,
)
from writerslogic_warm_sdk.llamaindex import WarmVectorStore
from writerslogic_warm_sdk.client import WarmError
from test_live_server import API_KEY, BIN, embedding, server

pytestmark = pytest.mark.skipif(not BIN, reason="WARM_SERVER_BIN not set")


@pytest.fixture
def store(server):
    value = WarmVectorStore(url=server, zero_trust_key="", api_key=API_KEY,
                            tenant_id=uuid.uuid4().hex)
    yield value
    if hasattr(value, "close"):
        value.close()


def test_filters_apply_before_top_k_and_preserve_types(store):
    store.add([
        TextNode(id_="private", text="private", embedding=embedding(1), metadata={"scope": "private", "n": 1}),
        TextNode(id_="public", text="public", embedding=embedding(2), metadata={"scope": "public", "n": 2}),
        TextNode(id_="string", text="string", embedding=embedding(1), metadata={"scope": "public", "n": "2"}),
    ])
    result = store.query(VectorStoreQuery(query_embedding=embedding(1), similarity_top_k=1,
        filters=MetadataFilters(filters=[MetadataFilter(key="scope", value="public"), MetadataFilter(key="n", value=2)])))
    assert result.ids == ["public"]
    assert result.nodes[0].metadata == {"scope": "public", "n": 2}
    assert result.nodes[0].text == "public"


@pytest.mark.parametrize("mode", [mode for mode in VectorStoreQueryMode if mode != VectorStoreQueryMode.DEFAULT])
def test_unsupported_modes_are_not_silently_dense_queries(store, mode):
    with pytest.raises(NotImplementedError):
        store.query(VectorStoreQuery(query_embedding=embedding(1), mode=mode))


@pytest.mark.parametrize("operator", [op for op in FilterOperator if op != FilterOperator.EQ])
def test_unsupported_operators_do_not_drop_filters(store, operator):
    with pytest.raises(NotImplementedError):
        store.query(VectorStoreQuery(query_embedding=embedding(1), filters=MetadataFilters(
            filters=[MetadataFilter(key="n", value=1, operator=operator)])))


@pytest.mark.parametrize("condition", [condition for condition in FilterCondition if condition != FilterCondition.AND])
def test_unsupported_conditions_do_not_broaden_results(store, condition):
    with pytest.raises(NotImplementedError):
        store.query(VectorStoreQuery(query_embedding=embedding(1), filters=MetadataFilters(
            filters=[MetadataFilter(key="n", value=1)], condition=condition)))


def test_nested_filters_are_rejected(store):
    try:
        filters = MetadataFilters(filters=[MetadataFilters(filters=[MetadataFilter(key="n", value=1)])])
    except ValidationError:
        return
    with pytest.raises(NotImplementedError):
        store.query(VectorStoreQuery(query_embedding=embedding(1), filters=filters))


@pytest.mark.parametrize("second", [1, 1.0, 2])
def test_duplicate_equalities_are_not_overwritten(store, second):
    with pytest.raises(ValueError):
        store.query(VectorStoreQuery(query_embedding=embedding(1), filters=MetadataFilters(
            filters=[MetadataFilter(key="n", value=1), MetadataFilter(key="n", value=second)])))


@pytest.mark.parametrize("field,value", [("doc_ids", []), ("node_ids", ["absent"]),
    ("alpha", 0.5), ("mmr_threshold", 0.5), ("output_fields", ["id"]),
    ("embedding_field", "custom"), ("sparse_top_k", 2), ("hybrid_top_k", 2)])
def test_unsupported_query_constraints_are_explicit(store, field, value):
    with pytest.raises(NotImplementedError):
        store.query(VectorStoreQuery(query_embedding=embedding(1), **{field: value}))


def test_client_lifetime_is_owned_and_close_is_idempotent(store):
    first = store.client
    assert store.client is first
    store.query(VectorStoreQuery(query_embedding=embedding(1)))
    store.close()
    store.close()
    assert first.client.is_closed
    with pytest.raises(RuntimeError):
        store.query(VectorStoreQuery(query_embedding=embedding(1)))


def test_reference_deletion_cannot_delete_an_unrelated_node_with_the_same_id(store):
    store.add([
        TextNode(id_="source-document", text="unrelated node", embedding=embedding(1)),
        TextNode(id_="chunk", text="source chunk", embedding=embedding(2)),
    ])
    with pytest.raises(NotImplementedError):
        store.delete("source-document")
    assert set(store.query(VectorStoreQuery(query_embedding=embedding(1), similarity_top_k=10)).ids) == {"source-document", "chunk"}


def test_explicit_node_deletion_preserves_other_nodes(store):
    ids = ["folder/node ?#%", "retained"]
    store.add([TextNode(id_=key, text=key, embedding=embedding(i)) for i, key in enumerate(ids)])
    store.delete_nodes([ids[0]])
    store.delete_nodes([ids[0], "absent"])
    assert store.query(VectorStoreQuery(query_embedding=embedding(1), similarity_top_k=10)).ids == ["retained"]


@pytest.mark.parametrize("node_ids", [None, "retained", ["retained", ""], ["retained", 1]])
def test_invalid_delete_ids_fail_before_any_deletion(store, node_ids):
    store.add([TextNode(id_="retained", text="retained", embedding=embedding(1))])
    with pytest.raises(ValueError):
        store.delete_nodes(node_ids)
    assert store.query(VectorStoreQuery(query_embedding=embedding(1))).ids == ["retained"]


def test_filtered_deletion_is_not_treated_as_unconditional(store):
    store.add([TextNode(id_="retained", text="retained", embedding=embedding(1), metadata={"scope": "private"})])
    with pytest.raises(NotImplementedError):
        store.delete_nodes(["retained"], filters=MetadataFilters(filters=[MetadataFilter(key="scope", value="public")]))
    assert store.query(VectorStoreQuery(query_embedding=embedding(1))).ids == ["retained"]



def test_node_deletion_propagates_authorization_failure(store):
    store.add([TextNode(id_="retained", text="retained", embedding=embedding(1))])
    store.client.client.headers["Authorization"] = "Bearer invalid"
    try:
        with pytest.raises(WarmError) as failure:
            store.delete_nodes(["retained"])
        assert failure.value.status_code == 401
    finally:
        store.client.client.headers["Authorization"] = f"Bearer {API_KEY}"
    assert store.query(VectorStoreQuery(query_embedding=embedding(1))).ids == ["retained"]
