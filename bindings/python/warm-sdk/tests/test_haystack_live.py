"""Haystack contracts against a real WARM server; no model or remote API calls."""
import uuid

import pytest

pytest.importorskip("haystack")
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from writerslogic_warm_sdk.haystack import WarmDocumentStore
from test_live_server import API_KEY, BIN, embedding, server

pytestmark = pytest.mark.skipif(not BIN, reason="WARM_SERVER_BIN not set")


@pytest.fixture
def store(server):
    value = WarmDocumentStore(url=server, api_key=API_KEY, tenant_id=uuid.uuid4().hex)
    yield value
    value.close()


def document(id, text="original"):
    return Document(id=id, content=text, embedding=embedding(1), meta={"kind": "note"})


def test_count_tracks_writes_replacements_deletes_and_tenant_isolation(store, server):
    other = WarmDocumentStore(url=server, api_key=API_KEY, tenant_id=uuid.uuid4().hex)
    try:
        assert store.count_documents() == 0
        assert store.write_documents([document("one"), document("two")]) == 2
        assert store.count_documents() == 2
        assert other.count_documents() == 0
        assert store.write_documents([document("one", "replacement")], policy=DuplicatePolicy.OVERWRITE) == 1
        assert store.count_documents() == 2
        store.delete_documents(["one"])
        assert store.count_documents() == 1
    finally:
        other.close()


@pytest.mark.parametrize("policy", [DuplicatePolicy.FAIL, DuplicatePolicy.SKIP])
def test_unsupported_duplicate_policies_reject_before_any_write(store, policy):
    store.write_documents([document("existing")])
    with pytest.raises(NotImplementedError):
        store.write_documents([document("new"), document("existing", "replacement")], policy=policy)
    matches = store._client.query(embedding(1), top_k=10)["matches"]
    assert [(m["id"], m["text"]) for m in matches] == [("existing", "original")]


@pytest.mark.parametrize("filters", [None, {}, {"field": "meta.kind", "operator": "==", "value": "note"}])
def test_filtering_is_explicitly_unsupported_instead_of_fabricating_absence(store, filters):
    store.write_documents([document("present")])
    with pytest.raises(NotImplementedError):
        store.filter_documents(filters)


@pytest.mark.parametrize("policy", ["overwrite", 1, False])
def test_invalid_policy_cannot_silently_overwrite(store, policy):
    store.write_documents([document("existing")])
    with pytest.raises(TypeError):
        store.write_documents([document("existing", "replacement")], policy=policy)
    assert store._client.query(embedding(1))["matches"][0]["text"] == "original"


def test_document_writer_uses_explicit_overwrite(store):
    from haystack.components.writers import DocumentWriter
    writer = DocumentWriter(document_store=store, policy=DuplicatePolicy.OVERWRITE)
    assert writer.run(documents=[document("one")]) == {"documents_written": 1}
    assert writer.run(documents=[document("one", "new")]) == {"documents_written": 1}
    assert store.count_documents() == 1
    assert store._client.query(embedding(1))["matches"][0]["text"] == "new"


def test_stats_authentication_errors_are_not_hidden_as_zero(store, server):
    from writerslogic_warm_sdk.client import WarmError
    bad = WarmDocumentStore(url=server, api_key="wrong")
    try:
        with pytest.raises(WarmError) as error:
            bad.count_documents()
        assert error.value.status_code == 401
    finally:
        bad.close()


def test_async_stats_preserve_tenant_scope_and_authentication(server):
    import asyncio
    from writerslogic_warm_sdk.client import AsyncWarmClient, WarmError
    async def run():
        tenant = uuid.uuid4().hex
        async with AsyncWarmClient(url=server, api_key=API_KEY, tenant_id=tenant) as client:
            assert (await client.stats())["documents"] == 0
            await client.add_documents([{"id": "one", "vector": embedding(1)}])
            assert (await client.stats())["documents"] == 1
        async with AsyncWarmClient(url=server, api_key=API_KEY, tenant_id=uuid.uuid4().hex) as other:
            assert (await other.stats())["documents"] == 0
        async with AsyncWarmClient(url=server, api_key="wrong", tenant_id=tenant) as bad:
            with pytest.raises(WarmError) as error:
                await bad.stats()
            assert error.value.status_code == 401
    asyncio.run(run())
