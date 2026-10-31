"""Bounded retained-chunk reads against a local server."""
import asyncio
import uuid

import pytest

from writerslogic_warm_sdk import AsyncWarmClient, WarmClient, WarmError
from test_live_server import API_KEY, BIN, embedding, server

pytestmark = pytest.mark.skipif(not BIN, reason="WARM_SERVER_BIN not set")


def test_chunk_read_preserves_stored_text_and_tenant_scope(server):
    tenant = uuid.uuid4().hex
    id = "folder/doc #1?"
    with WarmClient(url=server, api_key=API_KEY, tenant_id=tenant) as client:
        client.add_documents([{"id": id, "text": "  café🙂\tβ  ", "vector": embedding(1), "metadata": {"kind": "note"}}])
        chunk = client.read_document_chunk(id)
        assert chunk["document"]["id"] == id
        assert chunk["document"]["metadata"] == {"kind": "note"}
        assert chunk["document"]["chunkCount"] == 1
        assert isinstance(chunk["revision"], str)
        assert client.read_document_chunk(id)["revision"] == chunk["revision"]
        client.add_documents([{"id": id, "text": "replacement", "vector": embedding(1)}])
        updated = client.read_document_chunk(id)
        assert updated["revision"] != chunk["revision"]
        assert updated["document"]["version"] != chunk["document"]["version"]
        assert updated["text"] == "replacement"
        assert (chunk["index"], chunk["startByte"], chunk["endByte"], chunk["text"]) == (0, 2, 14, "café🙂\tβ")
        for index in [1, 2]:
            with pytest.raises(WarmError) as error:
                client.read_document_chunk(id, index)
            assert error.value.status_code == 404
        for index in [-1, True, 0.5, "0"]:
            with pytest.raises(ValueError):
                client.read_document_chunk(id, index)
        with WarmClient(url=server, api_key=API_KEY, tenant_id=uuid.uuid4().hex) as other:
            with pytest.raises(WarmError) as error:
                other.read_document_chunk(id)
            assert error.value.status_code == 404
        client.add_documents([{"id": "vector-only", "vector": embedding(2)}])
        assert client.read_document_chunk("vector-only")["text"] is None
        client.delete_document(id)
        with pytest.raises(WarmError) as error:
            client.read_document_chunk(id)
        assert (error.value.status_code, error.value.code) == (404, "not_found")


def test_async_chunk_read_preserves_auth_and_absence(server):
    async def run():
        tenant = uuid.uuid4().hex
        async with AsyncWarmClient(url=server, api_key=API_KEY, tenant_id=tenant) as client:
            await client.add_documents([{"id": "a/b", "text": "passage", "vector": embedding(1)}])
            assert (await client.read_document_chunk("a/b"))["text"] == "passage"
            with pytest.raises(ValueError):
                await client.read_document_chunk("a/b", -1)
            with pytest.raises(WarmError) as error:
                await client.read_document_chunk("missing")
            assert error.value.status_code == 404
        async with AsyncWarmClient(url=server, api_key="wrong", tenant_id=tenant) as bad:
            with pytest.raises(WarmError) as error:
                await bad.read_document_chunk("a/b")
            assert error.value.status_code == 401
    asyncio.run(run())
