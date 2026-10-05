import asyncio
import json
import math

import httpx
import pytest

from holographic_sdk import AsyncHolographicClient, HolographicClient, VectorMask

SALT = b"0123456789abcdef"


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_mask_preserves_inner_products_and_hides_layout():
    mask = VectorMask("passphrase", SALT)
    a = [math.sin(i) for i in range(64)]
    b = [math.cos(i * 0.7) for i in range(64)]
    ma, mb = mask.apply(a), mask.apply(b)

    assert dot(ma, mb) == pytest.approx(dot(a, b), abs=1e-12)
    assert sorted(abs(x) for x in ma) == sorted(abs(x) for x in a)
    assert ma != a
    assert VectorMask("passphrase", SALT).apply(a) == ma
    assert VectorMask("other", SALT).apply(a) != ma
    assert VectorMask("passphrase", b"fedcba9876543210").apply(a) != ma


def test_mask_rejects_weak_parameters():
    with pytest.raises(ValueError):
        VectorMask("", SALT)
    with pytest.raises(ValueError):
        VectorMask("passphrase", SALT[:15])
    VectorMask("passphrase", SALT[:16])


def recording_transport(requests, batch_status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/documents/batch"):
            return httpx.Response(batch_status)
        if request.url.path.endswith("/query"):
            return httpx.Response(200, json={"matches": []})
        return httpx.Response(200)

    return httpx.MockTransport(handler)


def test_documents_are_masked_without_mutating_caller_and_headers_sent():
    requests = []
    vector = [0.25, -0.5, 0.75, 1.0]
    docs = [{"id": "a", "text": "t", "vector": list(vector)}, {"id": "b", "text": "no vector"}]
    with HolographicClient(
        zero_trust_key="passphrase",
        mask_salt=SALT,
        api_key="token",
        tenant_id="tenant",
        transport=recording_transport(requests),
    ) as client:
        client.add_documents(docs)
        client.query(vector, top_k=2, filter={"k": "v"})

    assert docs[0]["vector"] == vector
    sent = json.loads(requests[0].content)
    expected = VectorMask("passphrase", SALT).apply(vector)
    assert sent[0]["vector"] == expected and sent[0]["vector"] != vector
    assert "vector" not in sent[1]
    assert requests[0].headers["authorization"] == "Bearer token"
    assert requests[0].headers["x-tenant-id"] == "tenant"
    assert json.loads(requests[1].content) == {
        "query_vector": expected,
        "top_k": 2,
        "filter": {"k": "v"},
    }


def test_without_key_vectors_pass_through_and_batch_404_falls_back():
    requests = []
    client = HolographicClient(transport=recording_transport(requests, batch_status=404))
    client.add_documents([{"id": "a", "vector": [1.0, 2.0]}, {"id": "b", "vector": [3.0, 4.0]}])
    paths = [r.url.path for r in requests]
    assert paths == ["/api/v1/documents/batch", "/api/v1/documents", "/api/v1/documents"]
    assert json.loads(requests[1].content)["vector"] == [1.0, 2.0]


def test_server_errors_raise():
    client = HolographicClient(transport=recording_transport([], batch_status=500))
    with pytest.raises(httpx.HTTPStatusError):
        client.add_documents([{"id": "a"}])


def test_async_client_matches_sync_masking():
    requests = []

    async def run():
        async with AsyncHolographicClient(
            zero_trust_key="passphrase", mask_salt=SALT, transport=recording_transport(requests)
        ) as client:
            await client.add_documents([{"id": "a", "vector": [1.0, 2.0, 3.0]}])

    asyncio.run(run())
    assert json.loads(requests[0].content)[0]["vector"] == VectorMask("passphrase", SALT).apply(
        [1.0, 2.0, 3.0]
    )
