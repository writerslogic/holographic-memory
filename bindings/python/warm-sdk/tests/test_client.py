import asyncio
import json
import math

import httpx
import pytest

from writerslogic_warm_sdk import AsyncWarmClient, WarmClient, WarmError, VectorMask

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
    with WarmClient(
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
    client = WarmClient(transport=recording_transport(requests, batch_status=404))
    client.add_documents([{"id": "a", "vector": [1.0, 2.0]}, {"id": "b", "vector": [3.0, 4.0]}])
    paths = [r.url.path for r in requests]
    assert paths == ["/api/v1/documents/batch", "/api/v1/documents", "/api/v1/documents"]
    assert json.loads(requests[1].content)["vector"] == [1.0, 2.0]


def test_server_errors_raise():
    client = WarmClient(transport=recording_transport([], batch_status=500))
    with pytest.raises(httpx.HTTPStatusError):
        client.add_documents([{"id": "a"}])


def error_transport(status, **body):
    return httpx.MockTransport(lambda request: httpx.Response(status, **body))


def test_service_errors_raise_holographic_error():
    body = {"error": {"code": "invalid_request", "message": "top_k must be at least 1"}}
    client = WarmClient(transport=error_transport(422, json=body))
    with pytest.raises(WarmError) as raised:
        client.query([1.0], top_k=0)
    error = raised.value
    assert (error.status_code, error.code, error.message) == (422, "invalid_request", "top_k must be at least 1")
    assert str(error) == "top_k must be at least 1"
    # still an HTTPStatusError, which the importer's retries catch
    assert isinstance(error, httpx.HTTPStatusError)
    assert error.response.status_code == 422


def test_errors_without_the_service_shape_stay_http_status_errors():
    for body in ({"text": "Bad Gateway"}, {"json": {"error": "no code"}}, {"json": {"detail": "x"}}):
        client = WarmClient(transport=error_transport(502, **body))
        with pytest.raises(httpx.HTTPStatusError) as raised:
            client.delete_document("a")
        assert not isinstance(raised.value, WarmError)


def test_async_client_raises_holographic_error():
    body = {"error": {"code": "tenant_limit", "message": "too many open tenants"}}

    async def run():
        async with AsyncWarmClient(transport=error_transport(429, json=body)) as client:
            await client.maintain_document_index()

    with pytest.raises(WarmError) as raised:
        asyncio.run(run())
    assert (raised.value.status_code, raised.value.code) == (429, "tenant_limit")


def document_transport(requests):
    """Answers DELETE /api/v1/documents/a%20b%2Fc and POST /api/v1/index/maintain; anything else is a 404."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "DELETE" and request.url.raw_path == b"/api/v1/documents/a%20b%2Fc":
            return httpx.Response(200, json={"deleted": True})
        if request.method == "POST" and request.url.path == "/api/v1/index/maintain":
            return httpx.Response(200, json={"rebuilt": True, "documents": 3})
        return httpx.Response(404, text="not found")

    return httpx.MockTransport(handler)


def test_delete_document_percent_encodes_the_id():
    requests = []
    with WarmClient(transport=document_transport(requests)) as client:
        client.delete_document("a b/c")
        with pytest.raises(httpx.HTTPStatusError) as raised:
            client.delete_document("missing")
    assert [(r.method, r.url.raw_path) for r in requests] == [
        ("DELETE", b"/api/v1/documents/a%20b%2Fc"),
        ("DELETE", b"/api/v1/documents/missing"),
    ]
    assert raised.value.response.status_code == 404


def test_maintain_document_index_posts_an_empty_object():
    requests = []
    with WarmClient(transport=document_transport(requests)) as client:
        assert client.maintain_document_index() == {"rebuilt": True, "documents": 3}
    assert requests[0].method == "POST"
    assert json.loads(requests[0].content) == {}


def test_async_delete_and_maintain():
    requests = []

    async def run():
        async with AsyncWarmClient(transport=document_transport(requests)) as client:
            await client.delete_document("a b/c")
            with pytest.raises(httpx.HTTPStatusError) as raised:
                await client.delete_document("missing")
            assert raised.value.response.status_code == 404
            return await client.maintain_document_index()

    assert asyncio.run(run()) == {"rebuilt": True, "documents": 3}
    assert [(r.method, r.url.raw_path) for r in requests] == [
        ("DELETE", b"/api/v1/documents/a%20b%2Fc"),
        ("DELETE", b"/api/v1/documents/missing"),
        ("POST", b"/api/v1/index/maintain"),
    ]
    assert json.loads(requests[2].content) == {}


def test_async_client_matches_sync_masking():
    requests = []

    async def run():
        async with AsyncWarmClient(
            zero_trust_key="passphrase", mask_salt=SALT, transport=recording_transport(requests)
        ) as client:
            await client.add_documents([{"id": "a", "vector": [1.0, 2.0, 3.0]}])

    asyncio.run(run())
    assert json.loads(requests[0].content)[0]["vector"] == VectorMask("passphrase", SALT).apply(
        [1.0, 2.0, 3.0]
    )
