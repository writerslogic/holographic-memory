# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import asyncio

import httpx
import pytest

from holographic_integrations.core import AsyncHmsBackend, HmsBackend, HmsConfig, HmsError


def transport(supported, response_kind="valid"):
    requests = []
    metadata = {"_hms_collection": "A", "_hms_id": "selected", "n": 10}

    def handle(request):
        requests.append((request.method, request.url.path))
        if request.url.path == "/openapi.json":
            properties = {"document_ids": {"type": "array"}} if supported else {}
            return httpx.Response(200, json={"components": {"schemas": {"QueryIn": {"properties": properties}}}})
        if request.url.path == "/api/v1/documents":
            return httpx.Response(200, json={"documents": [{"id": "A/selected", "metadata": metadata}], "next_cursor": None})
        assert request.url.path == "/api/v1/query"
        hit = {"id": "A/selected", "metadata": dict(metadata), "score": 1.0, "text": "selected text"}
        if response_kind == "unselected_id":
            hit["id"] = "A/unselected"
            hit["metadata"]["_hms_id"] = "unselected"
        elif response_kind == "other_collection":
            hit["metadata"]["_hms_collection"] = "B"
        elif response_kind == "changed_identity":
            hit["metadata"]["_hms_id"] = "changed"
        elif response_kind == "malformed":
            return httpx.Response(200, content=b"invalid JSON")
        return httpx.Response(200, json={"matches": [hit]})

    return httpx.MockTransport(handle), requests


def run_case(asynchronous, supported, response_kind, operation):
    wire, requests = transport(supported, response_kind)
    config = HmsConfig(url="http://capability.invalid", input_dim=8, collection="A", transport=wire)
    if asynchronous:
        async def run():
            async with AsyncHmsBackend(config) as backend:
                return await operation(backend, True)
        asyncio.run(run())
    else:
        with HmsBackend(config) as backend:
            asyncio.run(operation(backend, False))
    return requests


@pytest.mark.parametrize("asynchronous", [False, True])
def test_old_servers_fail_closed_and_cache_the_capability_result(asynchronous):
    async def operation(backend, async_backend):
        for _ in range(2):
            with pytest.raises(HmsError) as error:
                result = backend.recall("alpha", filter={"n": {"$gt": 1}}, k=1)
                if async_backend:
                    await result
            assert error.value.code == "unsupported_capability"

    requests = run_case(asynchronous, False, "valid", operation)
    assert requests == [("GET", "/openapi.json")]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_direct_equality_queries_stay_compatible_without_capability_discovery(asynchronous):
    async def operation(backend, async_backend):
        result = backend.recall("alpha", filter={"n": 10}, k=1)
        hits = await result if async_backend else result
        assert [hit.id for hit in hits] == ["selected"]

    assert run_case(asynchronous, False, "valid", operation) == [("POST", "/api/v1/query")]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_supported_id_selection_discovers_the_capability_once(asynchronous):
    async def operation(backend, async_backend):
        for _ in range(2):
            result = backend.recall("alpha", filter={"n": {"$gt": 1}}, k=1)
            hits = await result if async_backend else result
            assert [hit.id for hit in hits] == ["selected"]

    requests = run_case(asynchronous, True, "valid", operation)
    assert requests.count(("GET", "/openapi.json")) == 1
    assert requests.count(("POST", "/api/v1/query")) == 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("response_kind", ["unselected_id", "other_collection", "changed_identity", "malformed"])
def test_id_selection_rejects_responses_outside_its_verified_population(asynchronous, response_kind):
    async def operation(backend, async_backend):
        with pytest.raises(HmsError) as error:
            result = backend.recall("alpha", filter={"n": {"$gt": 1}}, k=1)
            if async_backend:
                await result
        assert error.value.code == "invalid_response"

    run_case(asynchronous, True, response_kind, operation)
