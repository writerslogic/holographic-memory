# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
import asyncio
import datetime as dt
import json

import httpx
import pytest

from holographic_sdk import AsyncHolographicClient, HolographicClient
from holographic_sdk import memory as memory_module
from holographic_sdk.memory import (
    AsyncMemoryClient,
    MemoryClient,
    MemoryError,
    to_datetime,
    to_millis,
)

SALT = b"0123456789abcdef"

REMEMBERED = {"id": "f1", "created": True, "supersedes": ["f0"], "supersededBy": [], "ingestTime": 5}
RECALLED = {
    "id": "f1", "text": "Alice lives in Berlin", "score": 1.5, "status": "current", "ingestTime": 5,
    "eventTime": None, "validFrom": 100, "validTo": None, "subject": "alice/city", "entities": ["Alice"],
    "supersedes": ["f0"], "supersededBy": [], "metadata": {"src": "chat"}, "sourceUri": None,
    "explanation": {
        "retrieved": True, "rank": 1, "lexicalScore": 2.0, "semanticScore": None, "retrievalScore": 1.5,
        "lexicalWeight": 1.0, "semanticWeight": 1.0, "fusion": "blend", "matchedTerms": ["alice"],
        "recencyFactor": 1.0, "historicalFactor": 1.0, "finalScore": 1.5, "status": "current", "asOf": 9,
        "knownAt": None, "validFrom": 100, "validTo": None, "withinValidity": True,
        "chain": [{"id": "f0", "relation": "predecessor", "effectiveFrom": 1, "ingestTime": 2}],
    },
}
FACT = {
    "id": "f0", "text": "Alice lives in Paris", "status": "superseded", "ingestTime": 2, "eventTime": None,
    "validFrom": 1, "validTo": None, "effectiveFrom": 1, "subject": "alice/city", "entities": [],
    "supersedes": [], "supersededBy": ["f1"], "metadata": {},
}


def transport(requests, error=None):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if error:
            return httpx.Response(error[0], json={"error": {"code": error[1], "message": error[2]}})
        op = request.url.path.rsplit("/", 1)[1]
        body = {
            "remember": {"remembered": [REMEMBERED]},
            "recall": {"results": [RECALLED]},
            "explain": {"explanation": RECALLED["explanation"]},
            "forget": {"forgotten": ["f0"], "dryRun": True, "purged": False},
            "get": {"fact": FACT},
            "history": {"history": [FACT]},
            "applyForgetting": {"forgotten": [], "dryRun": True, "purged": False},
            "consolidate": {"created": [REMEMBERED], "forgotten": [], "dryRun": True},
        }[op]
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


def body(request: httpx.Request):
    return json.loads(request.content)


def test_times_convert_to_milliseconds_and_back():
    assert to_millis(1500) == 1500
    assert to_millis(None) is None
    assert to_millis(dt.datetime(1970, 1, 1, 0, 0, 1, 500_000, tzinfo=dt.timezone.utc)) == 1500
    assert to_millis(dt.datetime(1970, 1, 1, 0, 0, 1)) == 1000  # naive means UTC
    assert to_millis(dt.datetime(1970, 1, 1, 1, tzinfo=dt.timezone(dt.timedelta(hours=1)))) == 0
    assert to_datetime(86_400_000) == dt.datetime(1970, 1, 2, tzinfo=dt.timezone.utc)
    with pytest.raises(TypeError):
        to_millis(True)
    with pytest.raises(TypeError):
        to_millis("yesterday")  # type: ignore[arg-type]


def test_remember_sends_camel_case_and_drops_unset_fields():
    requests = []
    with MemoryClient(HolographicClient(transport=transport(requests))) as memory:
        out = memory.remember(
            "Alice lives in Berlin",
            id="f1",
            valid_from=dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(milliseconds=100),
            subject="alice/city",
            supersedes=["f0"],
            entities=("Alice",),
            metadata={"src": "chat"},
        )
    assert out.id == "f1" and out.created and out.supersedes == ["f0"] and out.ingest_time == 5
    request = requests[0]
    assert request.url.path == "/api/v1/memory/remember"
    assert body(request) == {
        "id": "f1", "text": "Alice lives in Berlin", "validFrom": 100, "subject": "alice/city",
        "supersedes": ["f0"], "entities": ["Alice"], "metadata": {"src": "chat"},
    }


def test_remember_many_wraps_facts_and_masks_embeddings():
    requests = []
    client = HolographicClient(zero_trust_key="passphrase", mask_salt=SALT, transport=transport(requests))
    memory = MemoryClient(client)
    vector = [0.5, -0.25, 1.0, 0.0]
    memory.remember_many([{"text": "a", "embedding": vector, "event_time": 7}, {"text": "b"}])
    sent = body(requests[0])["facts"]
    assert sent[0]["eventTime"] == 7
    assert sent[0]["embedding"] != vector
    assert sorted(abs(x) for x in sent[0]["embedding"]) == sorted(abs(x) for x in vector)
    assert "embedding" not in sent[1]
    client.close()


def test_recall_parses_results_and_explanation():
    requests = []
    memory = MemoryClient(HolographicClient(transport=transport(requests)))
    results = memory.recall(
        "where does Alice live",
        k=3,
        as_of=dt.datetime(1970, 1, 1, 0, 0, 2, tzinfo=dt.timezone.utc),
        history="all",
        entities=["Alice"],
        explain=True,
    )
    assert body(requests[0]) == {
        "query": "where does Alice live", "k": 3, "asOf": 2000, "history": "all",
        "entities": ["Alice"], "explain": True,
    }
    top = results[0]
    assert top.status == "current" and top.superseded_by == [] and top.metadata == {"src": "chat"}
    assert top.explanation is not None
    assert top.explanation.matched_terms == ["alice"]
    assert top.explanation.chain[0].relation == "predecessor"
    assert top.explanation.semantic_score is None


def test_explain_forget_get_history_and_maintenance():
    requests = []
    memory = MemoryClient(HolographicClient(transport=transport(requests)))
    explained = memory.explain("alice", "f1", as_of=9)
    assert body(requests[-1]) == {"query": "alice", "asOf": 9, "id": "f1"}
    assert explained.within_validity and explained.final_score == 1.5

    report = memory.forget(ids=["f0"], include_chain=True, dry_run=True)
    assert body(requests[-1]) == {
        "scope": {"kind": "ids", "ids": ["f0"]}, "includeChain": True, "dryRun": True, "purge": False,
    }
    assert report.forgotten == ["f0"] and report.dry_run
    memory.forget(ingested_before=dt.datetime(1970, 1, 1, 0, 0, 1, tzinfo=dt.timezone.utc))
    assert body(requests[-1])["scope"] == {"kind": "ingestedBefore", "time": 1000}
    memory.forget(matching={"tenant": "t1"})
    assert body(requests[-1])["scope"] == {"kind": "matching", "filter": {"tenant": "t1"}}
    with pytest.raises(ValueError):
        memory.forget()
    with pytest.raises(ValueError):
        memory.forget(subject="a", entity="b")

    assert memory.get("f0").superseded_by == ["f1"]
    assert [fact.id for fact in memory.history("f0")] == ["f0"]
    assert memory.apply_forgetting().dry_run is True
    assert body(requests[-1]) == {"includeChain": False, "dryRun": True, "purge": False}
    consolidated = memory.consolidate(allow_destructive=False)
    assert consolidated.created[0].id == "f1"
    assert body(requests[-1]) == {"dryRun": True, "allowDestructive": False}


def test_service_errors_become_memory_errors():
    memory = MemoryClient(HolographicClient(transport=transport([], error=(409, "conflict", "id exists"))))
    with pytest.raises(MemoryError) as caught:
        memory.remember("x", id="f")
    assert caught.value.code == "conflict" and caught.value.status == 409
    assert "id exists" in str(caught.value)


def test_async_client_matches_sync_payloads():
    requests = []

    async def run():
        client = AsyncHolographicClient(transport=httpx.MockTransport(lambda r: _async_handler(r, requests)))
        async with AsyncMemoryClient(client) as memory:
            await memory.remember("t", id="f1", valid_to=9)
            results = await memory.recall("q", k=2, explain=True)
            await memory.forget(subject="s", purge=True)
            return results

    results = asyncio.run(run())
    assert results[0].id == "f1"
    assert [r.url.path.rsplit("/", 1)[1] for r in requests] == ["remember", "recall", "forget"]
    assert body(requests[0]) == {"id": "f1", "text": "t", "validTo": 9}
    assert body(requests[2])["scope"] == {"kind": "subject", "subject": "s"}


def _async_handler(request, requests):
    requests.append(request)
    op = request.url.path.rsplit("/", 1)[1]
    payload = {
        "remember": {"remembered": [REMEMBERED]},
        "recall": {"results": [RECALLED]},
        "forget": {"forgotten": [], "dryRun": False, "purged": True},
    }[op]
    return httpx.Response(200, json=payload)


def test_client_exposes_memory_helpers():
    requests = []
    client = HolographicClient(transport=transport(requests))
    assert isinstance(client.memory, MemoryClient)
    assert client.memory.recall("q")[0].id == "f1"


def test_stub_declares_every_public_name():
    import ast
    import pathlib

    stub = pathlib.Path(memory_module.__file__).with_suffix(".pyi")
    tree = ast.parse(stub.read_text())
    declared = {node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    declared |= {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert set(memory_module.__all__) <= declared
    runtime_methods = {
        name for name in dir(memory_module.MemoryClient) if not name.startswith("_")
    }
    stub_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MemoryClient")
    stub_methods = {n.name for n in stub_class.body if isinstance(n, ast.FunctionDef)}
    assert runtime_methods <= stub_methods
