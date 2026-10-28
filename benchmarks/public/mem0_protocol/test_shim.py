# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""uv run --no-project --with pytest --with aiohttp==3.13.2 pytest benchmarks/public/mem0_protocol/test_shim.py"""

import asyncio
import copy
import json
import re
from datetime import datetime, timezone

import pytest
import shim


def test_tenant_ids_are_valid_and_injective_on_harness_ids():
    for uid in ("locomo_0_4c92d691", "longmemeval_gpt4_2655b836_abs_1a2b3c4d", "beam_10M_3_deadbeef"):
        assert shim.tenant_for(uid) == uid
    for uid in ("a/b", "x" * 65, "q.1", "ü"):
        t = shim.tenant_for(uid)
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", t)
    assert shim.tenant_for("a/b") != shim.tenant_for("a.b")


def test_split_fact_boundaries():
    word = "abcdefghi"  # 9 bytes + 1 separator
    at_limit = " ".join([word] * shim.MAX_FACT_WORDS)
    assert len(at_limit.encode()) <= shim.MAX_FACT_BYTES
    assert shim.split_fact(at_limit) == [at_limit]
    assert shim.split_fact(" ".join([word] * (shim.MAX_FACT_WORDS - 1))) == [" ".join([word] * (shim.MAX_FACT_WORDS - 1))]
    over = " ".join([word] * (shim.MAX_FACT_WORDS + 1))
    parts = shim.split_fact(over)
    assert len(parts) == 2 and " ".join(parts) == over
    big = "y" * 70_000
    parts = shim.split_fact(big)
    assert len(parts) > 1 and all(len(p.encode()) <= shim.MAX_FACT_BYTES for p in parts)
    assert "".join(p.replace(" ", "") for p in parts) == big
    long_bytes = " ".join(["z" * 999] * 70)  # 70,000 bytes, 70 words
    parts = shim.split_fact(long_bytes)
    assert all(len(p.encode()) <= shim.MAX_FACT_BYTES for p in parts) and " ".join(parts) == long_bytes


def test_fact_text_keeps_harness_strings_verbatim():
    msgs = [{"role": "user", "content": "Caroline: hi [Sharing image - query: x. The image shows: y]"},
            {"role": "assistant", "content": "ok"}]
    assert shim.fact_text(msgs) == "user: Caroline: hi [Sharing image - query: x. The image shows: y]\nassistant: ok"


@pytest.mark.parametrize("separator", ["\n", "\t", "\r\n", "\u2003"])
@pytest.mark.parametrize("count", [shim.MAX_FACT_WORDS - 1, shim.MAX_FACT_WORDS, shim.MAX_FACT_WORDS + 1])
def test_split_fact_respects_engine_word_limit_for_all_whitespace(separator, count):
    text = separator.join(["word"] * count)
    parts = shim.split_fact(text)
    assert [word for part in parts for word in part.split()] == text.split()
    assert all(len(part.split()) <= shim.MAX_FACT_WORDS for part in parts)
    if count <= shim.MAX_FACT_WORDS:
        assert parts == [text]


def test_created_at_parses_with_the_harness_date_rule():
    # memory-benchmarks _to_human_date: strptime(iso[:26].rstrip("Z"), "%Y-%m-%dT%H:%M:%S") first.
    epoch = int(datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc).timestamp())
    iso = shim.iso_from_ms(epoch * 1000)
    assert iso == "2023-05-08T13:56:00"
    assert datetime.strptime(iso[:26].rstrip("Z"), "%Y-%m-%dT%H:%M:%S").strftime("%A, %B %d, %Y") == "Monday, May 08, 2023"
    assert shim.iso_from_ms(None) is None
    # lexicographic order of the strings equals time order (the harness sorts them as strings)
    assert shim.iso_from_ms(1_000) < shim.iso_from_ms(86_400_000) < shim.iso_from_ms(10**12)


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"max_k": 200, "recall": {"lexicalWeight": 1}}')
    assert shim.load_config(str(p))["max_k"] == 200
    p.write_text('{"max_k": 200, "answers": {}}')
    try:
        shim.load_config(str(p))
    except ValueError:
        pass
    else:
        raise AssertionError("unknown key accepted")


def test_budget_fills_in_rank_order_and_skips_overflow():
    hits = ["a", "b", "c", "d"]
    assert shim.fit_budget(hits, [3, 5, 2, 1], 6) == ["a", "c", "d"]
    assert shim.fit_budget(hits, [3, 3, 1, 1], 6) == ["a", "b"]  # limit: exactly full
    assert shim.fit_budget(hits, [3, 2, 1, 1], 6) == ["a", "b", "c"]  # limit - 1 leaves room for one
    assert shim.fit_budget(hits, [7, 1, 1, 1], 6) == ["b", "c", "d"]  # oversized top hit is skipped
    assert shim.fit_budget([], [], 6) == []


def test_embedding_config_bounds(tmp_path):
    p = tmp_path / "c.json"
    for dimensions in (1, 2, 1024, 3071, 3072):
        p.write_text('{"embedding":{"model":"text-embedding-3-large","dimensions":%d}}' % dimensions)
        assert shim.load_config(str(p))["embedding"]["dimensions"] == dimensions
    for dimensions in (0, 3073, True):
        p.write_text('{"embedding":{"model":"text-embedding-3-large","dimensions":%s}}' % str(dimensions).lower())
        with pytest.raises(ValueError):
            shim.load_config(str(p))


def test_same_time_ordering_restores_dialogue_without_changing_selected_content():
    hits = [
        {"id": "reply", "text": "assistant: The orchard.", "score": 0.9, "eventTime": 1000, "ingestTime": 30},
        {"id": "other", "text": "user: Another day.", "score": 0.8, "eventTime": 2000, "ingestTime": 10},
        {"id": "question", "text": "user: Where did we go?", "score": 0.7, "eventTime": 1000, "ingestTime": 20},
        {"id": "untimed", "text": "user: No session.", "score": 0.6},
        {"id": "unknown-order", "text": "user: No ingest time.", "score": 0.5, "eventTime": 1000},
    ]
    before = copy.deepcopy(hits)
    ordered = shim.order_same_time(hits)
    assert hits == before
    assert [row["id"] for row in ordered] == ["question", "other", "reply", "untimed", "unknown-order"]
    assert ordered == sorted(ordered, key=lambda row: row["score"], reverse=True)
    strip_score = lambda rows: {row["id"]: {k: v for k, v in row.items() if k != "score"} for row in rows}
    assert strip_score(ordered) == strip_score(hits)
    assert shim.order_same_time(ordered) == ordered


def test_same_time_ordering_preserves_native_rank_on_ingest_time_ties():
    hits = [{"id": name, "eventTime": 1000, "ingestTime": time}
            for name, time in [("z", 30), ("y", 20), ("x", 20), ("a", 10)]]
    assert [row["id"] for row in shim.order_same_time(hits)] == ["a", "y", "x", "z"]
    assert shim.order_same_time([]) == []


def test_search_orders_only_after_identical_budget_selection():
    hits = [
        {"id": "later", "text": "Second.", "score": 0.9, "eventTime": 1000, "ingestTime": 20},
        {"id": "large", "text": "too big " * 100, "score": 0.8, "eventTime": 1000, "ingestTime": 15},
        {"id": "earlier", "text": "First.", "score": 0.7, "eventTime": 1000, "ingestTime": 10},
        {"id": "extra", "text": "Extra.", "score": 0.6, "eventTime": 2000, "ingestTime": 30},
    ]

    class Request:
        async def json(self):
            return {"query": "Any question", "user_id": "tenant", "limit": 200}

    async def run():
        outputs = []
        for enabled in (False, True):
            adapter = shim.Shim("http://unused", {"max_context_tokens": 4, "order_same_time": enabled})

            async def hms(operation, tenant, body):
                assert operation == "recall" and tenant == "tenant"
                return {"results": hits}

            adapter.hms = hms
            response = await adapter.search(Request())
            outputs.append(json.loads(response.text)["results"])
        assert [row["id"] for row in outputs[0]] == ["later", "earlier"]
        assert [row["id"] for row in outputs[1]] == ["earlier", "later"]
        content = lambda rows: {row["id"]: (row["memory"], row["created_at"]) for row in rows}
        assert content(outputs[0]) == content(outputs[1])

    asyncio.run(run())


def test_same_time_ordering_config_is_explicit_boolean(tmp_path):
    path = tmp_path / "config.json"
    for value in (True, False):
        path.write_text(json.dumps({"order_same_time": value}))
        assert shim.load_config(str(path))["order_same_time"] is value
    for value in (0, 1, None, "true"):
        path.write_text(json.dumps({"order_same_time": value}))
        with pytest.raises(ValueError):
            shim.load_config(str(path))


def test_embedding_response_validates_indices_and_vectors(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = shim.EmbeddingClient(None, 2, None)
    good = {"data": [{"index": 1, "embedding": [0.1, 0.2]},
                     {"index": 0, "embedding": [0.3, 0.4]}],
            "usage": {"total_tokens": 9}}
    assert client._parse(good, 2) == [[0.3, 0.4], [0.1, 0.2]]
    for data in ([{"index": 0, "embedding": [0.1, 0.2]}],
                 [{"index": 0, "embedding": [0.1, 0.2]}, {"index": 0, "embedding": [0.3, 0.4]}],
                 [{"index": 0, "embedding": [float("nan"), 0.2]}, {"index": 1, "embedding": [0.3, 0.4]}],
                 [{"index": 0, "embedding": [0.1]}, {"index": 1, "embedding": [0.3, 0.4]}]):
        with pytest.raises(ValueError):
            client._parse({"data": data, "usage": {"total_tokens": 9}}, 2)


def hybrid_with_store(store, embedder):
    adapter = shim.Shim("http://unused", {
        "embedding": {"model": "text-embedding-3-large", "dimensions": 2},
    })
    adapter.embedder = embedder

    async def hms(operation, tenant, body):
        if operation == "get":
            return {"fact": store.get((tenant, body["id"]))}
        assert operation == "remember"
        fact = copy.deepcopy(body["facts"][0])
        key = (tenant, fact["id"])
        previous = store.get(key)
        if previous is not None and previous != fact:
            raise shim.web.HTTPConflict()
        store[key] = fact
        return {"remembered": [{"id": fact["id"], "created": previous is None}]}

    adapter.hms = hms
    return adapter


def test_hybrid_retries_survive_restart_and_keep_times_parts_and_tenants_distinct():
    class VaryingEmbedder:
        calls = 0

        async def embed(self, texts):
            self.calls += 1
            return [[1.0, self.calls / 100] for _ in texts]

    async def run():
        store, embedder = {}, VaryingEmbedder()
        adapter = hybrid_with_store(store, embedder)
        facts = [{"text": text, "eventTime": 1000}
                 for text in shim.split_fact(" ".join(["word"] * (shim.MAX_FACT_WORDS + 1)))]
        first = await adapter.remember_embedded("a", copy.deepcopy(facts))
        assert len(store) == 2 and embedder.calls == 2
        restarted = hybrid_with_store(store, embedder)
        repeated = await restarted.remember_embedded("a", copy.deepcopy(facts))
        assert [r["id"] for r in first["remembered"]] == [r["id"] for r in repeated["remembered"]]
        assert all(not row["created"] for row in repeated["remembered"])
        assert len(store) == 2 and embedder.calls == 2
        await restarted.remember_embedded("a", [{"text": facts[0]["text"], "eventTime": 2000}])
        await restarted.remember_embedded("b", copy.deepcopy(facts))
        assert len(store) == 5

    asyncio.run(run())


def test_concurrent_hybrid_retries_with_different_embeddings_keep_one_fact():
    class RacingEmbedder:
        def __init__(self):
            self.calls = 0
            self.ready = asyncio.Event()

        async def embed(self, texts):
            self.calls += 1
            number = self.calls
            if number == 2:
                self.ready.set()
            await self.ready.wait()
            return [[1.0, number / 100] for _ in texts]

    async def run():
        store, embedder = {}, RacingEmbedder()
        adapters = [hybrid_with_store(store, embedder) for _ in range(2)]
        outcomes = await asyncio.gather(*(
            adapter.remember_embedded("a", [{"text": "A durable fact", "eventTime": 1000}])
            for adapter in adapters
        ))
        assert embedder.calls == 2 and len(store) == 1
        assert outcomes[0]["remembered"][0]["id"] == outcomes[1]["remembered"][0]["id"]
        assert sum(outcome["remembered"][0]["created"] for outcome in outcomes) == 1

    asyncio.run(run())


def test_retry_verifies_native_trimmed_text_and_rejects_changed_content():
    class Embedder:
        async def embed(self, texts):
            return [[1.0, 0.1] for _ in texts]

    async def run():
        store = {}
        adapter = hybrid_with_store(store, Embedder())
        for text in ("  user: fact \t\n", "\u2003user: fact\u00a0", "user: control\u001c"):
            fact = {"text": text, "eventTime": 1000}
            outcome = await adapter.remember_embedded("tenant", [dict(fact)])
            key = ("tenant", outcome["remembered"][0]["id"])
            store[key]["text"] = text.strip(shim.RUST_WHITESPACE)
            repeated = await adapter.remember_embedded("tenant", [dict(fact)])
            assert not repeated["remembered"][0]["created"]
            store[key]["text"] += " changed"
            with pytest.raises(RuntimeError, match="stable identity"):
                await adapter.remember_embedded("tenant", [dict(fact)])

    asyncio.run(run())
