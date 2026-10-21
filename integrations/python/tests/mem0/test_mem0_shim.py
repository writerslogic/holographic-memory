# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import datetime
import json
import sqlite3
import warnings

import pytest

from holographic_integrations.core import HmsConfig
from holographic_integrations.mem0 import AsyncMemory, LLMExtractor, Memory


def make(server, **kw):
    return Memory(hms=HmsConfig(url=server.url, tenant_id=server.tenant(), input_dim=server.dim), **kw)


def amake(server, **kw):
    return AsyncMemory(hms=HmsConfig(url=server.url, tenant_id=server.tenant(), input_dim=server.dim), **kw)


def test_add_search_get_update_delete_history(server):
    with make(server) as m:
        with pytest.warns(UserWarning, match="VERBATIM"):
            added = m.add("I love hiking in the alps", user_id="alice", metadata={"topic": "hobby"})
        (r,) = added["results"]
        assert set(r) == {"id", "memory", "event", "actor_id", "role"} and r["event"] == "ADD"
        got = m.get(r["id"])
        assert got["memory"] == "I love hiking in the alps" and got["user_id"] == "alice"
        assert got["metadata"] == {"topic": "hobby"} and got["score"] is None and got["role"] == "user"
        hit = m.search("I love hiking in the alps", filters={"user_id": "alice"})["results"][0]
        assert hit["id"] == r["id"] and hit["score"] > 0.9
        assert m.update(r["id"], text="I love skiing in the alps") == {"message": "Memory updated successfully!"}
        new = m.get(r["id"])
        assert new["memory"] == "I love skiing in the alps" and new["created_at"] == got["created_at"]
        assert new["updated_at"] >= got["updated_at"] and new["hash"] != got["hash"]
        assert m.delete(r["id"]) == {"message": "Memory deleted successfully!"}
        assert m.get(r["id"]) is None
        events = [(h["event"], h["old_memory"], h["new_memory"], h["is_deleted"]) for h in m.history(r["id"])]
        assert events == [
            ("ADD", None, "I love hiking in the alps", False),
            ("UPDATE", "I love hiking in the alps", "I love skiing in the alps", False),
            ("DELETE", "I love skiing in the alps", None, True),
        ]
        with pytest.raises(ValueError, match="not found"):
            m.delete(r["id"])
        with pytest.raises(ValueError, match="not found"):
            m.update(r["id"], text="x")


def test_scoping_validation_and_messages(server):
    with make(server) as m:
        msgs = [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "my name is Bob", "name": "bob"},
            {"role": "assistant", "content": "hello Bob"},
        ]
        res = m.add(msgs, user_id="u1", agent_id="a1", infer=False)["results"]
        assert [(x["role"], x["actor_id"]) for x in res] == [("user", "bob"), ("assistant", None)]
        m.add("other user fact", user_id="u2", infer=False)
        assert len(m.get_all(filters={"user_id": "u1"})["results"]) == 2
        assert len(m.get_all(filters={"user_id": "u1", "agent_id": "a1"})["results"]) == 2
        assert m.get_all(filters={"user_id": "u1", "agent_id": "zzz"})["results"] == []
        assert all("score" not in i for i in m.get_all(filters={"user_id": "u2"})["results"])
        with pytest.raises(ValueError, match="at least one of"):
            m.search("q", filters={})
        with pytest.raises(ValueError, match="Top-level entity"):
            m.search("q", user_id="u1")
        with pytest.raises(ValueError, match="cannot contain whitespace"):
            m.add("x", user_id="a b")
        with pytest.raises(ValueError, match="At least one"):
            m.add("x")
        with pytest.raises(ValueError, match="At least one filter"):
            m.delete_all()
        with pytest.raises(ValueError):
            m.search("q", filters={"user_id": "u1"}, threshold=2)
        assert m.search("anything", filters={"user_id": "u1"}, top_k=0) == {"results": []}


def test_v2_filters(server):
    with make(server) as m:
        for i, (color, n) in enumerate([("red", 1), ("blue", 5), ("green", 9), ("red", 12)]):
            m.add(f"item number{i} thing", user_id="u", metadata={"color": color, "n": n, "name": f"Item-{i}"}, infer=False)

        def ids(filters):
            return sorted(r["metadata"]["n"] for r in m.get_all(filters={"user_id": "u", **filters})["results"])

        assert ids({"color": "red"}) == [1, 12]
        assert ids({"n": {"gte": 5, "lt": 12}}) == [5, 9]
        assert ids({"color": {"in": ["red", "green"]}, "n": {"gt": 1}}) == [9, 12]
        assert ids({"OR": [{"color": "blue"}, {"n": {"eq": 12}}]}) == [5, 12]
        assert ids({"AND": [{"color": "red"}, {"n": {"ne": 1}}]}) == [12]
        assert ids({"NOT": [{"color": "red"}]}) == [5, 9]
        assert ids({"name": {"icontains": "ITEM-2"}}) == [9]
        assert ids({"name": {"contains": "Item"}, "color": "*"}) == [1, 5, 9, 12]
        assert ids({"missing": "*"}) == []
        hits = m.search("item thing", filters={"user_id": "u", "n": {"gt": 4}}, threshold=0.0)["results"]
        assert sorted(h["metadata"]["n"] for h in hits) == [5, 9, 12]
        with pytest.raises(ValueError, match="Unsupported metadata filter operator"):
            m.get_all(filters={"user_id": "u", "n": {"regex": "x"}})


def test_expiration(server):
    with make(server) as m:
        past = (datetime.date.today() - datetime.timedelta(days=2)).isoformat()
        future = (datetime.date.today() + datetime.timedelta(days=2)).isoformat()
        m.add("old fact", user_id="u", expiration_date=past, infer=False)
        m.add("new fact", user_id="u", expiration_date=future, infer=False)
        assert [r["memory"] for r in m.get_all(filters={"user_id": "u"})["results"]] == ["new fact"]
        assert len(m.get_all(filters={"user_id": "u"}, show_expired=True)["results"]) == 2
        assert [r["memory"] for r in m.search("fact", filters={"user_id": "u"}, threshold=0)["results"]] == ["new fact"]
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            m.add("x", user_id="u", expiration_date="tomorrow")


def test_delete_all_pages_past_the_service_cap(server_factory):
    srv = server_factory(64, max_top_k=10)
    m = Memory(hms=HmsConfig(url=srv.url, tenant_id=srv.tenant(), input_dim=64, max_fetch=10), buckets=8)
    with m:
        for i in range(23):
            m.add(f"fact {i}", user_id="big", infer=False)
        m.add("keep", user_id="other", infer=False)
        m.delete_all(user_id="big")
        assert m.get_all(filters={"user_id": "big"})["results"] == []
        assert len(m.get_all(filters={"user_id": "other"})["results"]) == 1


@pytest.mark.parametrize("total", [9, 10, 11, 25])
def test_get_all_beyond_one_page(server_factory, total):
    """limit-1, limit, limit+1 and several pages for max_fetch=10."""
    srv = server_factory(64, max_top_k=10)
    m = Memory(hms=HmsConfig(url=srv.url, tenant_id=srv.tenant(), input_dim=64, max_fetch=10), buckets=8)
    with m:
        for i in range(total):
            m.add(f"fact {i}", user_id="u", infer=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = m.get_all(filters={"user_id": "u"}, top_k=100)["results"]
        assert sorted(r["memory"] for r in got) == sorted(f"fact {i}" for i in range(total))
        assert len(m.get_all(filters={"user_id": "u"}, top_k=5)["results"]) == 5


def test_reset_and_history_persistence(server, tmp_path):
    db = str(tmp_path / "history.db")
    with make(server, history_db_path=db) as m:
        mid = m.add("one", user_id="u", infer=False)["results"][0]["id"]
        m.add("two", user_id="v", infer=False)
        assert len(m.history(mid)) == 1
    with sqlite3.connect(db) as con:
        assert con.execute("select count(*) from history").fetchone()[0] == 2
    with make(server) as m:
        m.add("x", user_id="u", infer=False)
        m.reset()
        assert m.get_all(filters={"user_id": "u"})["results"] == []


def test_extractor_infer_and_dedupe(server):
    seen = {}

    def llm(messages):
        seen["messages"] = messages
        return '```json\n{"memory": [{"text": "Alice likes tea"}, {"text": "Alice lives in Oslo", "attributed_to": "user"}, {"text": "Alice likes tea"}]}\n```'

    with make(server, extractor=LLMExtractor(llm)) as m, warnings.catch_warnings():
        warnings.simplefilter("error")
        res = m.add("I like tea and I live in Oslo", user_id="alice")["results"]
        assert [r["memory"] for r in res] == ["Alice likes tea", "Alice lives in Oslo"]
        assert "I like tea and I live in Oslo" in seen["messages"][1]["content"]
        assert m.add("I like tea and I live in Oslo", user_id="alice")["results"] == []  # hash dedupe
        got = {r["memory"]: r for r in m.get_all(filters={"user_id": "alice"})["results"]}
        assert got["Alice lives in Oslo"]["attributed_to"] == "user"
        # infer=False bypasses the extractor entirely
        assert m.add("raw", user_id="alice", infer=False)["results"][0]["memory"] == "raw"


def test_strict_infer_and_async_extractor_in_sync_memory(server):
    with make(server, strict_infer=True) as m:
        with pytest.raises(ValueError, match="needs an extractor"):
            m.add("x", user_id="u")

    async def extract(messages, existing, prompt):
        return ["fact"]

    with make(server, extractor=extract) as m:
        with pytest.raises(TypeError, match="AsyncMemory"):
            m.add("x", user_id="u")


def test_from_config_mem0_dict(server):
    cfg = {
        "vector_store": {"provider": "hms", "config": {"url": server.url, "tenant_id": server.tenant(), "collection_name": "c1", "embedding_model_dims": 64}},
        "embedder": {"provider": "hashing", "config": {"embedding_dims": 64}},
        "llm": lambda messages: json.dumps({"memory": [{"text": "fact from llm"}]}),
    }
    with Memory.from_config(cfg) as m:
        assert m.add("hi", user_id="u")["results"][0]["memory"] == "fact from llm"
        assert m.settings.hms.collection == "c1"
    with pytest.raises(ValueError, match="not supported here"):
        Memory.from_config({"vector_store": {"provider": "qdrant", "config": {}}})
    with pytest.raises(ValueError, match="not supported here"):
        Memory.from_config({"embedder": {"provider": "openai", "config": {}}})
    with pytest.raises(ValueError, match="hosted-LLM"):
        Memory.from_config({"llm": {"provider": "openai", "config": {}}})


def test_rerank_callable_and_explain(server):
    calls = []

    def rerank(query, items, top_k):
        calls.append(query)
        return list(reversed(items))[:top_k]

    with make(server, reranker=rerank) as m:
        m.add("alpha beta", user_id="u", infer=False)
        m.add("alpha beta gamma", user_id="u", infer=False)
        plain = m.search("alpha beta", filters={"user_id": "u"}, threshold=0)["results"]
        re = m.search("alpha beta", filters={"user_id": "u"}, threshold=0, rerank=True, explain=True)["results"]
        assert calls == ["alpha beta"] and [r["id"] for r in re] == [r["id"] for r in reversed(plain)]
        assert "score_details" in re[0]


def test_unsupported_params(server):
    with make(server) as m:
        with pytest.raises(ValueError, match="timestamp"):
            m.add("x", user_id="u", timestamp=1)
        with pytest.raises(NotImplementedError):
            m.add("x", agent_id="a", memory_type="procedural_memory")
        with pytest.raises(ValueError, match="Invalid 'memory_type'"):
            m.add("x", user_id="u", memory_type="nope")
        with pytest.raises(ValueError, match="reference_date"):
            m.search("x", filters={"user_id": "u"}, reference_date="2025-01-01")


def test_identity_keys_in_metadata_are_ignored(server):
    with make(server) as m:
        r = m.add("x", user_id="u", metadata={"user_id": "hijack", "keep": 1}, infer=False)["results"][0]
        got = m.get(r["id"])
        assert got["user_id"] == "u" and got["metadata"] == {"keep": 1}
        m.update(r["id"], metadata={"user_id": "hijack", "more": 2})
        assert m.get(r["id"])["user_id"] == "u" and m.get(r["id"])["metadata"] == {"keep": 1, "more": 2}


async def test_async_memory_end_to_end(server):
    async def llm(messages):
        return '{"memory": [{"text": "async fact"}]}'

    async with amake(server, extractor=LLMExtractor(llm)) as m:
        res = (await m.add("hello", user_id="u"))["results"]
        assert res[0]["memory"] == "async fact"
        mid = res[0]["id"]
        assert (await m.get(mid))["memory"] == "async fact"
        assert (await m.search("async fact", filters={"user_id": "u"}))["results"][0]["id"] == mid
        await m.update(mid, text="async fact 2")
        assert [h["event"] for h in await m.history(mid)] == ["ADD", "UPDATE"]
        assert len((await m.get_all(filters={"user_id": "u"}))["results"]) == 1
        await m.delete_all(user_id="u")
        assert (await m.get_all(filters={"user_id": "u"}))["results"] == []
        await m.reset()


async def test_async_get_all_buckets(server_factory):
    srv = server_factory(64, max_top_k=10)
    cfg = HmsConfig(url=srv.url, tenant_id=srv.tenant(), input_dim=64, max_fetch=10)
    async with AsyncMemory(hms=cfg, buckets=8) as m:
        for i in range(25):
            await m.add(f"fact {i}", user_id="u", infer=False)
        got = (await m.get_all(filters={"user_id": "u"}, top_k=100))["results"]
        assert len(got) == 25
