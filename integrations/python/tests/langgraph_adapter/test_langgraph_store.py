# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Differential tests: LangGraph's own InMemoryStore is the oracle for BaseStore semantics."""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langgraph.store.memory import InMemoryStore

from holographic_integrations.langgraph import HolographicStore

DIM = 16

DATA = [
    (("users", "alice", "prefs"), "theme", {"theme": "dark", "n": 1, "tags": ["a", "b"], "meta": {"lang": "en"}}),
    (("users", "alice", "prefs"), "font", {"font": "mono", "n": 7, "tags": ["b"], "meta": {"lang": "fr"}}),
    (("users", "bob", "prefs"), "theme", {"theme": "light", "n": 3, "tags": [], "meta": {"lang": "en"}}),
    (("users", "bob", "notes"), "n1", {"text": "buy milk", "n": 10, "flag": True}),
    (("docs",), "readme", {"text": "hello world", "n": 2.5, "flag": False}),
]


@pytest.fixture
def srv(server_factory):
    return server_factory(DIM)


def fresh(srv, **kw):
    return HolographicStore(url=srv.url, tenant_id=srv.tenant(), input_dim=DIM, **kw)


def loaded(srv, **kw):
    hms, mem = fresh(srv, **kw), InMemoryStore()
    for ns, key, value in DATA:
        hms.put(ns, key, value)
        mem.put(ns, key, value)
    return hms, mem


def view(items):
    return [(it.namespace, it.key, it.value) for it in items]


def test_get_put_delete_and_timestamps(srv):
    s = fresh(srv)
    s.put(("a", "b"), "k", {"v": 1})
    first = s.get(("a", "b"), "k")
    assert first.value == {"v": 1} and first.namespace == ("a", "b") and first.key == "k"
    s.put(("a", "b"), "k", {"v": 2})
    second = s.get(("a", "b"), "k")
    assert second.value == {"v": 2} and second.created_at == first.created_at and second.updated_at >= first.updated_at
    assert s.get(("a", "b"), "missing") is None
    s.delete(("a", "b"), "k")
    assert s.get(("a", "b"), "k") is None
    s.delete(("a", "b"), "k")  # idempotent


@pytest.mark.parametrize(
    "prefix,flt",
    [
        ((), None),
        (("users",), None),
        (("users", "alice"), None),
        (("users", "alice", "prefs"), {"theme": "dark"}),
        (("users",), {"n": {"$gt": 2}}),
        (("users",), {"n": {"$gte": 3, "$lt": 10}}),
        (("users",), {"n": {"$ne": 1}}),
        ((), {"flag": True}),
        ((), {"flag": False}),
        ((), {"n": 2.5}),
        (("users",), {"meta": {"lang": "en"}}),
        (("users",), {"tags": ["b"]}),
        (("users",), {"theme": "dark", "n": 1}),
        ((), {"nothing": "here"}),
    ],
)
def test_filter_only_search_matches_oracle(srv, prefix, flt):
    hms, mem = loaded(srv)
    got = sorted(view(hms.search(prefix, filter=flt, limit=50)), key=repr)
    want = sorted(view(mem.search(prefix, filter=flt, limit=50)), key=repr)
    assert got == want


def test_search_limit_and_offset(srv):
    hms, _ = loaded(srv)
    everything = view(hms.search((), limit=50))
    assert len(everything) == len(DATA)
    assert view(hms.search((), limit=2)) == everything[:2]
    assert view(hms.search((), limit=2, offset=2)) == everything[2:4]
    with pytest.raises(ValueError):
        hms.search((), limit=100, offset=1)  # offset + limit above max_fetch


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"prefix": ("users",)},
        {"prefix": ("users", "alice")},
        {"suffix": ("prefs",)},
        {"suffix": ("*", "prefs")},
        {"prefix": ("users", "*"), "suffix": ("prefs",)},
        {"max_depth": 1},
        {"max_depth": 2},
        {"prefix": ("users",), "max_depth": 2},
        {"limit": 2, "offset": 1},
    ],
)
def test_list_namespaces_matches_oracle(srv, kwargs):
    hms, mem = loaded(srv)
    assert hms.list_namespaces(**kwargs) == mem.list_namespaces(**kwargs)


def test_namespace_marker_removed_with_last_item(srv):
    hms, _ = loaded(srv)
    hms.delete(("users", "bob", "notes"), "n1")
    assert ("users", "bob", "notes") not in hms.list_namespaces()
    hms.delete(("users", "bob", "prefs"), "theme")
    assert ("users", "bob", "prefs") not in hms.list_namespaces()
    assert ("users", "alice", "prefs") in hms.list_namespaces()
    hms.delete(("users", "alice", "prefs"), "theme")
    assert ("users", "alice", "prefs") in hms.list_namespaces()  # still holds "font"


def test_semantic_search_and_index_controls(srv):
    emb = DeterministicFakeEmbedding(size=DIM)
    s = fresh(srv, index={"dims": DIM, "embed": emb, "fields": ["text"]})
    s.put(("m",), "milk", {"text": "buy milk at the store", "x": 1})
    s.put(("m",), "bread", {"text": "bake bread tomorrow", "x": 2})
    s.put(("m",), "hidden", {"text": "buy milk at the store", "x": 3}, index=False)
    s.put(("m",), "other", {"text": "irrelevant", "note": "bake bread tomorrow"}, index=["note"])
    top = s.search(("m",), query="buy milk at the store", limit=4)
    assert top[0].key == "milk" and top[0].score == pytest.approx(1.0, abs=1e-6)
    assert "hidden" not in [r.key for r in top]
    assert {r.key for r in top[1:]} == {"bread", "other"}
    narrowed = s.search(("m",), query="buy milk at the store", filter={"x": {"$gte": 2}}, limit=4)
    assert [r.key for r in narrowed] and all(r.value["x"] >= 2 for r in narrowed if "x" in r.value)
    plain = fresh(srv)  # no index: query is ignored, as for every BaseStore without an index
    plain.put(("m",), "a", {"text": "x"})
    assert [r.key for r in plain.search(("m",), query="anything")] == ["a"]


async def test_async_paths(srv):
    emb = DeterministicFakeEmbedding(size=DIM)
    s = fresh(srv, index={"dims": DIM, "embed": emb})
    try:
        await s.aput(("t",), "1", {"text": "alpha"})
        await s.aput(("t",), "2", {"text": "beta"})
        assert (await s.aget(("t",), "1")).value == {"text": "alpha"}
        hits = await s.asearch(("t",), query='{"text": "beta"}', limit=2)
        assert hits[0].key == "2"
        assert await s.alist_namespaces(prefix=("t",)) == [("t",)]
        await s.adelete(("t",), "1")
        assert await s.aget(("t",), "1") is None
        assert [r.key for r in await s.asearch(("t",), filter={"text": "beta"})] == ["2"]
    finally:
        await s.aclose()


def test_batch_preserves_order_and_rejects_ttl(srv):
    from langgraph.store.base import GetOp, PutOp

    s = fresh(srv)
    results = s.batch([PutOp(("b",), "k", {"v": 1}, None, None), GetOp(("b",), "k", True), PutOp(("b",), "k", None, None, None), GetOp(("b",), "k", True)])
    assert results[0] is None and results[1].value == {"v": 1} and results[3] is None
    with pytest.raises(NotImplementedError):
        s.put(("b",), "k", {"v": 1}, ttl=5)


def test_namespace_depth_and_value_size_limits(srv):
    from holographic_integrations.core import HmsError

    s = fresh(srv)
    s.put(tuple(str(i) for i in range(16)), "k", {"v": 1})
    with pytest.raises(ValueError):
        s.put(tuple(str(i) for i in range(17)), "k", {"v": 1})
    with pytest.raises(HmsError):
        s.put(("big",), "k", {"v": "x" * 70_000})
