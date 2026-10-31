# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest

from writerslogic_warm_integrations.core import (
    AsyncWarmBackend, HashingEmbedder, WarmBackend, WarmConfig, WarmError, Record, parse_filter,
)
from writerslogic_warm_integrations.core.filters import Cmp, split


def cfg(server, **kw):
    return WarmConfig(url=server.url, tenant_id=server.tenant(), input_dim=server.dim, **kw)


def recs(n, **meta):
    return [Record(f"d{i}", f"alpha beta gamma number{i} topic{i % 3}", {"n": i, "parity": i % 2, **meta}) for i in range(n)]


def test_roundtrip_get_list_forget(server):
    with WarmBackend(cfg(server)) as b:
        b.remember(recs(5))
        hits = b.recall("alpha beta gamma number3 topic0", k=2)
        assert hits[0].id == "d3" and hits[0].metadata == {"n": 3, "parity": 1}
        assert [r.id for r in b.get(["d1", "nope", "d4"]) if r] == ["d1", "d4"]
        page = b.list()
        assert sorted(r.id for r in page.records) == [f"d{i}" for i in range(5)] and not page.truncated
        assert b.forget(["d0", "ghost"]) == 1
        assert b.get(["d0"]) == [None]


def test_replace_same_id(server):
    with WarmBackend(cfg(server)) as b:
        b.remember([Record("x", "first text", {"v": 1})])
        b.remember([Record("x", "second text", {"v": 2})])
        got = b.get(["x"])[0]
        assert got.text == "second text" and got.metadata == {"v": 2}
        assert len(b.list().records) == 1


def test_filters_pushed_and_residual(server):
    with WarmBackend(cfg(server)) as b:
        b.remember(recs(20))
        assert {h.metadata["parity"] for h in b.recall("alpha", k=5, filter={"parity": 1})} == {1}
        ids = {r.id for r in b.list(filter={"n": {"$gte": 15}}).records}
        assert ids == {f"d{i}" for i in range(15, 20)}
        ids = {r.id for r in b.list(filter={"$or": [{"n": 1}, {"n": 2}], "parity": 1}).records}
        assert ids == {"d1"}


def test_collections_isolated_in_one_tenant(server):
    c = cfg(server)
    with WarmBackend(c) as plain, WarmBackend(dataclasses.replace(c, collection="A")) as a:
        plain.remember([Record("same", "plain doc")])
        a.remember([Record("same", "collection doc")])
        assert plain.get(["same"])[0].text == "plain doc"
        assert a.get(["same"])[0].text == "collection doc"
        assert [r.text for r in a.list().records] == ["collection doc"]
        assert a.forget(["same"]) == 1 and plain.get(["same"])[0] is not None


def test_reserved_metadata_and_bad_ids_rejected(server):
    with WarmBackend(cfg(server)) as b:
        with pytest.raises(ValueError):
            b.remember([Record("a", "t", {"_warm_id": "x"})])
        with pytest.raises(ValueError):
            b.remember([Record("hms:a", "t")])
        with pytest.raises(ValueError):
            b.remember([Record("é" * 129, "t")])
        with pytest.raises(WarmError) as e:
            b.remember([Record("big", "w " * 40000)])
        assert e.value.status == 422


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_list_truncation_at_limit(server_factory, delta):
    srv = server_factory(64, max_top_k=10)
    c = WarmConfig(url=srv.url, tenant_id=srv.tenant(), input_dim=64, max_fetch=10)
    with WarmBackend(c) as b:
        b.remember(recs(10 + delta))
        page = b.list()
        assert len(page.records) == min(10, 10 + delta)
        assert page.truncated == (10 + delta > 10)
        with pytest.raises(ValueError):
            b.recall("alpha", k=11)


def test_forget_where_pages_past_the_cap(server_factory):
    srv = server_factory(64, max_top_k=10)
    c = WarmConfig(url=srv.url, tenant_id=srv.tenant(), input_dim=64, max_fetch=10)
    with WarmBackend(c) as b:
        b.remember(recs(25, kind="x") + [Record("keep", "keep me", {"kind": "y"})])
        assert b.forget_where({"kind": "x"}) == 25
        assert [r.id for r in b.list().records] == ["keep"]


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_residual_filters_find_matches_past_the_query_cap(server_factory, delta):
    srv = server_factory(64, max_top_k=10)
    config = WarmConfig(url=srv.url, tenant_id=srv.tenant(), input_dim=64, max_fetch=10)
    with WarmBackend(config) as backend:
        backend.remember(recs(1000 + delta))
        filter = {"n": {"$gte": 995}}
        expected = {f"d{i}" for i in range(995, 1000 + delta)}
        assert {record.id for record in backend.list(filter).records} == expected
        assert {hit.id for hit in backend.recall("alpha", k=10, filter=filter)} == expected
        assert backend.forget_where(filter) == len(expected)
        assert backend.list(filter).records == []


def test_residual_vector_rank_merges_multiple_metadata_pages(server_factory):
    srv = server_factory(64, max_top_k=10)
    config = WarmConfig(url=srv.url, tenant_id=srv.tenant(), input_dim=64, max_fetch=10)
    with WarmBackend(config) as backend:
        v = [1.0] + [0.0] * 63
        backend.remember([Record(f"id-{i:04}", "same", {"n": i}, vector=v) for i in range(1001)])
        hits = backend.recall(vector=v, k=10, filter={"n": {"$gte": 0}})
        assert [hit.id for hit in hits] == [f"id-{i:04}" for i in range(10)]


async def test_async_residual_filters_page_and_delete_completely(server_factory):
    srv = server_factory(64, max_top_k=10)
    config = WarmConfig(url=srv.url, tenant_id=srv.tenant(), input_dim=64, max_fetch=10)
    async with AsyncWarmBackend(config) as backend:
        await backend.remember(recs(31))
        filter = {"n": {"$gte": 25}}
        expected = {f"d{i}" for i in range(25, 31)}
        assert {record.id for record in (await backend.list(filter)).records} == expected
        assert {hit.id for hit in await backend.recall("alpha", k=10, filter=filter)} == expected
        assert await backend.forget_where(filter) == len(expected)
        assert (await backend.list(filter)).records == []


def test_masked_vectors_still_rank(server):
    with WarmBackend(cfg(server, zero_trust_key="correct horse battery staple")) as b:
        b.remember(recs(6))
        assert b.recall("alpha beta gamma number4 topic1", k=1)[0].id == "d4"
        assert b.get(["d2"])[0].id == "d2"


def test_custom_vectors_and_dim_mismatch(server):
    with WarmBackend(cfg(server)) as b:
        v = HashingEmbedder(64).embed_query("custom")
        b.remember([Record("v", "text", vector=v)])
        assert b.recall(vector=v, k=1)[0].id == "v"
        with pytest.raises(WarmError) as e:
            b.remember([Record("w", "text", vector=[1.0, 2.0])])
        assert e.value.status == 422


async def test_async_matches_sync(server):
    async with AsyncWarmBackend(cfg(server)) as b:
        await b.remember(recs(8))
        assert (await b.recall("alpha beta gamma number5 topic2", k=1))[0].id == "d5"
        assert [r.id for r in await b.get(["d2", "d7", "zz"]) if r] == ["d2", "d7"]
        assert {r.id for r in (await b.list(filter={"n": {"$lt": 2}})).records} == {"d0", "d1"}
        assert await b.forget_where({"parity": 0}) == 4
        assert len((await b.list()).records) == 4


def test_filter_semantics():
    f = parse_filter({"a": 1, "b": {"$in": [1, 2]}, "$not": {"c": True}})
    assert f.matches({"a": 1, "b": 2, "c": False})
    assert not f.matches({"a": 1, "b": 2, "c": True})
    assert not parse_filter({"a": 1}).matches({"a": True})  # bool is not int
    assert parse_filter({"a": {"$ne": 1}}).matches({})
    assert parse_filter({"k.x": 3}).matches({"k": {"x": 3}})
    pushed, rest = split(parse_filter({"a": 1, "b": {"$gt": 2}}))
    assert pushed == {"a": 1} and rest == Cmp("b", "gt", 2)
    assert split(parse_filter({"a": 1, "$or": [{"b": 1}, {"c": 2}]}))[0] == {"a": 1}
