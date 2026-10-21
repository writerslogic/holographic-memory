# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import warnings
from datetime import datetime, timedelta

import pytest

pytest.importorskip("crewai")

from crewai.knowledge.knowledge import Knowledge  # noqa: E402
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource  # noqa: E402
from crewai.memory.storage.backend import StorageBackend  # noqa: E402
from crewai.memory.types import MemoryRecord  # noqa: E402
from crewai.memory.unified_memory import Memory  # noqa: E402

from holographic_integrations.core import HashingEmbedder, HmsConfig  # noqa: E402
from holographic_integrations.crewai import HolographicKnowledgeStorage, HolographicMemoryStorage  # noqa: E402

DIM = 64
EMB = HashingEmbedder(DIM)


def cfg(server, **kw):
    return HmsConfig(url=server.url, tenant_id=server.tenant(), input_dim=DIM, **kw)


def rec(content, scope="/", categories=(), metadata=None, **kw):
    return MemoryRecord(
        content=content, scope=scope, categories=list(categories), metadata=metadata or {},
        embedding=EMB.embed_query(content), **kw,
    )


def test_implements_storage_backend_protocol(server):
    assert isinstance(HolographicMemoryStorage(cfg(server)), StorageBackend)


def test_search_filters_scope_category_metadata(server):
    s = HolographicMemoryStorage(cfg(server))
    s.save([
        rec("alpha deploy runbook", "/ops/deploy", ["runbook"], {"team": "a"}),
        rec("alpha deploy checklist", "/ops/deploy/prod", ["checklist", "runbook"], {"team": "b"}),
        rec("alpha billing notes", "/finance", ["notes"], {"team": "a"}),
        rec("alpha deploy opsx", "/opsx", ["x"]),
    ])
    q = EMB.embed_query("alpha deploy")
    assert {r.scope for r, _ in s.search(q, scope_prefix="/ops")} == {"/ops/deploy", "/ops/deploy/prod"}
    assert {r.scope for r, _ in s.search(q, scope_prefix="/ops/deploy/prod")} == {"/ops/deploy/prod"}
    assert len(s.search(q, categories=["checklist", "notes"])) == 2
    assert [r.scope for r, _ in s.search(q, metadata_filter={"team": "a"}, scope_prefix="/finance")] == ["/finance"]
    top, score = s.search(q, limit=1)[0]
    assert top.content.startswith("alpha deploy") and 0 < score <= 1
    assert s.search(q, min_score=1.01) == []


def test_filters_apply_before_ranking_even_past_the_page(server_factory):
    srv = server_factory(DIM, max_top_k=5)
    s = HolographicMemoryStorage(HmsConfig(url=srv.url, tenant_id=srv.tenant(), input_dim=DIM, max_fetch=5))
    s.save([rec(f"alpha match {i}", "/a", metadata={"k": "no"}) for i in range(12)])
    s.save([rec("zzz rare needle", "/b", metadata={"k": "yes"})])
    hits = s.search(EMB.embed_query("alpha match"), metadata_filter={"k": "yes"}, limit=3)
    assert [r.content for r, _ in hits] == ["zzz rare needle"]


def test_roundtrip_fields_update_and_get(server):
    s = HolographicMemoryStorage(cfg(server))
    r = rec("remember me", "/x/y", ["c1", "c2"], {"k": 1, "nested": {"a": 1}}, importance=0.9, source="u1", private=True)
    s.save([r])
    got = s.get_record(r.id)
    assert (got.content, got.scope, got.categories, got.metadata) == ("remember me", "/x/y", ["c1", "c2"], r.metadata)
    assert (got.importance, got.source, got.private) == (0.9, "u1", True)
    assert abs((got.created_at - r.created_at).total_seconds()) < 1e-3
    s.update(got.model_copy(update={"content": "changed", "embedding": EMB.embed_query("changed")}))
    assert s.get_record(r.id).content == "changed" and s.count() == 1
    assert s.get_record("missing") is None


def test_scopes_categories_count_and_ordering(server):
    s = HolographicMemoryStorage(cfg(server))
    base = datetime.utcnow()
    s.save([
        rec("one", "/a/b", ["p"], created_at=base - timedelta(days=2)),
        rec("two", "/a/c", ["p", "q"], created_at=base - timedelta(days=1)),
        rec("three", "/d", ["q"], created_at=base),
    ])
    assert s.count() == 3 and s.count("/a") == 2
    assert s.list_scopes("/") == ["/a", "/d"] and s.list_scopes("/a") == ["/a/b", "/a/c"]
    assert s.list_categories() == {"p": 2, "q": 2}
    info = s.get_scope_info("/a")
    assert (info.record_count, info.categories, info.child_scopes) == (2, ["p", "q"], ["/a/b", "/a/c"])
    assert info.oldest_record < info.newest_record
    assert [r.content for r in s.list_records()] == ["three", "two", "one"]
    assert [r.content for r in s.list_records(limit=1, offset=1)] == ["two"]


def test_delete_variants_and_reset(server):
    s = HolographicMemoryStorage(cfg(server))
    base = datetime.utcnow()
    old = rec("old", "/a", ["x"], {"k": 1}, created_at=base - timedelta(days=10))
    s.save([old, rec("new", "/a", ["x"], {"k": 2}), rec("other", "/b", ["y"]), rec("more", "/b/c", ["y"])])
    deleted = s.delete(record_ids=[old.id, "ghost"])
    assert deleted == 1
    deleted = s.delete(scope_prefix="/b")
    assert deleted == 2
    deleted = s.delete(categories=["x"], metadata_filter={"k": 1})
    assert deleted == 0
    s.save([old])
    deleted = s.delete(scope_prefix="/a", older_than=base - timedelta(days=1))
    assert deleted == 1
    deleted = s.delete(categories=["x"])
    assert deleted == 1
    s.save([rec("a", "/p"), rec("b", "/q")])
    s.reset("/p")
    assert s.count() == 1
    s.reset()
    assert s.count() == 0


def test_listing_warns_when_truncated(server_factory):
    srv = server_factory(DIM, max_top_k=5)
    s = HolographicMemoryStorage(HmsConfig(url=srv.url, tenant_id=srv.tenant(), input_dim=DIM, max_fetch=5))
    s.save([rec(f"n{i}", "/s") for i in range(4)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert s.count() == 4
    s.save([rec("n4", "/s")])
    with pytest.warns(UserWarning, match="truncated"):
        assert s.count() == 5


async def test_async_paths(server):
    s = HolographicMemoryStorage(cfg(server))
    await s.asave([rec("async alpha", "/a", ["c"]), rec("async beta", "/b", ["c"])])
    hits = await s.asearch(EMB.embed_query("async alpha"), scope_prefix="/a")
    assert [r.content for r, _ in hits] == ["async alpha"]
    assert await s.adelete(categories=["c"]) == 2
    assert s.count() == 0


def test_unified_memory_end_to_end(server):
    storage = HolographicMemoryStorage(cfg(server))
    mem = Memory(storage=storage, embedder=EMB.embed_documents, llm="gpt-4o-mini", consolidation_threshold=1.0)
    kw = dict(scope="/team", categories=["fact"], importance=0.7)
    mem.remember("The release train departs every Thursday", **kw)
    mem.remember("Invoices are paid on the first of the month", **kw)
    mem.remember("Secret rota", scope="/team", categories=["fact"], importance=0.5, source="bob", private=True)
    hits = mem.recall("release train Thursday", depth="shallow", limit=2, source="alice")
    assert hits[0].record.content.startswith("The release train")
    assert all(m.record.content != "Secret rota" for m in hits)
    assert any(m.record.content == "Secret rota" for m in mem.recall("Secret rota", depth="shallow", include_private=True))
    assert mem.info("/team").record_count == 3
    assert mem.forget(scope="/team", categories=["fact"]) == 3
    assert mem.list_records() == []


def test_knowledge_storage_through_knowledge(server):
    storage = HolographicKnowledgeStorage(config=cfg(server))
    know = Knowledge(
        collection_name="docs",
        sources=[StringKnowledgeSource(content="The aurora project ships in March. Billing runs monthly.")],
        storage=storage,
    )
    know.add_sources()
    out = know.query(["aurora project ships"], results_limit=3, score_threshold=0.0)
    assert out and "aurora" in out[0]["content"]
    assert storage.search(["aurora project ships"], score_threshold=1.01) == []
    storage.reset()
    assert storage.search(["aurora"], score_threshold=0.0) == []


def test_knowledge_metadata_filter_and_dedup(server):
    storage = HolographicKnowledgeStorage(config=cfg(server))
    storage.save_with(["alpha one", "alpha two"], {"src": "a"})
    storage.save_with(["alpha three"], {"src": "b"})
    storage.save_with(["alpha one"], {"src": "a"})
    got = storage.search(["alpha"], limit=10, metadata_filter={"src": "a"}, score_threshold=0.0)
    assert sorted(r["content"] for r in got) == ["alpha one", "alpha two"]
    assert got[0]["metadata"] == {"src": "a"}


def test_crew_kickoff_recalls_from_holographic_memory(server):
    from crewai import Agent, Crew, Task
    from crewai.llms.base_llm import BaseLLM

    class Scripted(BaseLLM):
        def call(self, messages, tools=None, callbacks=None, available_functions=None,
                 from_task=None, from_agent=None, response_model=None):
            text = messages if isinstance(messages, str) else " ".join(str(m.get("content", "")) for m in messages)
            return f"Thought: ok\nFinal Answer: saw={'Bluebird' in text}"

        def supports_function_calling(self):
            return False

    storage = HolographicMemoryStorage(cfg(server))
    llm = Scripted(model="scripted")
    mem = Memory(storage=storage, embedder=EMB.embed_documents, llm=llm, consolidation_threshold=1.0)
    mem.remember("The launch codename is Bluebird", scope="/project", categories=["fact"], importance=0.9)
    agent = Agent(role="Analyst", goal="Answer", backstory="Knows.", llm=llm)
    task = Task(description="What is the launch codename?", expected_output="codename", agent=agent)
    out = Crew(agents=[agent], tasks=[task], memory=mem).kickoff()
    assert "saw=True" in str(out)
    assert storage.count() >= 2  # the crew saved the task result back through the backend
