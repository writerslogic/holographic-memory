# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Run one scenario through real mem0ai and through the shim, and require the same result shapes.

mem0 is configured fully offline: a local qdrant path, the hashing embedder in place of a hosted one, and a
dummy OpenAI key that is never used because every add() passes infer=False.
"""

import os

import pytest

mem0 = pytest.importorskip("mem0")

from holographic_integrations.core import HashingEmbedder, HmsConfig  # noqa: E402
from holographic_integrations.mem0 import Memory as ShimMemory  # noqa: E402

DIM = 64


@pytest.fixture
def real(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unused")
    monkeypatch.setenv("MEM0_TELEMETRY", "False")
    monkeypatch.setenv("MEM0_DIR", str(tmp_path / "mem0dir"))
    from mem0.embeddings.base import EmbeddingBase
    from mem0.utils.factory import EmbedderFactory

    class Local(EmbeddingBase):
        def embed(self, text, memory_action=None):
            return HashingEmbedder(DIM).embed_query(text)

    monkeypatch.setattr(EmbedderFactory, "create", classmethod(lambda cls, provider, config, vector_config: Local()))
    from mem0 import Memory

    return Memory.from_config(
        {
            "vector_store": {"provider": "qdrant", "config": {"path": str(tmp_path / "q"), "collection_name": "c", "embedding_model_dims": DIM, "on_disk": False}},
            "history_db_path": str(tmp_path / "h.db"),
        }
    )


def shapes(value):
    """Key sets of every dict in a result, with value types for scalars, ignoring ids and timestamps."""
    if isinstance(value, dict):
        return {k: shapes(v) for k, v in value.items()}
    if isinstance(value, list):
        return sorted((shapes(v) for v in value), key=repr)
    return type(value).__name__ if value is not None else "None"


def scenario(m):
    out = {}
    out["add"] = m.add(
        [{"role": "user", "content": "I live in Oslo", "name": "ann"}, {"role": "assistant", "content": "Noted"}],
        user_id="u1", metadata={"src": "chat"}, infer=False,
    )
    mid = out["add"]["results"][0]["id"]
    out["get"] = m.get(mid)
    out["get_missing"] = m.get("does-not-exist")
    out["get_all"] = m.get_all(filters={"user_id": "u1"})
    out["search"] = m.search("I live in Oslo", filters={"user_id": "u1"}, threshold=0.0)
    out["search_explain_keys"] = sorted(m.search("I live in Oslo", filters={"user_id": "u1"}, threshold=0.0, explain=False)["results"][0])
    out["update"] = m.update(mid, text="I live in Bergen", metadata={"src": "edit"})
    out["after_update"] = m.get(mid)
    out["history"] = m.history(mid)
    out["delete"] = m.delete(mid)
    out["delete_all"] = m.delete_all(user_id="u1")
    out["empty"] = m.get_all(filters={"user_id": "u1"})
    return out


def test_result_shapes_match_real_mem0(real, server):
    shim = ShimMemory(hms=HmsConfig(url=server.url, tenant_id=server.tenant(), input_dim=DIM, embedder=HashingEmbedder(DIM)))
    a, b = scenario(real), scenario(shim)
    for key in a:
        assert shapes(a[key]) == shapes(b[key]), key
    # behaviour, not just shape
    assert [r["memory"] for r in a["get_all"]["results"]] and sorted(r["memory"] for r in a["get_all"]["results"]) == sorted(
        r["memory"] for r in b["get_all"]["results"]
    )
    assert [h["event"] for h in a["history"]] == [h["event"] for h in b["history"]]
    assert a["delete"] == b["delete"] and a["delete_all"] == b["delete_all"] and a["update"] == b["update"]
    shim.close()
