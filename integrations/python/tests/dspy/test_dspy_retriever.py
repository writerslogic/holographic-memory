# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import copy

import numpy as np
import pytest

dspy = pytest.importorskip("dspy")

from dspy.utils import DummyLM  # noqa: E402

from holographic_integrations.core import HashingEmbedder  # noqa: E402
from holographic_integrations.dspy import HolographicRetriever, HolographicRM  # noqa: E402

DIM = 64
FACTS = [
    "The telescope observed a distant galaxy last night",
    "Fresh pasta needs ripe tomato and basil",
    "The striker scored a late goal in the football match",
    "The compiler rejected the python code with a syntax error",
]


def retriever(server, **kw):
    return HolographicRetriever(url=server.url, tenant_id=server.tenant(), input_dim=DIM, **kw)


class RAG(dspy.Module):
    def __init__(self, retrieve):
        super().__init__()
        self.retrieve = retrieve
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.answer(context=context, question=question)


def test_prediction_shape_and_ranking(server):
    r = retriever(server, corpus=FACTS)
    pred = r("pasta with tomato and basil", k=2)
    assert isinstance(pred, dspy.Prediction)
    assert pred.passages[0] == FACTS[1]
    assert len(pred.passages) == len(pred.scores) == len(pred.ids) == len(pred.metadatas) == 2
    assert pred.scores[0] >= pred.scores[1]


def test_metadata_filters_and_threshold(server):
    r = retriever(server)
    r.add(FACTS, metadatas=[{"topic": t, "rank": i} for i, t in enumerate(["space", "food", "sport", "code"])])
    assert r("something", k=4, filter={"topic": "sport"}).passages == [FACTS[2]]
    got = r("something", k=4, filter={"rank": {"$gte": 2}}).metadatas
    assert {m["rank"] for m in got} <= {2, 3}
    strict = retriever(server, score_threshold=0.999)
    strict.add(FACTS)
    assert strict(FACTS[0]).passages == [FACTS[0]]
    assert strict("unrelated words entirely").passages == []


def test_delete_and_replace_by_id(server):
    r = retriever(server)
    r.add(["first"], ids=["x"])
    r.add(["second"], ids=["x"])
    assert r("second", k=3).passages == ["second"]
    assert r.delete(["x", "ghost"]) == 1
    assert r("second", k=3).passages == []


def test_real_program_with_dummy_lm(server):
    r = retriever(server, corpus=FACTS)
    lm = DummyLM([{"reasoning": "the context mentions a goal", "answer": "the striker"}])
    with dspy.context(lm=lm):
        out = RAG(r)("who scored the football goal")
    assert out.answer == "the striker"
    prompt = lm.history[-1]["messages"][-1]["content"]
    assert "late goal in the football match" in prompt


def test_program_deepcopy_and_state_survive_optimizer_style_copy(server):
    r = retriever(server, corpus=FACTS)
    prog = RAG(r)
    clone = prog.deepcopy()
    assert clone.retrieve is not prog.retrieve and clone.retrieve.config is r.config
    assert clone.retrieve("galaxy telescope", k=1).passages == [FACTS[0]]
    assert copy.deepcopy(r)("galaxy telescope", k=1).passages == [FACTS[0]]


def test_react_tool(server):
    r = retriever(server, corpus=FACTS)
    lm = DummyLM(
        [
            {"next_thought": "look it up", "next_tool_name": "search_memory", "next_tool_args": {"query": "telescope galaxy"}},
            {"next_thought": "done", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "found", "answer": "a galaxy"},
        ]
    )
    agent = dspy.ReAct("question -> answer", tools=[r.as_tool()])
    with dspy.context(lm=lm):
        out = agent(question="what did the telescope see?")
    assert out.answer == "a galaxy"
    assert "distant galaxy" in str(out.trajectory["observation_0"])


def test_legacy_rm_with_dspy_retrieve(server):
    r = retriever(server, corpus=FACTS)
    dspy.settings.configure(rm=HolographicRM(r))
    try:
        got = dspy.Retrieve(k=2)("pasta tomato basil")
        assert got.passages[0] == FACTS[1] and len(got.passages) == 2
    finally:
        dspy.settings.configure(rm=None)


def test_dspy_embedder_callable_is_accepted(server):
    h = HashingEmbedder(DIM)

    def embed(texts):  # shape of dspy.Embedder: list[str] -> ndarray
        return np.array(h.embed_documents(texts))

    r = HolographicRetriever(url=server.url, tenant_id=server.tenant(), embedder=embed, corpus=FACTS)
    assert r.config.input_dim == DIM
    assert r("football striker goal", k=1).passages == [FACTS[2]]


async def test_async_paths_and_acall(server):
    r = retriever(server)
    try:
        await r.aadd(FACTS, metadatas=[{"i": i} for i in range(4)])
        pred = await r.acall("compiler syntax error python", k=2)
        assert pred.passages[0] == FACTS[3]
        assert (await r.aforward("anything", k=1, filter={"i": 1})).passages == [FACTS[1]]
        assert await r.adelete([pred.ids[0]]) == 1
    finally:
        await r.aclose()


def test_config_and_embedder_conflicts(server):
    with pytest.raises(ValueError):
        HolographicRetriever(config=retriever(server).config, url="http://x")
