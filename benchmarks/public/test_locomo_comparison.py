# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest
import tiktoken

from locomo_comparison import cap_prediction, complete_grade, prepare


ENCODING = tiktoken.get_encoding("o200k_base")


def prediction(memories):
    return {
        "question_id": "conv0_q0",
        "question": "This question must not affect selection.",
        "ground_truth_answer": "Neither may this reference answer.",
        "retrieval": {"search_results": [{"memory": text, "id": str(i)} for i, text in enumerate(memories)], "total_results": len(memories)},
        "cutoff_results": {"top_200": {"score": 1, "generated_answer": "stale"}},
    }


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_exact_budget_boundary(offset):
    text = "A fact with a precisely measured token length."
    cost = len(ENCODING.encode(text))
    source = prediction([text])
    before = copy.deepcopy(source)
    capped, old_cost, new_cost = cap_prediction(source, ENCODING, cost + offset)
    assert source == before
    assert old_cost == cost
    assert new_cost == (cost if offset >= 0 else 0)
    assert len(capped["retrieval"]["search_results"]) == (1 if offset >= 0 else 0)
    assert "cutoff_results" not in capped


def test_overflow_skip_preserves_later_fitting_fact_and_ignores_grades():
    source = prediction(["first", "long " * 20, "last"])
    budget = len(ENCODING.encode("first")) + len(ENCODING.encode("last"))
    result, _, used = cap_prediction(source, ENCODING, budget)
    assert [hit["id"] for hit in result["retrieval"]["search_results"]] == ["0", "2"]
    assert used == budget
    source.update(question="different", ground_truth_answer="different", cutoff_results={"top_200": {"score": 0}})
    other, _, _ = cap_prediction(source, ENCODING, budget)
    assert result["retrieval"] == other["retrieval"]


def test_prepare_checks_completeness_before_writing_and_preserves_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    original = json.dumps(prediction(["a fact"]))
    (source / "conv0_q0.json").write_text(original)
    target = tmp_path / "matched"
    with pytest.raises(ValueError, match="expected 2"):
        prepare(source, target, ENCODING, 7000, 2)
    assert not target.exists()
    receipt = prepare(source, target, ENCODING, 7000, 1)
    assert receipt["actual_questions"] == 1
    assert receipt["changed_retrievals"] == 0
    assert (source / "conv0_q0.json").read_text() == original
    assert "cutoff_results" not in json.loads((target / "conv0_q0.json").read_text())
    with pytest.raises(FileExistsError):
        prepare(source, target, ENCODING, 7000, 1)
    with pytest.raises(ValueError, match="overlap"):
        prepare(source, source / "nested", ENCODING, 7000, 1)


@pytest.mark.parametrize("bad_memory", [None, 12, {}, []])
def test_rejects_invalid_memory_instead_of_silently_dropping_it(bad_memory):
    with pytest.raises(ValueError, match="memory string"):
        cap_prediction(prediction([bad_memory]), ENCODING, 7000)


def test_empty_retrieval_is_preserved_and_reported(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "conv0_q0.json").write_text(json.dumps(prediction([])))
    report = prepare(source, None, ENCODING, 7000, 1)
    assert report["empty_retrievals_after"] == 1
    assert report["tokens_after"]["max"] == 0


@pytest.mark.parametrize("score,answer", [(0, ""), (0, "wrong"), (1, "right")])
def test_reuses_only_identical_retrieval_independent_of_grade(tmp_path, score, answer):
    source = tmp_path / "source"
    source.mkdir()
    data = prediction(["short", "long " * 20])
    data["cutoff_results"] = {"top_200": {"score": score, "generated_answer": answer, "memories_evaluated": 2}}
    (source / "conv0_q0.json").write_text(json.dumps(data))
    unchanged = tmp_path / "unchanged"
    report = prepare(source, unchanged, ENCODING, 7000, 1, reuse_unchanged=True)
    assert report["reused_grades"] == 1
    assert json.loads((unchanged / "conv0_q0.json").read_text())["cutoff_results"] == data["cutoff_results"]
    changed = tmp_path / "changed"
    report = prepare(source, changed, ENCODING, len(ENCODING.encode("short")), 1, reuse_unchanged=True)
    assert report["reused_grades"] == 0
    assert "cutoff_results" not in json.loads((changed / "conv0_q0.json").read_text())


def test_incomplete_grades_cannot_be_reused(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "conv0_q0.json").write_text(json.dumps(prediction(["short"])))
    with pytest.raises(ValueError, match="incomplete"):
        prepare(source, tmp_path / "dest", ENCODING, 7000, 1, reuse_unchanged=True)
    assert not (tmp_path / "dest").exists()


@pytest.mark.parametrize("score", [None, True, "1", float("nan"), float("inf"), -0.01, 0.5, 1.01])
def test_reuse_rejects_scores_outside_official_binary_protocol(score):
    data = {"cutoff_results": {"top_200": {"score": score, "generated_answer": "", "memories_evaluated": 10}}}
    with pytest.raises(ValueError, match="incomplete"):
        complete_grade(data, 10)


@pytest.mark.parametrize("count", [9, 10, 11])
def test_reuse_requires_exact_evaluated_memory_count(count):
    data = {"cutoff_results": {"top_200": {"score": 0, "generated_answer": "", "memories_evaluated": count}}}
    if count == 10:
        assert complete_grade(data, 10) == data["cutoff_results"]
    else:
        with pytest.raises(ValueError, match="incomplete"):
            complete_grade(data, 10)
