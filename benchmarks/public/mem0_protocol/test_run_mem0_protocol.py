# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from run_mem0_protocol import audit_completion, audit_outputs, protocol_exit_code, validate_run_budget


def prediction(qid, cutoff=None):
    row = {"question_id": qid, "question": "A synthetic question", "retrieval": {"search_results": []}}
    if cutoff is not None:
        row["cutoff_results"] = cutoff
    return row


def test_missing_prediction_cannot_be_reported_complete(tmp_path):
    (tmp_path / "q0.json").write_text(json.dumps(prediction("q0")))
    result = audit_completion(tmp_path, ["q0", "q1"], scored=False)
    assert not result["complete"] and result["missing"] == ["q1"]


@pytest.mark.parametrize("cutoff", [None, {}, {"top_200": {}}, {"top_200": {"score": 1}},
                                    {"top_200": {"score": float("nan"), "generated_answer": "answer"}}])
def test_absent_or_incomplete_scored_cutoff_fails(tmp_path, cutoff):
    (tmp_path / "q0.json").write_text(json.dumps(prediction("q0", cutoff)))
    result = audit_completion(tmp_path, ["q0"], scored=True)
    assert not result["complete"] and result["scored_questions"] == 0


def test_length_truncated_scored_wrong_answer_remains_a_complete_result(tmp_path):
    row = prediction("q0", {"top_200": {"score": 0.0, "generated_answer": "", "judgment": "WRONG",
                                         "memories_evaluated": 0}})
    (tmp_path / "q0.json").write_text(json.dumps(row))
    result = audit_completion(tmp_path, ["q0"], scored=True)
    assert result["complete"] and result["scored_questions"] == 1


def test_partial_scoring_and_out_of_scope_files_do_not_inflate_completion(tmp_path):
    (tmp_path / "q0.json").write_text(json.dumps(prediction("q0", {
        "top_200": {"score": 1.0, "generated_answer": "answer", "memories_evaluated": 0},
    })))
    (tmp_path / "q1.json").write_text(json.dumps(prediction("q1")))
    (tmp_path / "heldout.json").write_text("not opened by scope validation")
    result = audit_completion(tmp_path, ["q0", "q1"], scored=True)
    assert not result["complete"] and result["scored_questions"] == 1
    assert result["unexpected"] == ["heldout"] and result["invalid"] == ["q1"]


@pytest.mark.parametrize("score", [-1, 0.5, 2, True, float("inf"), float("nan")])
def test_completion_reuses_binary_grade_validation(tmp_path, score):
    row = prediction("q0", {"top_200": {"score": score, "generated_answer": "",
                                         "memories_evaluated": 0}})
    (tmp_path / "q0.json").write_text(json.dumps(row))
    audit = audit_completion(tmp_path, ["q0"], scored=True)
    assert not audit["complete"] and audit["scored_questions"] == 0


@pytest.mark.parametrize("count", [None, False, 0, 1, 2])
def test_completion_requires_exact_evaluated_memory_count(tmp_path, count):
    row = prediction("q0", {"top_200": {"score": 0, "generated_answer": "",
                                         "memories_evaluated": count}})
    row["retrieval"]["search_results"] = [{"memory": "source"}]
    (tmp_path / "q0.json").write_text(json.dumps(row))
    audit = audit_completion(tmp_path, ["q0"], scored=True)
    assert audit["complete"] is (type(count) is int and count == 1)


@pytest.mark.parametrize("budget", [0, 0.01, 48])
def test_lexical_budget_remains_supported_but_embedding_presets_fail_closed(budget):
    validate_run_budget({"max_k": 200, "max_context_tokens": 7000}, budget)
    for ordered in (False, True):
        config = {"embedding": {"model": "text-embedding-3-large", "dimensions": 1024},
                  "order_same_time": ordered}
        with pytest.raises(ValueError, match="embeddings are unavailable"):
            validate_run_budget(config, budget)


@pytest.mark.parametrize("budget", [-1, float("inf"), -float("inf"), float("nan"), True])
def test_nonfinite_or_invalid_run_budgets_are_rejected(budget):
    with pytest.raises(ValueError, match="finite nonnegative"):
        validate_run_budget({}, budget)


@pytest.mark.parametrize("ids", [[], ["../q"], ["q", "q"], [None], [{}]])
def test_invalid_scope_is_rejected(tmp_path, ids):
    with pytest.raises(ValueError):
        audit_completion(tmp_path, ids, scored=False)


def test_missing_question_is_not_a_valid_prediction(tmp_path):
    row = prediction("q0")
    row.pop("question")
    (tmp_path / "q0.json").write_text(json.dumps(row))
    assert not audit_completion(tmp_path, ["q0"], scored=False)["complete"]


def test_cap_failure_cannot_pass_when_the_harness_wrote_wrong_scores():
    audit = {"completion": {"complete": True}, "ingestion_failures": 0,
             "invalid_ingestion_checkpoints": [], "failed_after_retries_lines": 0,
             "empty_generated_answers": ["q0"]}
    assert protocol_exit_code(0, audit, {"blocked": 1}) == 2
    assert protocol_exit_code(0, audit, {"blocked": 0, "finish_reasons": {"length": 1}}) == 0
    assert protocol_exit_code(7, audit, {"blocked": 0}) == 7


def test_pinned_client_final_transport_retries_make_output_incomplete(tmp_path):
    log = tmp_path / "harness.log"
    log.write_text("Generation attempt 4/5 failed: transient\n"
                   "Generation attempt 5/5 failed: credit_balance_exhausted\n"
                   "Structured output attempt 5/5 failed: credit_balance_exhausted\n"
                   "Structured output attempt 5/5 timed out\n"
                   "Generation returned None (finish_reason=length)\n"
                   "Structured output parse error attempt 5/5: invalid model JSON\n")
    audit = audit_outputs(tmp_path, log)
    assert audit["failed_after_retries_lines"] == 3
    audit.update(completion={"complete": True}, ingestion_failures=0, invalid_ingestion_checkpoints=[])
    assert protocol_exit_code(0, audit, {"blocked": 0}) == 2
