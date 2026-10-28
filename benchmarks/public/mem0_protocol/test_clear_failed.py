# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from clear_failed import clear_failed


def setup_run(root, llm, answer="", qid="q0"):
    predictions = root / "harness-out/predicted_test"
    predictions.mkdir(parents=True)
    path = predictions / "q0.json"
    path.write_text(json.dumps({"question_id": "q0", "retrieval": {"search_results": []},
                               "cutoff_results": {"top_200": {"score": 0, "generated_answer": answer}}}))
    (root / "manifest-test.json").write_text(json.dumps({
        "audit": {"empty_generated_answers": [qid]}, "llm": llm,
    }))
    return path


@pytest.mark.parametrize("finish", ["length", "stop"])
def test_successful_empty_completion_is_preserved_even_when_other_calls_failed(tmp_path, finish):
    path = setup_run(tmp_path, {"calls": 3, "errors": 1, "blocked": 1, "finish_reasons": {finish: 1}})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="successful or unclassified"):
        clear_failed(tmp_path)
    assert path.read_bytes() == before


def test_entirely_failed_attempt_can_be_cleared_idempotently(tmp_path):
    path = setup_run(tmp_path, {"calls": 2, "errors": 1, "blocked": 1})
    assert clear_failed(tmp_path) == 1
    assert json.loads(path.read_text()) == {"question_id": "q0", "retrieval": {"search_results": []}}
    assert clear_failed(tmp_path) == 0


def test_failure_receipt_cannot_clear_a_nonempty_answer(tmp_path):
    path = setup_run(tmp_path, {"calls": 2, "errors": 1, "blocked": 1}, answer="A scored answer")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="nonempty"):
        clear_failed(tmp_path)
    assert path.read_bytes() == before


def test_unsafe_question_id_is_rejected_before_any_change(tmp_path):
    path = setup_run(tmp_path, {"calls": 2, "errors": 1, "blocked": 1}, qid="../q0")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="question IDs"):
        clear_failed(tmp_path)
    assert path.read_bytes() == before
