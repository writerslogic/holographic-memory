# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest

from reuse_locomo_dev import COMMON, PINNED_COMMIT, freeze_source, prepare, retrieval_hash


def prediction(qid="conv0_q0", score=0, answer=""):
    return {"question_id": qid, "question": "Where?", "category": 4,
            "ground_truth_answer": "orchard", "reference_date": "January 2023", "evidence": ["turn1"],
            "retrieval": {"search_results": [{"id": "a", "memory": "At the orchard.", "score": 1,
                                               "created_at": "2023-01-01T00:00:00"},
                                              {"id": "b", "memory": "Another fact.", "score": 0.5,
                                               "created_at": "2023-01-01T00:00:00"}]},
            "cutoff_results": {"top_200": {"score": score, "generated_answer": answer, "memories_evaluated": 2}}}


def manifest(folder, scored, count):
    params = [{"model": "gpt-5", "max_completion_tokens": 4096},
              {"model": "gpt-5", "max_completion_tokens": 4096, "response_format": {"type": "json_object"}}]
    return {"benchmark": "locomo", "phase": "evaluate" if scored else "predict", "harness": {"commit": PINNED_COMMIT,
             "command": ["--categories", "1,2,3,4", *COMMON]},
            "warm": {"dirty": False, "server_binary_sha256": "binary"}, "adapter": {"shim_sha256": "shim"},
            "wrapper_exit_code": 0, "harness_exit_code": 0,
            "audit": {"completion": {"complete": True, "expected_questions": count,
                                      "scored_questions": count if scored else 0}},
            "llm": {"errors": 0, "blocked": 0, "calls": 2 * count if scored else 0,
                    "requested_params": {json.dumps(p): count for p in params} if scored else {},
                    "resolved_models": {"gpt-5-2025-08-07": 2 * count} if scored else {}},
            "retrieval_config": {"max_context_tokens": 7000, "max_k": 200, "order_same_time": True},
            "retrieval_config_sha256": "config", "retrieval_after": retrieval_hash(folder.parent)}


def setup(tmp_path, mutate=lambda data: None, score=0, answer=""):
    source = tmp_path / "old" / "predicted_old"
    fresh = tmp_path / "new" / "predicted_new"
    source.mkdir(parents=True)
    fresh.mkdir(parents=True)
    old = prediction(score=score, answer=answer)
    new = copy.deepcopy(old)
    new.pop("cutoff_results")
    mutate(new)
    (source / "conv0_q0.json").write_text(json.dumps(old))
    (fresh / "conv0_q0.json").write_text(json.dumps(new))
    heldout = prediction("conv3_q0")
    heldout.pop("cutoff_results")
    (fresh / "conv3_q0.json").write_text(json.dumps(heldout))
    old_manifest, new_manifest = tmp_path / "old.json", tmp_path / "new.json"
    old_manifest.write_text(json.dumps(manifest(source, True, 1)))
    new_manifest.write_text(json.dumps(manifest(fresh, False, 2)))
    return source, fresh, old_manifest, new_manifest


def render(data):
    return json.dumps([data["question"], data["retrieval"], data.get("prompt_extra")])


@pytest.mark.parametrize("score,answer", [(0, ""), (0, "wrong"), (1, "orchard")])
def test_reuses_all_grade_outcomes_and_preserves_both_input_trees(tmp_path, score, answer):
    source, fresh, old_manifest, new_manifest = setup(tmp_path, score=score, answer=answer)
    frozen = freeze_source(source, old_manifest, 1)
    before = {p: p.read_bytes() for folder in (source, fresh) for p in folder.iterdir()}
    target = tmp_path / "scorable"
    result = prepare(frozen, fresh, new_manifest, target, render, 2)
    assert result["reused_grades"] == 1
    reused = json.loads((target / "conv0_q0.json").read_text())["cutoff_results"]["top_200"]
    assert reused["score"] == score and reused["generated_answer"] == answer
    assert (target / "conv3_q0.json").read_bytes() == before[fresh / "conv3_q0.json"]
    assert all(p.read_bytes() == content for p, content in before.items())
    with pytest.raises(FileExistsError):
        prepare(frozen, fresh, new_manifest, target, render, 2)


@pytest.mark.parametrize("mutation", [
    lambda d: d["retrieval"]["search_results"].reverse(),
    lambda d: d["retrieval"]["search_results"][0].update(created_at="2024-01-01T00:00:00"),
    lambda d: d.update(question="When?"), lambda d: d.update(ground_truth_answer="park"),
    lambda d: d.update(category=2), lambda d: d.update(reference_date="2024"),
    lambda d: d.update(user_profile={"fact": "new"}), lambda d: d.update(evidence=["other"]),
    lambda d: d.update(prompt_extra="different rendered prompt"),
])
def test_changed_retrieval_or_official_inputs_require_fresh_scoring(tmp_path, mutation):
    source, fresh, old_manifest, new_manifest = setup(tmp_path, mutation)
    result = prepare(freeze_source(source, old_manifest, 1), fresh, new_manifest,
                     tmp_path / "scorable", render, 2)
    assert result["reused_grades"] == 0
    assert "cutoff_results" not in json.loads((tmp_path / "scorable/conv0_q0.json").read_text())


@pytest.mark.parametrize("mutation", [
    lambda d: d["retrieval_config"].update(order_same_time=False),
    lambda d: d["adapter"].update(shim_sha256="changed"),
    lambda d: d["llm"].update(blocked=1),
    lambda d: d["audit"]["completion"].update(complete=False),
    lambda d: d["harness"]["command"].append("--another-judge-mode"),
    lambda d: d.update(phase="full"),
])
def test_rejects_protocol_changes_before_writing(tmp_path, mutation):
    source, fresh, old_manifest, new_manifest = setup(tmp_path)
    frozen = freeze_source(source, old_manifest, 1)
    data = json.loads(new_manifest.read_text()); mutation(data); new_manifest.write_text(json.dumps(data))
    target = tmp_path / "scorable"
    with pytest.raises(ValueError):
        prepare(frozen, fresh, new_manifest, target, render, 2)
    assert not target.exists()


def test_frozen_grade_tampering_and_heldout_source_are_rejected(tmp_path):
    source, fresh, old_manifest, new_manifest = setup(tmp_path)
    frozen = freeze_source(source, old_manifest, 1)
    data = json.loads((source / "conv0_q0.json").read_text())
    data["cutoff_results"]["top_200"]["score"] = 1
    (source / "conv0_q0.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="frozen"):
        prepare(frozen, fresh, new_manifest, tmp_path / "scorable", render, 2)
    (source / "conv0_q0.json").rename(source / "conv3_q0.json")
    old_manifest.write_text(json.dumps(manifest(source, True, 1)))
    with pytest.raises(ValueError, match="development"):
        freeze_source(source, old_manifest, 1)


def test_unexpected_scored_heldout_or_symlink_is_rejected_before_copy(tmp_path):
    source, fresh, old_manifest, new_manifest = setup(tmp_path)
    frozen = freeze_source(source, old_manifest, 1)
    heldout = fresh / "conv3_q0.json"
    heldout.write_text(json.dumps(prediction("conv3_q0")))
    with pytest.raises(ValueError, match="already scored"):
        prepare(frozen, fresh, new_manifest, tmp_path / "scorable", render, 2)
    heldout.unlink(); heldout.symlink_to(source / "conv0_q0.json")
    with pytest.raises(ValueError, match="regular"):
        prepare(frozen, fresh, new_manifest, tmp_path / "scorable", render, 2)
    assert not (tmp_path / "scorable").exists()
