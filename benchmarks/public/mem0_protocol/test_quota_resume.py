# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import hashlib
import json

import pytest

from quota_resume import CachedAnswerer, judge_saved_answer, prepare, recovery_action, request_hash


def result(score=0, answer=""):
    return {"question_id": "conv0_q0", "conversation_idx": 0, "category": 4,
            "question": "Where?", "ground_truth_answer": "orchard",
            "retrieval": {"search_results": [{"id": "a", "memory": "At the orchard."}]},
            "cutoff_results": {"top_200": {"score": score, "generated_answer": answer, "memories_evaluated": 1}}}


def ok(reason="stop"):
    return {"status": 200, "finish_reason": [reason]}


QUOTA = {"status": 429, "error": "credit_balance_exhausted"}


@pytest.mark.parametrize("score,answer,finish", [(0, "", "length"), (0, "", "stop"),
                                                (0, "wrong", "stop"), (1, "orchard", "stop")])
def test_completed_grade_is_preserved_independent_of_outcome(score, answer, finish):
    data = result(score, answer)
    assert recovery_action(data, [ok(finish)], [ok()]) == "preserve_completed"
    assert recovery_action(data, [], [], data["cutoff_results"]) == "preserve_inherited"


@pytest.mark.parametrize("answer,finish", [("orchard", "stop"), ("", "length"), ("", "stop")])
def test_successful_answer_is_not_regenerated_when_only_judge_failed(answer, finish):
    assert recovery_action(result(0, answer), [ok(finish)], [QUOTA] * 5) == "judge_only_preserve_answer"


@pytest.mark.parametrize("score", [0, 1])
def test_quota_failure_recovery_does_not_select_by_placeholder_grade(score):
    assert recovery_action(result(score, ""), [QUOTA] * 5, [ok()]) == "fresh_answer_and_judge_quota_failure"
    data = result(); data.pop("cutoff_results")
    assert recovery_action(data, [], []) == "fresh_answer_and_judge_not_attempted"
    assert recovery_action(data, [QUOTA], []) == "fresh_answer_and_judge_quota_failure"


@pytest.mark.parametrize("answers,judges", [([], []), ([ok()], []), ([ok(), QUOTA], [ok()]),
                                           ([ok(), ok()], [ok()]), ([{"status": 500}], [ok()])])
def test_uncertain_or_mismatched_transport_does_not_clear_a_grade(answers, judges):
    with pytest.raises(ValueError):
        recovery_action(result(), answers, judges)


def test_failed_answer_transport_cannot_explain_nonempty_saved_answer():
    with pytest.raises(ValueError):
        recovery_action(result(0, "saved nonempty answer"), [QUOTA], [QUOTA])


def test_cached_answer_binds_prompt_and_parameters():
    async def run():
        cached = CachedAnswerer("", request_hash("", "original prompt"))
        assert await cached.generate("", "original prompt") == ""
        with pytest.raises(ValueError):
            await cached.generate("", "changed prompt")
        with pytest.raises(ValueError):
            await cached.generate("", "original prompt", max_tokens=8192)
        assert cached.calls == 1
    asyncio.run(run())


def test_judge_only_replay_preserves_answer_and_input_even_when_empty():
    class Judge:
        calls = 0

        async def generate_structured(self, **kwargs):
            self.calls += 1
            return {"label": "WRONG", "reasoning": "synthetic empty answer"}

    async def official_apply(data, qa, index, answerer, judge, cutoffs, evidence_lookup):
        assert cutoffs == [200] and evidence_lookup is None
        assert "cutoff_results" not in data
        answer = await answerer.generate(system="", user="exact prompt")
        judgment = await judge.generate_structured(system="judge", user=answer)
        data["cutoff_results"] = {"top_200": {"score": 0, "generated_answer": answer,
                                             "memories_evaluated": 1, "reason": judgment["reasoning"]}}

    async def run():
        data = result()
        before = copy.deepcopy(data)
        judge = Judge()
        replayed = await judge_saved_answer(data, request_hash("", "exact prompt"), judge, official_apply)
        assert data == before and judge.calls == 1
        assert replayed["cutoff_results"]["top_200"]["generated_answer"] == ""
    asyncio.run(run())


def recovery_fixture(tmp_path, extra_success=False):
    source = tmp_path / "original"
    source.mkdir()
    frozen_source = tmp_path / "development"
    frozen_source.mkdir()
    signature = {"harness_commit": "pinned", "server_binary_sha256": "binary",
                 "shim_sha256": "shim", "retrieval_config_sha256": "config"}
    manifest = {"hms": {"dirty": False, "server_binary_sha256": "binary"},
                "harness": {"commit": "pinned"}, "adapter": {"shim_sha256": "shim"},
                "retrieval_config_sha256": "config", "llm": {"spend_usd": 7}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    audit = {"summary": {"ambiguous_questions": 0}, "source_prediction_directory": str(source),
             "source_file_sha256": {}, "questions": {}}
    lines = []
    for index, (answer, a_call, j_call) in enumerate([
        ("", ok("length"), ok()), ("saved answer", ok(), QUOTA), ("", QUOTA, QUOTA),
    ]):
        data = result(answer=answer)
        qid = f"conv0_q{index}"
        data["question_id"] = qid
        path = source / f"{qid}.json"
        path.write_text(json.dumps(data))
        audit["source_file_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        ids = []
        for kind, call in [("answer", a_call), ("judge", j_call)]:
            call = {**call, "request": {"model": "gpt-5", "max_completion_tokens": 4096},
                    "prompt_sha256": request_hash("", f"{kind}:{qid}"), "resolved_model": "gpt-5-snapshot"}
            if kind == "judge":
                call["request"]["response_format"] = {"type": "json_object"}
            line = json.dumps(call)
            lines.append(line)
            ids.append(f"line{len(lines)}:" + hashlib.sha256(line.encode()).hexdigest())
        audit["questions"][qid] = {"action": recovery_action(data, [a_call], [j_call]),
                                   "answer_prompt_sha256": request_hash("", f"answer:{qid}"),
                                   "judge_prompt_sha256": request_hash("", f"judge:{qid}"),
                                   "answer_transport_ids": [ids[0]], "judge_transport_ids": [ids[1]]}
    if extra_success:
        lines.append(json.dumps({"status": 200, "resolved_model": "gpt-5-snapshot", "prompt_sha256": "unassigned",
                                 "request": {"model": "gpt-5", "max_completion_tokens": 4096}}))
    log = tmp_path / "calls.jsonl"
    log.write_text("\n".join(lines) + "\n")
    audit["calls_sha256"] = hashlib.sha256(log.read_bytes()).hexdigest()
    audit["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit))
    frozen = {"binding": signature, "resolved_model": "gpt-5-snapshot", "source": str(frozen_source)}
    render = lambda data: (request_hash("", "answer:" + data["question_id"]),
                           request_hash("", "judge:" + data["question_id"]))
    return audit_path, hashlib.sha256(audit_path.read_bytes()).hexdigest(), manifest_path, log, frozen, render, source


class ReplayJudge:
    def __init__(self):
        self.calls = 0
        self.spend = 0.0

    async def generate_structured(self, **kwargs):
        self.calls += 1
        self.spend += .25
        return {"label": "CORRECT"}


async def replay_official(data, qa, index, answerer, judge, cutoffs, evidence_lookup):
    assert "cutoff_results" not in data
    assert cutoffs == [200] and evidence_lookup is None
    answer = await answerer.generate(system="", user="answer:" + data["question_id"])
    judgment = await judge.generate_structured(system="", user="judge:" + data["question_id"])
    data["cutoff_results"] = {"top_200": {"score": int(judgment["label"] == "CORRECT"),
                                         "generated_answer": answer,
                                         "memories_evaluated": len(data["retrieval"]["search_results"])}}


def test_recovery_copy_preserves_valid_grade_and_both_original_answers(tmp_path):
    audit, digest, manifest, log, frozen, render, source = recovery_fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    target = tmp_path / "resumed"
    judge = ReplayJudge()
    receipt = prepare(audit, digest, manifest, log, frozen, target, render,
                      judge=judge, official_apply=replay_official, replay_spend_usd=lambda: judge.spend)
    assert receipt["counts"] == {"preserve_completed": 1, "judge_only_preserve_answer": 1,
                                  "fresh_answer_and_judge_quota_failure": 1}
    assert json.loads((target / "conv0_q0.json").read_text())["cutoff_results"]["top_200"]["generated_answer"] == ""
    replayed = json.loads((target / "conv0_q1.json").read_text())["cutoff_results"]["top_200"]
    assert replayed["generated_answer"] == "saved answer" and replayed["score"] == 1
    assert judge.calls == 1 and receipt["replayed_judges"] == ["conv0_q1.json"]
    assert "cutoff_results" not in json.loads((target / "conv0_q2.json").read_text())
    assert all((source / name).read_bytes() == raw for name, raw in before.items())
    assert receipt["additional_cost_usd"] == .25 and receipt["remaining_final_cap_usd"] == 40.75


def test_judge_only_preparation_without_replay_cannot_publish_regenerating_copy(tmp_path):
    audit, digest, manifest, log, frozen, render, source = recovery_fixture(tmp_path)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    target = tmp_path / "resumed"
    with pytest.raises(ValueError, match="requires a budgeted judge"):
        prepare(audit, digest, manifest, log, frozen, target, render)
    assert not target.exists()
    assert all((source / name).read_bytes() == raw for name, raw in before.items())


def test_failed_or_input_changing_judge_replay_publishes_nothing(tmp_path):
    audit, digest, manifest, log, frozen, render, source = recovery_fixture(tmp_path)

    async def invalid_replay(data, *args):
        data["question"] = "changed"
        await replay_official(data, *args)

    target = tmp_path / "resumed"
    judge = ReplayJudge()
    with pytest.raises(ValueError, match="prediction inputs"):
        prepare(audit, digest, manifest, log, frozen, target, render,
                judge=judge, official_apply=invalid_replay, replay_spend_usd=lambda: judge.spend)
    assert not target.exists()


@pytest.mark.parametrize("spent", [-1, float("nan"), float("inf"), 42])
def test_invalid_or_exhausted_replay_allocation_stops_before_judge(tmp_path, spent):
    audit, digest, manifest, log, frozen, render, source = recovery_fixture(tmp_path)
    judge = ReplayJudge()
    target = tmp_path / "resumed"
    with pytest.raises(ValueError, match="spend"):
        prepare(audit, digest, manifest, log, frozen, target, render,
                judge=judge, official_apply=replay_official, replay_spend_usd=lambda: spent)
    assert judge.calls == 0 and not target.exists()


@pytest.mark.parametrize("changed", ["prediction", "log", "audit"])
def test_changed_provenance_aborts_before_destination_exists(tmp_path, changed):
    audit, digest, manifest, log, frozen, render, source = recovery_fixture(tmp_path)
    path = {"prediction": source / "conv0_q1.json", "log": log, "audit": audit}[changed]
    path.write_text(path.read_text() + " ")
    target = tmp_path / "resumed"
    with pytest.raises(ValueError):
        prepare(audit, digest, manifest, log, frozen, target, render)
    assert not target.exists()


def test_unassigned_successful_call_blocks_recovery(tmp_path):
    audit, digest, manifest, log, frozen, render, source = recovery_fixture(tmp_path, extra_success=True)
    target = tmp_path / "resumed"
    with pytest.raises(ValueError, match="successful call"):
        prepare(audit, digest, manifest, log, frozen, target, render)
    assert not target.exists()
