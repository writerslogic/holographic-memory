# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare quota-interrupted LoCoMo recovery from immutable transport evidence."""

import asyncio
import copy
import hashlib
import json
import math
import re
import shutil
import tempfile
from pathlib import Path

from reuse_locomo_dev import complete_grade, read_json


def request_hash(system, user):
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def transport_state(calls):
    if not calls:
        return "not_attempted"
    successes = [call for call in calls if call["status"] == 200]
    if not successes and all(call["status"] == 429 and call.get("error") == "credit_balance_exhausted" for call in calls):
        return "quota_failed"
    if len(successes) == 1 and calls[-1]["status"] == 200:
        return "completed"
    raise ValueError("transport provenance is ambiguous")


def recovery_action(data, answer_calls, judge_calls, inherited=None):
    grade = data.get("cutoff_results", {}).get("top_200")
    if grade is not None:
        complete_grade(data, len(data["retrieval"]["search_results"]))
    if inherited is not None:
        if data.get("cutoff_results") != inherited:
            raise ValueError("inherited grade changed")
        return "preserve_inherited"
    answer_state = transport_state(answer_calls)
    if answer_state == "not_attempted" and grade is None:
        return "fresh_answer_and_judge_not_attempted"
    if answer_state == "quota_failed" and (grade is None or grade["generated_answer"] == ""):
        return "fresh_answer_and_judge_quota_failure"
    if grade is not None and answer_state == "completed":
        judge_state = transport_state(judge_calls)
        if judge_state == "completed":
            return "preserve_completed"
        if judge_state == "quota_failed":
            return "judge_only_preserve_answer"
    raise ValueError("saved output lacks sufficient transport provenance")


class CachedAnswerer:
    def __init__(self, answer, expected_prompt_hash):
        self.answer = answer
        self.expected_prompt_hash = expected_prompt_hash
        self.calls = 0

    async def generate(self, system, user, **kwargs):
        if request_hash(system, user) != self.expected_prompt_hash or kwargs:
            raise ValueError("cached answer prompt or generation parameters changed")
        self.calls += 1
        return self.answer


async def judge_saved_answer(data, expected_prompt_hash, judge, official_apply):
    original = copy.deepcopy(data)
    cached = CachedAnswerer(data["cutoff_results"]["top_200"]["generated_answer"], expected_prompt_hash)
    qa = {"category": data["category"], "answer": data["ground_truth_answer"], "evidence": data.get("evidence", [])}
    result = copy.deepcopy(data)
    result.pop("cutoff_results", None)
    await official_apply(result, qa, data["conversation_idx"], cached, judge, [200], None)
    if cached.calls != 1 or result["cutoff_results"]["top_200"]["generated_answer"] != cached.answer:
        raise ValueError("official judge replay changed the cached answer")
    if data != original:
        raise ValueError("original result changed during replay")
    if ({key: value for key, value in result.items() if key != "cutoff_results"}
            != {key: value for key, value in original.items() if key != "cutoff_results"}):
        raise ValueError("official judge replay changed prediction inputs")
    complete_grade(result, len(result["retrieval"]["search_results"]))
    return result


def prepare(audit_path, audit_sha256, source_manifest, call_log, development_freeze,
            destination, render_hashes, *, judge=None, official_apply=None, replay_spend_usd=None):
    """Replay judges before publication; the getter measures spend beyond the source manifest."""
    audit, actual_hash = read_json(audit_path)
    if actual_hash != audit_sha256 or audit["summary"]["ambiguous_questions"]:
        raise ValueError("reviewed transport audit changed or remains ambiguous")
    if hashlib.sha256(call_log.read_bytes()).hexdigest() != audit["calls_sha256"]:
        raise ValueError("transport log changed")
    manifest, manifest_hash = read_json(source_manifest)
    if manifest_hash != audit["manifest_sha256"]:
        raise ValueError("partial manifest changed")
    original_spend = manifest["llm"]["spend_usd"]
    if (type(original_spend) not in (int, float) or not math.isfinite(original_spend)
            or not 0 <= original_spend <= 48):
        raise ValueError("original spend must fit the finite recovery allocation")
    signature = development_freeze["binding"]
    if (manifest["warm"]["dirty"] or manifest["harness"]["commit"] != signature["harness_commit"]
            or manifest["warm"]["server_binary_sha256"] != signature["server_binary_sha256"]
            or manifest["adapter"]["shim_sha256"] != signature["shim_sha256"]
            or manifest["retrieval_config_sha256"] != signature["retrieval_config_sha256"]):
        raise ValueError("partial attempt uses different frozen quality sources")
    calls, by_hash = {}, {}
    for number, line in enumerate(call_log.read_text().splitlines(), 1):
        call = json.loads(line)
        identifier = f"line{number}:" + hashlib.sha256(line.encode()).hexdigest()
        calls[identifier] = call
        by_hash.setdefault(call["prompt_sha256"], []).append(identifier)
        expected = {"model": "gpt-5", "max_completion_tokens": 4096}
        if call["request"].get("response_format"):
            expected["response_format"] = {"type": "json_object"}
        if call["request"] != expected:
            raise ValueError("recorded model parameters differ")
        if call["status"] == 200 and call.get("resolved_model") != development_freeze["resolved_model"]:
            raise ValueError("recorded model snapshot differs")
        if call["status"] != 200 and (call["status"] != 429 or call.get("error") != "credit_balance_exhausted"):
            raise ValueError("unexpected transport failure requires separate diagnosis")
    source = Path(audit["source_prediction_directory"]).resolve(strict=True)
    destination = destination.resolve()
    for origin in (source, Path(development_freeze["source"]).resolve()):
        if destination == origin or origin in destination.parents or destination in origin.parents:
            raise ValueError("recovery destination must not overlap original inputs")
    if destination.exists():
        raise FileExistsError("recovery destination exists")
    prepared, counts, assigned, replays = {}, {}, set(), {}
    for name, expected_hash in audit["source_file_sha256"].items():
        if not re.fullmatch(r"conv\d+_q\d+\.json", name):
            raise ValueError("invalid prediction filename")
        data, data_hash = read_json(source / name)
        if data_hash != expected_hash or data["question_id"] != Path(name).stem:
            raise ValueError("immutable prediction changed")
        proof = audit["questions"][data["question_id"]]
        answer_hash, judge_hash = render_hashes(data)
        if (answer_hash, judge_hash) != (proof["answer_prompt_sha256"], proof["judge_prompt_sha256"]):
            raise ValueError("rendered official prompts differ from transport proof")
        answer_ids, judge_ids = by_hash.get(answer_hash, []), by_hash.get(judge_hash, [])
        if answer_ids != proof["answer_transport_ids"] or judge_ids != proof["judge_transport_ids"]:
            raise ValueError("transport associations changed")
        inherited = None
        if proof["action"] == "preserve_inherited":
            old, old_hash = read_json(Path(development_freeze["source"]) / name)
            if old_hash != development_freeze["source_sha256"].get(name):
                raise ValueError("frozen development grade changed")
            if (old["retrieval"]["search_results"] != data["retrieval"]["search_results"]
                    or render_hashes(old) != (answer_hash, judge_hash)):
                raise ValueError("inherited retrieval or prompt changed")
            inherited = old["cutoff_results"]
        action = recovery_action(data, [calls[key] for key in answer_ids], [calls[key] for key in judge_ids], inherited)
        if action != proof["action"]:
            raise ValueError("recomputed recovery action differs from reviewed audit")
        assigned.update(answer_ids + judge_ids)
        counts[action] = counts.get(action, 0) + 1
        if action == "judge_only_preserve_answer":
            replays[name] = (data, answer_hash, judge_hash)
        elif not action.startswith("preserve_"):
            data.pop("cutoff_results", None)
        prepared[name] = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if set(prepared) != {path.name for path in source.glob("conv*_q*.json")}:
        raise ValueError("prediction scope differs from the immutable audit")
    unmatched = [key for key in calls if key not in assigned]
    if any(calls[key]["status"] == 200 for key in unmatched):
        raise ValueError("a successful call has no verified question association")
    entries = list(source.iterdir())
    if any(path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024
           for path in entries):
        raise ValueError("prediction inputs must be bounded regular files")
    additional_spend = 0.0
    if replays:
        if judge is None or not callable(official_apply) or not callable(replay_spend_usd):
            raise ValueError("judge-only recovery requires a budgeted judge and pinned official scoring callback")

        def measured_spend():
            value = replay_spend_usd()
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not 0 <= value <= 48 - original_spend):
                raise ValueError("judge replay spend exceeds its finite allocation or is invalid")
            return float(value)

        additional_spend = measured_spend()
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise ValueError("synchronous recovery preparation must run outside an active event loop")

        async def replay_saved_answers():
            nonlocal additional_spend
            for name, (data, answer_hash, judge_hash) in replays.items():
                if original_spend + additional_spend >= 48:
                    raise ValueError("judge replay allocation is exhausted")
                replayed = await judge_saved_answer(data, answer_hash, judge, official_apply)
                spent = measured_spend()
                if spent < additional_spend:
                    raise ValueError("judge replay spend cannot decrease")
                additional_spend = spent
                if render_hashes(replayed) != (answer_hash, judge_hash):
                    raise ValueError("official replay changed the verified answer or judge prompt")
                prepared[name] = json.dumps(replayed, ensure_ascii=False, indent=2) + "\n"

        asyncio.run(replay_saved_answers())
    receipt = {"audit_sha256": audit_sha256, "original_predictions": str(source),
               "counts": counts, "unmatched_quota_transport_ids": unmatched,
               "replayed_judges": sorted(replays),
               "original_cost_usd": original_spend,
               "additional_cost_usd": additional_spend,
               "remaining_final_cap_usd": round(48 - original_spend - additional_spend, 6),
               "rule": "Preserve completed grades and successful answers; recovery is selected only by transport provenance."}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".locomo-quota-resume-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "predictions"
        stage.mkdir()
        for path in entries:
            shutil.copy2(path, stage / path.name)
        for name, text in prepared.items():
            (stage / name).write_text(text, encoding="utf-8")
        (stage / "_quota_resume_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        stage.rename(destination)
    return receipt
