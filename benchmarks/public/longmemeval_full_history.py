# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13"
# dependencies = ["tiktoken==0.14.0", "tokenizers==0.23.3", "openai==3.27.0", "python-dotenv==1.2.4"]
# ///
"""Freeze and evaluate all available dev user history under a larger reader budget."""

import argparse
import copy
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import tiktoken

import longmemeval_reader_api as reader
import longmemeval_user_first as user_first
from check_longmemeval_answers import artifact_path, recompute_scores, require, same, wilson
from longmemeval_answer_evidence import reader_payload
from longmemeval_dev_sources import load_dev_sources

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks/results"
DEFAULT_WORK = RESULTS / "longmemeval_full_history_v1"
SYSTEM = "matched_gpt54"
ARM = "full_history"
CONTEXT_CAP = 131072
FRAMING_RESERVE = 512
NEW_SPEND_CAP_USD = 40.0
REASONING_RULES = (
    "\nApply these rules to the evidence: Apply a requested temporal window to the relation being asked about. "
    "Payment or booking time can differ from event time; purchase time can differ from later use. "
    "Count distinct entities or events once even when several sources mention them, and establish that the "
    "date of the requested relation falls within the requested window. Calling an item 'old' does not itself "
    "show that it was discarded, sold, replaced, or is no longer owned. Distinguish recommendations and plans "
    "from actual ownership and completed actions. Personalize recommendations using explicit likes, dislikes, "
    "and relevant history; topical generic advice alone does not establish personalization. Resolve relative "
    "or implicit time expressions against the observation date of their own source. Answer at the precision "
    "the evidence supports, including approximate elapsed time, rather than abstaining solely because an exact "
    "clock time is unavailable. Keep relation scopes distinct, such as a current apartment versus a district "
    "and event, fee, or payment dates. Use the plausible ordinary reading of the question and give a qualified "
    "direct answer when the facts support it. Give the requested fact directly. When the sources support "
    "only a bound or a qualified answer, state that bound and the qualifying uncertainty explicitly. "
    "Abstain when the requested fact lacks source support; do not fill gaps with unmentioned events.\n"
)


def make_protocol(base):
    settings = copy.deepcopy(base["systems"][SYSTEM])
    for key, expected in (("reader_model", "gpt-5.4-2026-03-05"), ("judge_model", "gpt-5.4-2026-03-05"),
                          ("reasoning_effort", "medium"), ("reader_max_output_tokens", 4096),
                          ("judge_max_output_tokens", 2048)):
        same(settings[key], expected, f"pinned {key} changed")
    return {
        "schema_version": 1, "dev_only": True, "heldout_run": False, "acceptance_correct": 100,
        "reader_prompt": base["reader_prompt"] + REASONING_RULES, "systems": {SYSTEM: settings},
        "judge_source": copy.deepcopy(base["judge_source"]), "token_cap": CONTEXT_CAP,
        "local_token_cap": CONTEXT_CAP - FRAMING_RESERVE, "framing_reserve_tokens": FRAMING_RESERVE,
        "source_policy": "all complete dev user observations and all available saved v1/v3 assistant turns",
        "automatic_transport_retries": 0, "reader_judge_concurrency": 1,
        "new_spend_cap_usd": NEW_SPEND_CAP_USD,
        "comparison_budget": "larger input context; not an equal-budget comparison",
        "limitations": ["Repeated-use dev only; no held-out claim.",
                        "Assistant availability is limited to saved v1/v3 complete turns.",
                        "Labeled user support does not establish semantic sufficiency.",
                        "A complete history exceeding the input budget is a selection failure; no source is truncated."],
    }


def serialize(question, groups, protocol):
    return protocol["reader_prompt"] + reader.canonical(reader_payload(question, {"source_groups": groups}))


def prepare_arm(question, groups, protocol, count_callback):
    require(type(protocol["token_cap"]) is int and protocol["token_cap"] > 0, "invalid context cap")
    reserve = protocol["framing_reserve_tokens"]
    require(type(reserve) is int and 0 <= reserve < protocol["token_cap"], "invalid framing reserve")
    require(protocol["local_token_cap"] == protocol["token_cap"] - reserve, "local input cap mismatch")
    require(isinstance(groups, list) and groups, "full history must contain source groups")
    copied = copy.deepcopy(groups)
    for group in copied:
        require(any(source["role"] == "user" for source in group["sources"]), "history group lacks a user turn")
        require(all(source["role"] in ("user", "assistant") and isinstance(source["text"], str)
                    for source in group["sources"]), "invalid history source")
    prompt = serialize(question, copied, protocol)
    tokens = count_callback(prompt)
    require(type(tokens) is int and tokens >= 0, "tokenizer returned invalid count")
    failed = tokens > protocol["local_token_cap"]
    arm = {
        "source_groups": copied, "reader_input": prompt, "reader_input_sha256": reader.digest(prompt),
        "retained_sources": reader.retained(copied), "input_tokens": tokens,
        "reader_cap": {"cap": protocol["token_cap"], "local_cap": protocol["local_token_cap"],
                       "input_tokens": tokens, "uncapped_input_tokens": tokens, "dropped_group_ids": [],
                       "selection_failure": failed,
                       "failure_reason": "complete_history_input_exceeds_cap" if failed else None,
                       "source_policy": "full_user_history_with_saved_assistants"},
        "reader": {"status": "selection_failure" if failed else "pending", "raw_text": "", "parsed": None},
        "judge": {"status": "pending", "raw_text": "", "score": False},
    }
    arm["scores"] = recompute_scores(arm, allow_incomplete=True)
    return arm


def reservation_usd(input_tokens, output_tokens):
    require(type(input_tokens) is int and input_tokens >= 0, "invalid input token count")
    require(type(output_tokens) is int and output_tokens > 0, "invalid output token limit")
    return (input_tokens + FRAMING_RESERVE) * 2.5e-6 + output_tokens * 15e-6


def experiment_call_ids(work, rows):
    return {reader.digest(f"full-history:{work.name}:{stage}:{row['qid']}")
            for row in rows for stage in ("reader", "judge")}


def variant_spend(ledger, call_ids):
    return sum(entry["cost_usd"] for entry in ledger["entries"] if entry.get("call_id") in call_ids)


def merge_ledgers(ledgers, caps):
    require(caps["reader_judge_dev"] == 120 and caps["total"] == 250 and caps["heldout400"] == 0,
            "global spend caps changed")
    entries = {}
    for ledger in ledgers:
        same(ledger["caps_usd"], caps, "ledger spend caps differ")
        for entry in ledger["entries"]:
            key, cost = entry["cache_key"], entry["cost_usd"]
            require(isinstance(key, str) and key and isinstance(cost, (int, float))
                    and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0, "invalid ledger entry")
            require(entry["category"] in caps and entry["category"] != "total", "invalid spend category")
            if key not in entries or cost > entries[key]["cost_usd"]:
                entries[key] = copy.deepcopy(entry)
    categories = {category: sum(entry["cost_usd"] for entry in entries.values() if entry["category"] == category)
                  for category in caps if category != "total"}
    return {"schema_version": 1, "caps_usd": copy.deepcopy(caps), "entries": list(entries.values()),
            "total_usd": sum(entry["cost_usd"] for entry in entries.values()), "by_category_usd": categories}


def refresh_ledger(work):
    paths = sorted({RESULTS / "spend_2026-10.json", *RESULTS.glob("longmemeval_dev_answers*_spend.json"),
                    *RESULTS.glob("longmemeval_*/spend.json"), work / "spend.json"})
    paths = [path for path in paths if path.exists()]
    primary = json.loads((RESULTS / "spend_2026-10.json").read_text())
    ledgers = [json.loads(path.read_text()) for path in paths]
    merged = merge_ledgers(ledgers, primary["caps_usd"])
    merged["observed_ledger_paths"] = [str(path.relative_to(ROOT)) for path in paths]
    reader.write(work / "spend.json", merged)
    return merged


def budget_rejection(ledger, call_ids, reserve, cap=NEW_SPEND_CAP_USD):
    require(isinstance(reserve, (float, int)) and not isinstance(reserve, bool)
            and math.isfinite(reserve) and reserve >= 0, "invalid cost reservation")
    if variant_spend(ledger, call_ids) + reserve > cap:
        return "new_variant_spend_cap"
    if ledger["total_usd"] + reserve > ledger["caps_usd"]["total"]:
        return "global_total_spend_cap"
    if ledger["by_category_usd"]["reader_judge_dev"] + reserve > ledger["caps_usd"]["reader_judge_dev"]:
        return "reader_judge_dev_spend_cap"
    return None


def guarded_api_call(api, protocol, req, call_id, work, call_ids, count_callback):
    ledger = refresh_ledger(work)
    key = reader.digest(reader.canonical({"request": req, "call_id": call_id,
                                          "protocol_sha256": reader.sha(ROOT / reader.PROTOCOL)}))
    if not (work / "cache" / (key + ".json")).exists():
        reason = budget_rejection(ledger, call_ids, reservation_usd(count_callback(req["input"]), req["max_output_tokens"]))
        if reason:
            return {"status": "budget_exhausted", "raw_text": "", "response": None, "usage": None,
                    "cost_usd": 0, "request": req, "request_sha256": reader.digest(reader.canonical(req)),
                    "cache_key": key, "budget_scope": reason}
    return reader.api_call(api, protocol, req, call_id)


def summarize(report):
    rows = report["rows"]
    require(len(rows) == 100 and len({row["qid"] for row in rows}) == 100, "accuracy requires all 100 dev questions")
    arms = [row["systems"][SYSTEM]["arms"][ARM] for row in rows]
    correct = sum(arm["scores"]["correct"] for arm in arms)
    terminal = all(arm["reader"]["status"] != "pending" and arm["judge"]["status"] != "pending" for arm in arms)
    counts = [arm["input_tokens"] for arm in arms]
    return {
        "n": 100, "correct": correct, "answer_accuracy": correct / 100,
        "answer_accuracy_ci95": wilson(correct, 100), "acceptance_correct": 100,
        "acceptance_met": correct == 100 and report["complete"] and terminal,
        "reader_status": dict(Counter(arm["reader"]["status"] for arm in arms)),
        "judge_status": dict(Counter(arm["judge"]["status"] for arm in arms)),
        "by_type": {kind: {"n": sum(row["qtype"] == kind for row in rows),
                           "correct": sum(row["qtype"] == kind and row["systems"][SYSTEM]["arms"][ARM]["scores"]["correct"]
                                          for row in rows)} for kind in sorted({row["qtype"] for row in rows})},
        "input_tokens": {"median": statistics.median(counts), "max": max(counts)},
        "user_occurrences": sum(source["role"] == "user" for arm in arms for source in arm["retained_sources"]),
        "assistant_occurrences": sum(source["role"] == "assistant" for arm in arms for source in arm["retained_sources"]),
        "coverage": {"scored": sum(row["scored_evidence"] for row in rows),
                     "actual_complete": sum(row["scored_evidence"] and row["actual_complete"] for row in rows)},
        "abstentions": sum(arm["scores"]["abstain"] for arm in arms),
        "exact_quote_claims": sum(arm["scores"]["citation_valid_claims"] for arm in arms),
        "total_claims": sum(arm["scores"]["citation_total_claims"] for arm in arms),
    }


def save(report, work):
    report["summary"] = summarize(report)
    ledger = json.loads((work / "spend.json").read_text())
    report["spend_ledger_sha256"] = reader.sha(work / "spend.json")
    report["incremental_spend_usd"] = variant_spend(ledger, experiment_call_ids(work, report["rows"]))
    report["global_recorded_spend_usd"] = ledger["total_usd"]
    reader.write(work / "answers.json", report)


def input_manifest(rows):
    return [{"qid": row["qid"], "reader_input_sha256": row["systems"][SYSTEM]["arms"][ARM]["reader_input_sha256"],
             "input_tokens": row["systems"][SYSTEM]["arms"][ARM]["input_tokens"]} for row in rows]


def prepare(work):
    require(not (work / "answers.json").exists(), "prepared experiment exists; resume it instead of overwriting")
    baseline = json.loads(user_first.BASELINE.read_text())
    require(baseline["dev_only"] is True and baseline["heldout_run"] is False and baseline["complete"] is True,
            "baseline is not complete dev-only evidence")
    dev = set(json.loads(user_first.SPLIT.read_text())["dev"])
    baseline_rows = user_first.membership(baseline["rows"], dev, "baseline")
    v1_path = RESULTS / "longmemeval_dev_answers_v1.json"
    v1_rows = user_first.membership(json.loads(v1_path.read_text())["rows"], dev, "v1 snapshots")
    historical = json.loads(user_first.HISTORICAL.read_text())
    history = user_first.membership(historical["rows"], dev, "historical identities")
    manifest_path = user_first.SOURCE_WORK / "manifest.json"
    require(reader.sha(manifest_path) == historical["source_manifest_sha256"], "dev source manifest mismatch")
    sources = load_dev_sources(user_first.SOURCE_WORK, user_first.SPLIT)
    base = user_first.load_protocol()
    user_first.reference_answers(baseline, dev)
    protocol = make_protocol(base)
    tokenizer = tiktoken.get_encoding("o200k_base")
    rows = []
    for qid in sorted(dev):
        old = baseline_rows[qid]
        question = {key: old[key] for key in ("qid", "qtype", "question", "question_date")}
        candidates = list(dict.fromkeys(tid for session in sources[qid] for tid, _ in session["turns"]))
        groups = user_first.source_groups(qid, sources[qid], candidates, [v1_rows[qid], old])
        arm = prepare_arm(question, groups, protocol, lambda text: len(tokenizer.encode(text)))
        expected = sum(len(session["turns"]) for session in sources[qid])
        require(sum(source["role"] == "user" for source in arm["retained_sources"]) == expected,
                "full history omitted a user occurrence")
        gold = {reader.digest(tid) for tid in history[qid]["ids"] if "answer" in tid}
        retained = {group["frozen_selected_id_sha256"] for group in groups}
        rows.append({**question, "systems": {SYSTEM: {"arms": {ARM: arm}}}, "scored_evidence": history[qid]["scored"],
                     "actual_complete": not arm["reader_cap"]["selection_failure"] and gold <= retained,
                     "candidate_groups": len(groups), "known_assistant_groups": sum(
                         any(source["role"] == "assistant" for source in group["sources"]) for group in groups)})
        if len(rows) % 10 == 0:
            print(json.dumps({"prepared_dev_questions": len(rows)}), flush=True)
    user_first.membership(rows, dev, "prepared answers")
    require(sum(source["role"] == "user" for row in rows for source in row["systems"][SYSTEM]["arms"][ARM]["retained_sources"])
            == json.loads(manifest_path.read_text())["n_turns"], "dev history total differs from pinned manifest")
    protocol["prepared_inputs"] = input_manifest(rows)
    protocol["global_spend_caps_usd"] = json.loads((ROOT / reader.PREREGISTRATION).read_text())["spend_caps_usd"]
    work.mkdir(parents=True, exist_ok=True)
    reader.write(work / "protocol.json", protocol)
    ledger = refresh_ledger(work)
    artifacts = [Path(__file__), Path(user_first.__file__), Path(reader.__file__),
                 Path(__file__).with_name("longmemeval_answer_evidence.py"),
                 Path(__file__).with_name("longmemeval_dev_sources.py"),
                 Path(__file__).with_name("check_longmemeval_answers.py"),
                 user_first.BASELINE, v1_path, user_first.HISTORICAL, user_first.SPLIT, manifest_path,
                 ROOT / reader.PROTOCOL, ROOT / reader.PREREGISTRATION, user_first.EXPORT,
                 artifact_path(protocol["judge_source"]["path"])]
    report = {"schema_version": 1, "dev_only": True, "heldout_run": False, "reader_systems": [SYSTEM],
              "variant": work.name, "complete": False, "rows": rows,
              "protocol_path": str((work / "protocol.json").relative_to(ROOT)), "protocol_sha256": reader.sha(work / "protocol.json"),
              "starting_spend_usd": ledger["total_usd"],
              "artifacts": [{"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                             "sha256": reader.sha(path)} for path in artifacts]}
    save(report, work)
    print(json.dumps(report["summary"]), flush=True)


def run_stage(report, work, stage):
    require(stage in ("reader", "judge"), "invalid experiment stage")
    require(report["dev_only"] is True and report["heldout_run"] is False, "only dev experiments are supported")
    dev = set(json.loads(user_first.SPLIT.read_text())["dev"])
    user_first.membership(report["rows"], dev, "prepared answers")
    for artifact in report["artifacts"]:
        require(reader.sha(artifact_path(artifact["path"])) == artifact["sha256"], "frozen experiment artifact changed")
    require(reader.sha(work / "protocol.json") == report["protocol_sha256"], "experiment protocol changed")
    protocol = json.loads((work / "protocol.json").read_text())
    same(input_manifest(report["rows"]), protocol["prepared_inputs"], "frozen reader inputs changed")
    require(stage != "judge" or all(row["systems"][SYSTEM]["arms"][ARM]["reader"]["status"] != "pending"
                                     for row in report["rows"]), "finish all readers before judging")
    references = user_first.reference_answers(json.loads(user_first.BASELINE.read_text()), dev) if stage == "judge" else None
    tokenizer = tiktoken.get_encoding("o200k_base")
    count = lambda text: len(tokenizer.encode(text))
    tasks = [row for row in report["rows"] if row["systems"][SYSTEM]["arms"][ARM][stage]["status"] == "pending"]
    api = reader.client() if tasks else None
    call_ids = experiment_call_ids(work, report["rows"])
    previous_ledger, previous_cache = reader.LEDGER, reader.CACHE
    reader.LEDGER, reader.CACHE = work / "spend.json", work / "cache"
    try:
        for done, row in enumerate(tasks, 1):
            arm = row["systems"][SYSTEM]["arms"][ARM]
            if stage == "judge" and arm["reader"]["status"] != "ok":
                arm["judge"] = {"status": "reader_failure", "raw_text": "", "score": False}
            else:
                if stage == "reader":
                    prompt = serialize(row, arm["source_groups"], protocol)
                    require(prompt == arm["reader_input"] and reader.digest(prompt) == arm["reader_input_sha256"],
                            "prepared reader input changed")
                    require(count(prompt) == arm["input_tokens"] <= protocol["local_token_cap"], "reader input exceeds local cap")
                else:
                    prompt = reader.judge_prompt(protocol, row, arm["reader"]["parsed"]["answer"], references[row["qid"]])
                call_id = reader.digest(f"full-history:{work.name}:{stage}:{row['qid']}")
                output = guarded_api_call(api, protocol, reader.request(protocol, prompt, stage), call_id, work, call_ids, count)
                if stage == "reader":
                    parsed = reader.parse_answer(output["raw_text"]) if output["status"] == "ok" else None
                    if output.get("usage"):
                        require(output["usage"]["input_tokens"] <= protocol["token_cap"], "actual reader input exceeded token cap")
                    arm["reader"] = {"status": "ok" if parsed is not None else "error", "raw_text": output["raw_text"],
                                     "parsed": parsed, "call": output}
                else:
                    text = output["raw_text"].strip().lower()
                    arm["judge"] = {"status": "ok" if output["status"] == "ok" and text in ("yes", "no") else "error",
                                    "raw_text": output["raw_text"], "score": output["status"] == "ok" and text == "yes", "call": output}
            arm["scores"] = recompute_scores(arm, allow_incomplete=True)
            save(report, work)
            print(json.dumps({"stage": stage, "done": done, "total": len(tasks), "status": arm[stage]["status"],
                              "correct": report["summary"]["correct"], "incremental_spend_usd": report["incremental_spend_usd"]}), flush=True)
    finally:
        reader.LEDGER, reader.CACHE = previous_ledger, previous_cache
    report["complete"] = all(arm["reader"]["status"] != "pending" and arm["judge"]["status"] != "pending"
                             for row in report["rows"] for arm in row["systems"][SYSTEM]["arms"].values())
    save(report, work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "reader", "judge", "run"))
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, "run this experiment from the repository root")
    work = args.work.resolve()
    require(work.is_relative_to(RESULTS), "experiment must live in benchmark results")
    if args.stage == "prepare":
        prepare(work)
    else:
        report = json.loads((work / "answers.json").read_text())
        for stage in (("reader", "judge") if args.stage == "run" else (args.stage,)):
            run_stage(report, work, stage)
        print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
