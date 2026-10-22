# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0", "tokenizers==0.23.3", "openai==3.27.0", "python-dotenv==1.2.4"]
# ///
"""Freeze complete original dev conversations for a separate high-reasoning reader experiment."""

import argparse
import copy
import json
from pathlib import Path

import tiktoken

import longmemeval_full_history as experiment
import longmemeval_reader_api as reader
import longmemeval_role_reader as role
import longmemeval_user_first as user_first
from check_longmemeval_answers import artifact_path, recompute_scores, require, same
from longmemeval_answer_evidence import reader_payload
from longmemeval_dev_conversations import load_dev_conversations

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK = ROOT / "benchmarks/results/longmemeval_complete_history_v1"
SYSTEM = experiment.SYSTEM
ARM = experiment.ARM
CONTEXT_CAP = 196608
FRAMING_RESERVE = 512
NEW_SPEND_CAP_USD = 50.0
SOURCE_POLICY = "all complete original dev user and assistant turns including unpaired assistant blocks; no retrieval, filtering, or truncation"
COMPLETE_REASONING_RULES = (
    "\nBefore the final answer, reason internally with source-linked bookkeeping for each relevant entity, "
    "relation, property or value, domain and scope, event or effective time, and observation time. Align references "
    "and implicit subjects with their source premises before combining facts across turns or sessions. An explicit "
    "later update to the same entity and property supersedes an older state only after checking its scope and "
    "effective time; observation recency alone is insufficient. For a current duration, align the duration with "
    "the correct referent and the observation date of its source, and check applicable explicit later updates. "
    "The supplied retained sources are eligible memory: question_date is observation context, not an implicit "
    "cutoff that excludes a later-recorded source. Apply an explicitly requested window to the event or relation "
    "being asked about, and distinguish that event time from the time it was reported. "
    "List or count only distinct entities or events meeting the requested domain, relation, and date window. "
    "Deduplicate an entity or event rather than counting its repeated mentions, and do not transfer facts between "
    "different scopes. Align the requested relation and event type before converting units or summing durations; "
    "a state tenure and an event duration are distinct. Avoid counting shared time twice when intervals overlap. "
    "Keep calendar months in their stated units unless exact dates or an explicit conversion convention support "
    "a conversion; do not invent month lengths or exact event dates. For a cost total, add the known priced items "
    "and explicitly distinguish priced from "
    "unpriced items; report supported totals or bounds with qualifiers instead of silently treating unknown "
    "prices as zero. Keep fee, payment or booking time distinct from the event time. Personalize using explicit "
    "preferences and relevant individual history. Distinguish the user's own assertions from quoted or pasted "
    "material supplied for rewriting; that material alone does not establish the user's personal interests. "
    "Keep personalized suggestions within the most specific supported combination of interests. "
    "Assistant recommendations or suggestions do not establish "
    "the user's ownership, adoption, or completed actions unless the user confirms them. Ground the user's "
    "state, actions, and preferences in user assertions; questions about what an assistant previously reported "
    "can use the original assistant text as assistant-reported evidence. A qualified direct answer is appropriate when its scope or "
    "bound is supported; abstain when the requested fact lacks support. Emit only the required strict JSON object. "
    "If abstain is true, claims MUST be an empty array. Preserve original source roles and verbatim citation "
    "spans; never change claims after generation to make a response pass.\n"
)


def make_protocol(base):
    protocol = experiment.make_protocol(base)
    settings = protocol["systems"][SYSTEM]
    settings.pop("reasoning_effort")
    settings.update(reader_reasoning_effort="high", judge_reasoning_effort="medium",
                    reader_max_output_tokens=8192, judge_max_output_tokens=2048,
                    framing_verification="complete local prompt <=196096; actual API input <=196608")
    protocol.update(reader_prompt=protocol["reader_prompt"] + COMPLETE_REASONING_RULES,
                    additional_reasoning_rules=COMPLETE_REASONING_RULES,
                    token_cap=CONTEXT_CAP, local_token_cap=CONTEXT_CAP - FRAMING_RESERVE,
                    framing_reserve_tokens=FRAMING_RESERVE, new_spend_cap_usd=NEW_SPEND_CAP_USD,
                    source_policy=SOURCE_POLICY,
                    source_gate={"metadata_ids_verified": 500, "dev_records_decoded": 100, "heldout_records_decoded": 0},
                    comparison_budget="complete original history and greater reader reasoning; not an equal-budget comparison",
                    limitations=["Repeated-use dev only; no held-out claim.",
                                 "Larger context and greater reader reasoning than prior variants; not an equal-budget comparison.",
                                 "All original user and assistant occurrences are included; evidence labels do not select sources.",
                                 "Input overflow is a selection failure without source truncation."])
    return protocol


def request(protocol, prompt, stage):
    require(stage in ("reader", "judge"), "invalid experiment stage")
    settings = protocol["systems"][SYSTEM]
    return {"model": settings[f"{stage}_model"], "input": prompt,
            "reasoning": {"effort": settings[f"{stage}_reasoning_effort"]},
            "max_output_tokens": settings[f"{stage}_max_output_tokens"], "store": False, "truncation": "disabled"}


def serialize(question, groups, protocol):
    return protocol["reader_prompt"] + reader.canonical(reader_payload(question, {"source_groups": groups}))


def prepare_arm(question, groups, protocol, count_callback):
    require(type(protocol["token_cap"]) is int and protocol["token_cap"] > 0, "invalid context cap")
    reserve = protocol["framing_reserve_tokens"]
    require(type(reserve) is int and 0 <= reserve < protocol["token_cap"], "invalid framing reserve")
    require(protocol["local_token_cap"] == protocol["token_cap"] - reserve, "local input cap mismatch")
    require(isinstance(groups, list) and groups, "complete history requires source groups")
    copied = copy.deepcopy(groups)
    for group in copied:
        require(isinstance(group, dict) and isinstance(group.get("sources"), list) and group["sources"],
                "invalid complete history group")
        require(all(isinstance(source, dict) and source.get("role") in ("user", "assistant")
                    and isinstance(source.get("text"), str) for source in group["sources"]), "invalid complete history source")
    prompt = serialize(question, copied, protocol)
    tokens = count_callback(prompt)
    require(type(tokens) is int and tokens >= 0, "tokenizer returned invalid count")
    failed = tokens > protocol["local_token_cap"]
    arm = {"source_groups": copied, "reader_input": prompt, "reader_input_sha256": reader.digest(prompt),
           "retained_sources": reader.retained(copied), "input_tokens": tokens,
           "reader_cap": {"cap": protocol["token_cap"], "local_cap": protocol["local_token_cap"],
                          "input_tokens": tokens, "uncapped_input_tokens": tokens, "dropped_group_ids": [],
                          "selection_failure": failed,
                          "failure_reason": "complete_history_input_exceeds_cap" if failed else None,
                          "source_policy": "complete_original_history"},
           "reader": {"status": "selection_failure" if failed else "pending", "raw_text": "", "parsed": None},
           "judge": {"status": "pending", "raw_text": "", "score": False}}
    arm["scores"] = recompute_scores(arm, allow_incomplete=True)
    return arm


def guarded_api_call(api, protocol, req, call_id, work, call_ids, count_callback):
    require(protocol["new_spend_cap_usd"] == NEW_SPEND_CAP_USD, "new experiment spend cap changed")
    ledger = experiment.refresh_ledger(work)
    key = reader.digest(reader.canonical({"request": req, "call_id": call_id,
                                          "protocol_sha256": reader.sha(ROOT / reader.PROTOCOL)}))
    if not (work / "cache" / (key + ".json")).exists():
        reserve = experiment.reservation_usd(count_callback(req["input"]), req["max_output_tokens"])
        reason = experiment.budget_rejection(ledger, call_ids, reserve, cap=NEW_SPEND_CAP_USD)
        if reason:
            return {"status": "budget_exhausted", "raw_text": "", "response": None, "usage": None,
                    "cost_usd": 0, "request": req, "request_sha256": reader.digest(reader.canonical(req)),
                    "cache_key": key, "budget_scope": reason}
    return reader.api_call(api, protocol, req, call_id)


def validate_inputs(rows, dev, protocol):
    user_first.membership(rows, dev, "prepared answers")
    same(experiment.input_manifest(rows), protocol["prepared_inputs"], "frozen reader inputs changed")
    for row in rows:
        arm = row["systems"][SYSTEM]["arms"][ARM]
        prompt = serialize(row, arm["source_groups"], protocol)
        require(prompt == arm["reader_input"] and reader.digest(prompt) == arm["reader_input_sha256"],
                "prepared complete history changed")


def prepare(work):
    require(not (work / "answers.json").exists(), "prepared experiment exists; resume it instead of overwriting")
    split = json.loads(user_first.SPLIT.read_text())
    dev = set(split["dev"])
    conversations = load_dev_conversations(role.SOURCE, user_first.SPLIT)
    baseline = json.loads(user_first.BASELINE.read_text())
    require(baseline["dev_only"] is True and baseline["heldout_run"] is False and baseline["complete"] is True,
            "baseline is not complete dev-only evidence")
    previous = user_first.membership(baseline["rows"], dev, "baseline")
    history = user_first.membership(json.loads(user_first.HISTORICAL.read_text())["rows"], dev, "historical identities")
    protocol = make_protocol(user_first.load_protocol())
    tokenizer = tiktoken.get_encoding("o200k_base")
    rows = []
    for qid in sorted(dev):
        raw, old = conversations[qid], previous[qid]
        require(raw["question"] == old["question"] and raw["question_date"] == old["question_date"]
                and raw["question_type"] == old["qtype"], "dev question differs from frozen baseline")
        question = {field: old[field] for field in ("qid", "qtype", "question", "question_date")}
        groups, _, orphans = role.groups_for_question(raw, set(history[qid]["ids"]))
        arm = prepare_arm(question, groups, protocol, lambda text: len(tokenizer.encode(text)))
        gold = {reader.digest(alias) for alias in history[qid]["ids"] if "answer" in alias}
        kept = {group["frozen_selected_id_sha256"] for group in groups}
        rows.append({**question, "systems": {SYSTEM: {"arms": {ARM: arm}}}, "scored_evidence": history[qid]["scored"],
                     "actual_complete": not arm["reader_cap"]["selection_failure"] and gold <= kept,
                     "candidate_groups": len(groups), "conversation_groups": len(groups), "orphan_assistant_groups": orphans})
        if len(rows) % 10 == 0:
            print(json.dumps({"prepared_dev_questions": len(rows)}), flush=True)
    user_first.membership(rows, dev, "prepared answers")
    protocol["prepared_inputs"] = experiment.input_manifest(rows)
    protocol["global_spend_caps_usd"] = json.loads((ROOT / reader.PREREGISTRATION).read_text())["spend_caps_usd"]
    validate_inputs(rows, dev, protocol)
    work.mkdir(parents=True, exist_ok=True)
    reader.write(work / "protocol.json", protocol)
    ledger = experiment.refresh_ledger(work)
    files = [Path(__file__), Path(experiment.__file__), Path(role.__file__), Path(user_first.__file__), Path(reader.__file__),
             Path(__file__).with_name("longmemeval_role_retrieval.py"),
             Path(__file__).with_name("longmemeval_dev_conversations.py"),
             Path(__file__).with_name("longmemeval_answer_evidence.py"),
             Path(__file__).with_name("check_longmemeval_answers.py"),
             Path(__file__).parent / "proxies/beir_proxies.py", role.SOURCE,
             user_first.BASELINE, user_first.HISTORICAL, user_first.SPLIT, user_first.EXPORT,
             ROOT / reader.PROTOCOL, ROOT / reader.PREREGISTRATION, artifact_path(protocol["judge_source"]["path"])]
    report = {"schema_version": 1, "dev_only": True, "heldout_run": False, "reader_systems": [SYSTEM],
              "variant": work.name, "complete": False, "rows": rows,
              "protocol_path": str((work / "protocol.json").relative_to(ROOT)), "protocol_sha256": reader.sha(work / "protocol.json"),
              "starting_spend_usd": ledger["total_usd"],
              "artifacts": [{"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                             "sha256": reader.sha(path)} for path in files]}
    experiment.save(report, work)
    print(json.dumps(report["summary"]), flush=True)


def run_stage(report, work, stage):
    require(stage in ("reader", "judge"), "invalid experiment stage")
    require(report["dev_only"] is True and report["heldout_run"] is False, "only dev experiments are supported")
    for artifact in report["artifacts"]:
        require(reader.sha(artifact_path(artifact["path"])) == artifact["sha256"], "frozen experiment artifact changed")
    require(reader.sha(work / "protocol.json") == report["protocol_sha256"], "experiment protocol changed")
    protocol = json.loads((work / "protocol.json").read_text())
    require(protocol["token_cap"] == CONTEXT_CAP and protocol["local_token_cap"] == CONTEXT_CAP - FRAMING_RESERVE
            and protocol["new_spend_cap_usd"] == NEW_SPEND_CAP_USD and protocol["source_policy"] == SOURCE_POLICY,
            "complete history protocol changed")
    dev = set(json.loads(user_first.SPLIT.read_text())["dev"])
    validate_inputs(report["rows"], dev, protocol)
    require(stage != "judge" or all(row["systems"][SYSTEM]["arms"][ARM]["reader"]["status"] != "pending"
                                     for row in report["rows"]), "finish all readers before judging")
    references = user_first.reference_answers(json.loads(user_first.BASELINE.read_text()), dev) if stage == "judge" else None
    tokenizer = tiktoken.get_encoding("o200k_base")
    count = lambda text: len(tokenizer.encode(text))
    tasks = [row for row in report["rows"] if row["systems"][SYSTEM]["arms"][ARM][stage]["status"] == "pending"]
    api = reader.client() if tasks else None
    call_ids = experiment.experiment_call_ids(work, report["rows"])
    previous_ledger, previous_cache = reader.LEDGER, reader.CACHE
    reader.LEDGER, reader.CACHE = work / "spend.json", work / "cache"
    try:
        for done, row in enumerate(tasks, 1):
            arm = row["systems"][SYSTEM]["arms"][ARM]
            if stage == "judge" and arm["reader"]["status"] != "ok":
                arm["judge"] = {"status": "reader_failure", "raw_text": "", "score": False}
            else:
                if stage == "reader":
                    prompt = arm["reader_input"]
                    require(count(prompt) == arm["input_tokens"] <= protocol["local_token_cap"], "reader input exceeds local cap")
                else:
                    prompt = reader.judge_prompt(protocol, row, arm["reader"]["parsed"]["answer"], references[row["qid"]])
                call_id = reader.digest(f"full-history:{work.name}:{stage}:{row['qid']}")
                output = guarded_api_call(api, protocol, request(protocol, prompt, stage), call_id, work, call_ids, count)
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
            experiment.save(report, work)
            print(json.dumps({"stage": stage, "done": done, "total": len(tasks), "status": arm[stage]["status"],
                              "correct": report["summary"]["correct"], "incremental_spend_usd": report["incremental_spend_usd"]}), flush=True)
    finally:
        reader.LEDGER, reader.CACHE = previous_ledger, previous_cache
    report["complete"] = all(arm["reader"]["status"] != "pending" and arm["judge"]["status"] != "pending"
                             for row in report["rows"] for arm in row["systems"][SYSTEM]["arms"].values())
    experiment.save(report, work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "reader", "judge", "run"))
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, "run this experiment from the repository root")
    work = args.work.resolve()
    require(work.is_relative_to(experiment.RESULTS), "experiment must live in benchmark results")
    if args.stage == "prepare":
        prepare(work)
    else:
        report = json.loads((work / "answers.json").read_text())
        for stage in (("reader", "judge") if args.stage == "run" else (args.stage,)):
            run_stage(report, work, stage)
        print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
