# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13"
# dependencies = ["tiktoken==0.14.0", "tokenizers==0.23.3", "openai==3.27.0", "python-dotenv==1.2.4"]
# ///
"""Run a frozen dev-only reader with user evidence admitted before assistant continuations."""

import argparse
import copy
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tiktoken

import longmemeval_reader_api as reader
from check_longmemeval_answers import artifact_path, check_protocol, recompute_scores, require, same, wilson
from longmemeval_answer_evidence import reader_payload
from longmemeval_dev_sources import load_dev_sources
from longmemeval_reader_budget import cap_sources_user_first

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "benchmarks/results/longmemeval_dev_answers_v3.json"
HISTORICAL = ROOT / "benchmarks/results/evidence_completeness_dev.json"
SPLIT = ROOT / "benchmarks/public/longmemeval_split.json"
DEFAULT_WORK = ROOT / "benchmarks/results/longmemeval_user_first_v1"
SOURCE_WORK = ROOT / "target/conversation-geometry-validation/models"
EXPORT = Path.home() / ".cache/warm-bench/holo_dev/export_dev.json"
ARM = "user_first"
SYSTEM = "matched_gpt54"


def load_protocol():
    protocol = json.loads((ROOT / reader.PROTOCOL).read_text())
    preregistration = json.loads((ROOT / reader.PREREGISTRATION).read_text())
    check_protocol(protocol, preregistration)
    return protocol


def membership(rows, dev, label):
    require(len(rows) == 100 and {row["qid"] for row in rows} == dev,
            f"{label} must cover exactly the frozen 100 dev questions")
    return {row["qid"]: row for row in rows}


def source_groups(qid, sessions, candidates, snapshots):
    available = {}
    for snapshot in snapshots:
        for arm in snapshot["systems"][SYSTEM]["arms"].values():
            for group in arm["source_groups"]:
                key = group["frozen_selected_id_sha256"]
                if key in available:
                    same(available[key], group["sources"], "snapshot source disagreement")
                available[key] = group["sources"]
    observations = {}
    for session_index, session in enumerate(sessions):
        for tid, text in session["turns"]:
            turn_index = int(tid.rsplit("_", 1)[1]) - 1
            source = {"source_id": reader.digest(reader.canonical(["lme-dev-archive-source-v1", qid, session_index, turn_index])),
                      "session_id": reader.digest(reader.canonical(["lme-dev-archive-session-v1", qid, session_index])),
                      "session_index": session_index, "turn_index": turn_index,
                      "observation_time": session["date"], "role": "user", "text": text}
            observations.setdefault(tid, []).append(source)
    by_handle = {reader.digest(tid): users for tid, users in observations.items()}
    session_map = {}
    for key, known in available.items():
        users = by_handle[key]
        known_users = [source for source in known if source["role"] == "user"]
        require(len(users) == len(known_users), "snapshot user occurrence count differs from dev archive")
        for source, original in zip(users, known_users):
            same({field: source[field] for field in ("text", "observation_time", "turn_index", "role")},
                 {field: original[field] for field in ("text", "observation_time", "turn_index", "role")},
                 "snapshot user source differs from pinned dev archive")
            session = (original["session_index"], original["session_id"])
            previous = session_map.setdefault(source["session_index"], session)
            require(previous == session, "snapshot session mapping conflicts")
    for users in observations.values():
        for source in users:
            if source["session_index"] in session_map:
                original_index, session_id = session_map[source["session_index"]]
                source["session_index"] = original_index
                source["session_id"] = session_id
                source["source_id"] = reader.digest(reader.canonical([
                    "lme-source-v1", qid, original_index, source["turn_index"]]))
    groups = []
    for rank, tid in enumerate(candidates):
        users = observations[tid]
        key = reader.digest(tid)
        known = available.get(key)
        if known is not None:
            same([source for source in known if source["role"] == "user"], users,
                 "snapshot user source differs from pinned dev archive")
            sources = copy.deepcopy(known)
        else:
            sources = copy.deepcopy(users)
        group_id = reader.digest(reader.canonical([
            "lme-reader-group-v1", qid, [source["source_id"] for source in users]]))
        groups.append({"group_id": group_id, "selection_rank": rank,
                       "frozen_selected_id_sha256": key, "sources": sources})
    return groups


def reference_answers(baseline, dev):
    exported = json.loads(EXPORT.read_text())
    questions = membership(exported["questions"], dev, "reference export")
    protocol = load_protocol()
    for row in baseline["rows"]:
        reference = questions[row["qid"]]
        require(reference["question"] == row["question"] and reference["date"] == row["question_date"],
                "reference question mismatch")
        checked = 0
        for arm in row["systems"][SYSTEM]["arms"].values():
            if arm["judge"]["status"] == "ok":
                expected = reader.judge_prompt(protocol, row, arm["reader"]["parsed"]["answer"], reference["answer"])
                require(expected == arm["judge"]["call"]["request"]["input"], "reference differs from frozen judge input")
                checked += 1
        require(checked > 0, "reference has no frozen judge input to verify")
    return {qid: question["answer"] for qid, question in questions.items()}


def seed_ledger(work):
    destination = work / "spend.json"
    if destination.exists():
        return
    entries = {}
    paths = [ROOT / "benchmarks/results/spend_2026-10.json",
             *sorted((ROOT / "benchmarks/results").glob("longmemeval_dev_answers*_spend.json"))]
    for path in paths:
        ledger = json.loads(path.read_text())
        for entry in ledger["entries"]:
            key = entry["cache_key"]
            if key not in entries or entry["cost_usd"] > entries[key]["cost_usd"]:
                entries[key] = copy.deepcopy(entry)
    caps = json.loads(paths[0].read_text())["caps_usd"]
    categories = {category: sum(entry["cost_usd"] for entry in entries.values() if entry["category"] == category)
                  for category in caps if category != "total"}
    reader.write(destination, {"schema_version": 1, "caps_usd": caps, "entries": list(entries.values()),
                               "total_usd": sum(entry["cost_usd"] for entry in entries.values()),
                               "by_category_usd": categories,
                               "seed_paths": [{"path": str(path.relative_to(ROOT)), "sha256": reader.sha(path)} for path in paths]})


def serialize(question, groups, protocol):
    return protocol["reader_prompt"] + reader.canonical(reader_payload(question, {"source_groups": groups}))


def summarize(report):
    rows = report["rows"]
    values = [row["systems"][SYSTEM]["arms"][ARM] for row in rows]
    correct = sum(arm["scores"]["correct"] for arm in values)
    by_type = {}
    for kind in sorted({row["qtype"] for row in rows}):
        selected = [row for row in rows if row["qtype"] == kind]
        by_type[kind] = {"n": len(selected), "correct": sum(row["systems"][SYSTEM]["arms"][ARM]["scores"]["correct"]
                                                           for row in selected)}
    return {"n": 100, "correct": correct, "answer_accuracy": correct / 100,
            "answer_accuracy_ci95": wilson(correct, 100), "acceptance_correct": 100,
            "acceptance_met": correct == 100 and report["complete"], "by_type": by_type,
            "reader_status": dict(Counter(arm["reader"]["status"] for arm in values)),
            "judge_status": dict(Counter(arm["judge"]["status"] for arm in values)),
            "coverage": {"scored": sum(row["scored_evidence"] for row in rows),
                         "baseline_actual_complete": sum(row["scored_evidence"] and row["baseline_actual_complete"] for row in rows),
                         "actual_complete": sum(row["scored_evidence"] and row["actual_complete"] for row in rows)},
            "abstentions": sum(arm["scores"]["abstain"] for arm in values),
            "exact_quote_claims": sum(arm["scores"]["citation_valid_claims"] for arm in values),
            "total_claims": sum(arm["scores"]["citation_total_claims"] for arm in values)}


def save(report, work):
    report["summary"] = summarize(report)
    ledger = work / "spend.json"
    if ledger.exists():
        report["spend_ledger_sha256"] = reader.sha(ledger)
        report["incremental_spend_usd"] = json.loads(ledger.read_text())["total_usd"] - report["starting_spend_usd"]
    reader.write(work / "answers.json", report)


def prepare(work, mode):
    require(not (work / "answers.json").exists(), "prepared experiment exists; resume it instead of overwriting")
    baseline = json.loads(BASELINE.read_text())
    require(baseline["dev_only"] is True and baseline["heldout_run"] is False and baseline["complete"] is True,
            "baseline is not complete dev-only evidence")
    split = json.loads(SPLIT.read_text())
    dev = set(split["dev"])
    baseline_rows = membership(baseline["rows"], dev, "baseline")
    evidence_path = ROOT / baseline["evidence_report_path"]
    require(reader.sha(evidence_path) == baseline["evidence_report_sha256"], "frozen evidence fingerprint mismatch")
    evidence_rows = membership(json.loads(evidence_path.read_text())["rows"], dev, "evidence")
    historical = json.loads(HISTORICAL.read_text())
    history = membership(historical["rows"], dev, "historical ranking")
    require(reader.sha(SOURCE_WORK / "manifest.json") == historical["source_manifest_sha256"], "dev source manifest mismatch")
    sources = load_dev_sources(SOURCE_WORK, SPLIT)
    v1_path = ROOT / "benchmarks/results/longmemeval_dev_answers_v1.json"
    v1_rows = membership(json.loads(v1_path.read_text())["rows"], dev, "v1 snapshots")
    protocol = load_protocol()
    reference_answers(baseline, dev)
    tokenizer = tiktoken.get_encoding("o200k_base")
    rows = []
    for qid in sorted(dev):
        old = baseline_rows[qid]
        candidates = evidence_rows[qid]["arms"]["operand"]["selected"] if mode == "selected" else evidence_rows[qid]["candidates"]
        require(1 <= len(candidates) <= 64 and len(set(candidates)) == len(candidates), "invalid candidate pool")
        question = {key: old[key] for key in ("qid", "qtype", "question", "question_date")}
        groups = source_groups(qid, sources[qid], candidates, [v1_rows[qid], old])
        original = {"source_groups": groups}
        arm = cap_sources_user_first(original, lambda selected: serialize(question, selected, protocol),
                                     lambda text: len(tokenizer.encode(text)), protocol["token_cap"] - 512)
        failed = arm["reader_cap"]["selection_failure"]
        prompt = "" if failed else serialize(question, arm["source_groups"], protocol)
        arm.update(reader_input=prompt, reader_input_sha256=reader.digest(prompt),
                   retained_sources=reader.retained(arm["source_groups"]), input_tokens=len(tokenizer.encode(prompt)),
                   reader={"status": "selection_failure" if failed else "pending", "raw_text": "", "parsed": None},
                   judge={"status": "pending", "raw_text": "", "score": False})
        arm["scores"] = reader.scores(arm)
        gold = {reader.digest(tid) for tid in history[qid]["ids"] if "answer" in tid}
        retained = {group["frozen_selected_id_sha256"] for group in arm["source_groups"]}
        previous = {group["frozen_selected_id_sha256"] for group in old["systems"][SYSTEM]["arms"]["operand"]["source_groups"]}
        rows.append({**question, "systems": {SYSTEM: {"arms": {ARM: arm}}}, "scored_evidence": history[qid]["scored"],
                     "actual_complete": gold <= retained, "baseline_actual_complete": gold <= previous,
                     "candidate_groups": len(groups), "known_assistant_groups": sum(any(source["role"] == "assistant" for source in group["sources"]) for group in groups)})
        if len(rows) % 10 == 0:
            print(json.dumps({"prepared_dev_questions": len(rows)}), flush=True)
    work.mkdir(parents=True, exist_ok=True)
    frozen = {"dev_only": True, "heldout_run": False, "acceptance_correct": 100,
              "source_policy": "all complete user observations first; complete known assistant turns admitted in remaining space",
              "candidate_mode": mode, "reader_prompt": protocol["reader_prompt"], "reader": protocol["systems"][SYSTEM],
              "token_cap": protocol["token_cap"], "local_token_cap": protocol["token_cap"] - 512,
              "automatic_transport_retries": 0, "reader_judge_concurrency": 2,
              "limitations": ["Repeated-use dev; no held-out claim.", "Assistant availability is limited to saved v1/v3 complete turns.",
                              "Labeled user support is measured after the reader cap; it does not establish semantic sufficiency."]}
    reader.write(work / "protocol.json", frozen)
    seed_ledger(work)
    report = {"schema_version": 1, "dev_only": True, "heldout_run": False, "reader_systems": [SYSTEM],
              "variant": work.name, "candidate_mode": mode, "complete": False, "rows": rows,
              "protocol_path": str((work / "protocol.json").relative_to(ROOT)), "protocol_sha256": reader.sha(work / "protocol.json"),
              "starting_spend_usd": json.loads((work / "spend.json").read_text())["total_usd"],
              "artifacts": [{"path": str(path.relative_to(ROOT)), "sha256": reader.sha(path)} for path in
                            (Path(__file__), Path(__file__).with_name("longmemeval_reader_budget.py"),
                             Path(__file__).with_name("longmemeval_dev_sources.py"), BASELINE, v1_path, HISTORICAL, SPLIT, evidence_path,
                             SOURCE_WORK / "manifest.json", ROOT / reader.PROTOCOL, ROOT / reader.PREREGISTRATION)]}
    report["artifacts"].append({"path": "~/.cache/warm-bench/holo_dev/export_dev.json", "sha256": reader.sha(EXPORT)})
    save(report, work)
    print(json.dumps(report["summary"]), flush=True)


def run_stage(report, work, stage):
    require(stage != "judge" or all(row["systems"][SYSTEM]["arms"][ARM]["reader"]["status"] != "pending"
                                     for row in report["rows"]), "finish all readers before judging")
    dev = set(json.loads(SPLIT.read_text())["dev"])
    membership(report["rows"], dev, "prepared answers")
    for artifact in report["artifacts"]:
        require(reader.sha(artifact_path(artifact["path"])) == artifact["sha256"], "frozen experiment artifact changed")
    require(reader.sha(work / "protocol.json") == report["protocol_sha256"], "experiment protocol changed")
    protocol = load_protocol()
    references = reference_answers(json.loads(BASELINE.read_text()), dev) if stage == "judge" else None
    reader.LEDGER = work / "spend.json"
    reader.CACHE = work / "cache"
    api = reader.client()
    tasks = [row for row in report["rows"] if row["systems"][SYSTEM]["arms"][ARM][stage]["status"] == "pending"]

    def call(row):
        arm = row["systems"][SYSTEM]["arms"][ARM]
        call_id = reader.digest(f"{work.name}:{stage}:{row['qid']}")
        if stage == "reader":
            if arm["reader"]["status"] == "selection_failure":
                return {"status": "selection_failure", "raw_text": "", "parsed": None}
            prompt = arm["reader_input"]
            require(reader.digest(prompt) == arm["reader_input_sha256"], "prepared reader input changed")
        else:
            if arm["reader"]["status"] != "ok":
                return {"status": "reader_failure", "raw_text": "", "score": False}
            prompt = reader.judge_prompt(protocol, row, arm["reader"]["parsed"]["answer"], references[row["qid"]])
        output = reader.api_call(api, protocol, reader.request(protocol, prompt, stage), call_id)
        if stage == "reader":
            parsed = reader.parse_answer(output["raw_text"]) if output["status"] == "ok" else None
            if output.get("usage"):
                require(output["usage"]["input_tokens"] <= protocol["token_cap"], "actual reader input exceeded token cap")
            return {"status": "ok" if parsed is not None else "error", "raw_text": output["raw_text"], "parsed": parsed, "call": output}
        text = output["raw_text"].strip().lower()
        return {"status": "ok" if output["status"] == "ok" and text in ("yes", "no") else "error",
                "raw_text": output["raw_text"], "score": output["status"] == "ok" and text == "yes", "call": output}

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(call, row): row for row in tasks}
        for done, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            arm = row["systems"][SYSTEM]["arms"][ARM]
            arm[stage] = future.result()
            arm["scores"] = recompute_scores(arm, allow_incomplete=True)
            save(report, work)
            print(json.dumps({"stage": stage, "done": done, "total": len(tasks), "status": arm[stage]["status"],
                              "correct": report["summary"]["correct"], "incremental_spend_usd": round(report["incremental_spend_usd"], 5)}), flush=True)
    report["complete"] = all(arm["reader"]["status"] != "pending" and arm["judge"]["status"] != "pending"
                             for row in report["rows"] for arm in row["systems"][SYSTEM]["arms"].values())
    save(report, work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "reader", "judge", "run"))
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--candidate-mode", choices=("selected", "pool"), default="pool")
    args = parser.parse_args()
    work = args.work.resolve()
    require(work.is_relative_to(ROOT / "benchmarks/results"), "experiment must live in benchmark results")
    if args.stage == "prepare":
        prepare(work, args.candidate_mode)
    else:
        report = json.loads((work / "answers.json").read_text())
        for stage in (("reader", "judge") if args.stage == "run" else (args.stage,)):
            run_stage(report, work, stage)
        print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
