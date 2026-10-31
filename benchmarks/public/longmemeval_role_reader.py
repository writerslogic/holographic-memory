# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0", "tokenizers==0.23.3", "openai==3.27.0", "python-dotenv==1.2.4"]
# ///
"""Retrieve complete user/assistant exchanges from dev conversations and run the pinned reader."""

import argparse
import copy
import json
from pathlib import Path

import tiktoken

import longmemeval_full_history as experiment
import longmemeval_reader_api as reader
import longmemeval_user_first as user_first
from check_longmemeval_answers import artifact_path, recompute_scores, require
from longmemeval_answer_evidence import _groups_for_question, cap_sources
from longmemeval_dev_conversations import load_dev_conversations
from longmemeval_role_retrieval import retrieve

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK = ROOT / "benchmarks/results/longmemeval_role_retrieval_v1"
SOURCE = Path.home() / ".cache/warm-bench/downloads/longmemeval_s_cleaned.json"


def groups_for_question(question, historical_ids):
    _, exchanges = _groups_for_question(question, historical_ids)
    groups, by_alias = [], {}
    for tid, sources in exchanges.items():
        key = reader.digest(tid)
        users = [source for source in sources if source["role"] == "user"]
        by_alias[tid] = len(groups)
        groups.append({"group_id": reader.digest(reader.canonical([
            "lme-group-v1", question["question_id"], [[source["session_index"], source["turn_index"]] for source in users]])),
            "frozen_selected_id_sha256": key, "sources": sources})
    covered = {source["source_id"] for group in groups for source in group["sources"]}
    orphan_count = 0
    for session_index, turns in enumerate(question["haystack_sessions"]):
        pending = []
        for turn_index, turn in enumerate(turns):
            source_id = reader.digest(reader.canonical(["lme-source-v1", question["question_id"], session_index, turn_index]))
            if turn["role"] != "assistant" or source_id in covered:
                continue
            pending.append({"source_id": source_id,
                            "session_id": reader.digest(reader.canonical(["lme-session-v1", question["question_id"], session_index])),
                            "session_index": session_index, "turn_index": turn_index,
                            "observation_time": question["haystack_dates"][session_index],
                            "role": "assistant", "text": turn["content"]})
        if pending:
            group_id = reader.digest(reader.canonical([
                "lme-assistant-group-v1", question["question_id"], [source["source_id"] for source in pending]]))
            groups.append({"group_id": group_id, "frozen_selected_id_sha256": group_id, "sources": pending})
            orphan_count += 1
    require(len({group["group_id"] for group in groups}) == len(groups), "duplicate conversation group")
    sources = [source for group in groups for source in group["sources"]]
    require(len({source["source_id"] for source in sources}) == len(sources), "duplicate conversation occurrence")
    require(len(sources) == sum(len(turns) for turns in question["haystack_sessions"]),
            "conversation index omitted an original turn")
    return groups, by_alias, orphan_count


def prepare_arm(question, selected, protocol, count):
    groups = copy.deepcopy(selected)
    require(groups and len(groups) <= 64, "invalid retrieved source count")
    for rank, group in enumerate(groups):
        group["selection_rank"] = rank
    capped = cap_sources({"source_groups": groups},
                         lambda retained: experiment.serialize(question, retained, protocol), count,
                         protocol["local_token_cap"])
    prompt = experiment.serialize(question, capped["source_groups"], protocol)
    tokens = count(prompt)
    failed = capped["reader_cap"]["selection_failure"]
    arm = {**capped, "reader_input": prompt, "reader_input_sha256": reader.digest(prompt),
           "retained_sources": reader.retained(capped["source_groups"]), "input_tokens": tokens,
           "reader": {"status": "selection_failure" if failed else "pending", "raw_text": "", "parsed": None},
           "judge": {"status": "pending", "raw_text": "", "score": False}}
    arm["scores"] = recompute_scores(arm, allow_incomplete=True)
    return arm


def prepare(work):
    require(not (work / "answers.json").exists(), "prepared experiment exists; resume it instead of overwriting")
    split = json.loads(user_first.SPLIT.read_text())
    dev = set(split["dev"])
    conversations = load_dev_conversations(SOURCE, user_first.SPLIT)
    baseline = json.loads(user_first.BASELINE.read_text())
    previous = user_first.membership(baseline["rows"], dev, "baseline")
    history = user_first.membership(json.loads(user_first.HISTORICAL.read_text())["rows"], dev, "historical ranking")
    base = user_first.load_protocol()
    user_first.reference_answers(baseline, dev)
    protocol = experiment.make_protocol(base)
    protocol["source_policy"] = "cached user hybrid ranking fused with assistant-only lexical ranking; complete original exchanges"
    protocol["retrieval"] = {"rrf_k0": 60, "user_prior_weight": 1, "assistant_weight": 1,
                             "reserved_user_prior": 5, "candidate_limit": 64,
                             "budget_policy": "drop lowest ranked whole exchanges; preserve the first anchor"}
    protocol["source_gate"] = {"metadata_ids_verified": 500, "dev_records_decoded": 100, "heldout_records_decoded": 0}
    protocol["limitations"] = ["Repeated-use dev only; no held-out claim.",
                               "Larger input budget than v3; not an equal-budget comparison.",
                               "User evidence labels measure full-turn retention rather than semantic sufficiency."]
    tokenizer = tiktoken.get_encoding("o200k_base")
    rows = []
    for qid in sorted(dev):
        raw = conversations[qid]
        old = previous[qid]
        require(raw["question"] == old["question"] and raw["question_date"] == old["question_date"],
                "dev question differs from frozen baseline")
        question = {field: old[field] for field in ("qid", "qtype", "question", "question_date")}
        pool, by_alias, orphans = groups_for_question(raw, set(history[qid]["ids"]))
        prior = [by_alias[tid] for tid in dict.fromkeys(history[qid]["baseline_ranking"])]
        result = retrieve(question["question"], pool, prior, limit=64)
        arm = prepare_arm(question, result["source_groups"], protocol, lambda text: len(tokenizer.encode(text)))
        gold = {reader.digest(tid) for tid in history[qid]["ids"] if "answer" in tid}
        kept = {group["frozen_selected_id_sha256"] for group in arm["source_groups"]}
        rows.append({**question, "systems": {experiment.SYSTEM: {"arms": {experiment.ARM: arm}}},
                     "scored_evidence": history[qid]["scored"],
                     "actual_complete": not arm["reader_cap"]["selection_failure"] and gold <= kept,
                     "candidate_groups": len(result["selected"]), "conversation_groups": len(pool),
                     "orphan_assistant_groups": orphans,
                     "retrieval": {key: result[key] for key in ("selected", "assistant_ranking", "fused_ranking", "reserved_user_prior", "rrf_k0")}})
        if len(rows) % 10 == 0:
            print(json.dumps({"prepared_dev_questions": len(rows)}), flush=True)
    protocol["prepared_inputs"] = experiment.input_manifest(rows)
    protocol["global_spend_caps_usd"] = json.loads((ROOT / reader.PREREGISTRATION).read_text())["spend_caps_usd"]
    work.mkdir(parents=True, exist_ok=True)
    reader.write(work / "protocol.json", protocol)
    ledger = experiment.refresh_ledger(work)
    v1_path = ROOT / "benchmarks/results/longmemeval_dev_answers_v1.json"
    files = [Path(__file__), Path(experiment.__file__), Path(user_first.__file__), Path(reader.__file__),
             Path(__file__).with_name("longmemeval_role_retrieval.py"),
             Path(__file__).with_name("longmemeval_dev_conversations.py"),
             Path(__file__).with_name("longmemeval_answer_evidence.py"),
             Path(__file__).with_name("check_longmemeval_answers.py"),
             Path(__file__).parent / "proxies/beir_proxies.py", SOURCE,
             user_first.BASELINE, v1_path, user_first.HISTORICAL, user_first.SPLIT, user_first.EXPORT,
             ROOT / reader.PROTOCOL, ROOT / reader.PREREGISTRATION,
             artifact_path(protocol["judge_source"]["path"])]
    report = {"schema_version": 1, "dev_only": True, "heldout_run": False, "reader_systems": [experiment.SYSTEM],
              "variant": work.name, "complete": False, "rows": rows,
              "protocol_path": str((work / "protocol.json").relative_to(ROOT)), "protocol_sha256": reader.sha(work / "protocol.json"),
              "starting_spend_usd": ledger["total_usd"],
              "artifacts": [{"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                             "sha256": reader.sha(path)} for path in files]}
    experiment.save(report, work)
    print(json.dumps(report["summary"]), flush=True)


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
            experiment.run_stage(report, work, stage)
        print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
