# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13"
# dependencies = ["tiktoken==0.14.0", "tokenizers==0.23.3", "openai==3.27.0", "python-dotenv==1.2.4"]
# ///
"""Add session-opening user context and measure retained support offline on frozen dev."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

SEED_TURNS = 20
OPENING_TURNS = 2
MAX_CANDIDATES = 64


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def closure_candidates(sessions, baseline, limit=64):
    _require(type(limit) is int and 1 <= limit <= MAX_CANDIDATES,
             "candidate limit must be in 1..=64")
    _require(isinstance(sessions, list) and sessions, "source sessions must be a nonempty list")
    occurrences = {}
    for index, session in enumerate(sessions):
        _require(isinstance(session, dict) and isinstance(session.get("turns"), list)
                 and session["turns"], "source session must contain user turns")
        for turn in session["turns"]:
            _require(isinstance(turn, list) and len(turn) == 2
                     and isinstance(turn[0], str) and turn[0] and isinstance(turn[1], str),
                     "invalid user turn")
            occurrences.setdefault(turn[0], []).append(index)
    _require(isinstance(baseline, list) and 1 <= len(baseline) <= MAX_CANDIDATES
             and all(isinstance(tid, str) and tid for tid in baseline), "invalid baseline candidates")
    _require(len(set(baseline)) == len(baseline), "baseline candidates must be unique")
    _require(all(tid in occurrences for tid in baseline), "baseline candidate has no source observation")
    seeded_sessions = list(dict.fromkeys(index for tid in baseline[:SEED_TURNS]
                                         for index in occurrences[tid]))
    opening = [turn[0] for index in seeded_sessions for turn in sessions[index]["turns"][:OPENING_TURNS]]
    return list(dict.fromkeys(baseline[:SEED_TURNS] + opening + baseline[SEED_TURNS:]))[:limit]


def offline_proof(output):
    import tiktoken

    from check_longmemeval_answers import artifact_path, canonical, digest, file_digest, same
    from longmemeval_answer_evidence import SOURCE_SHA256
    from longmemeval_dev_sources import load_dev_sources
    from longmemeval_reader_budget import cap_sources_user_first
    from longmemeval_user_first import (
        BASELINE, HISTORICAL, ROOT, SOURCE_WORK, SPLIT, SYSTEM, load_protocol, membership,
        reader, serialize, source_groups,
    )

    split = json.loads(SPLIT.read_text())
    dev = set(split["dev"])
    baseline_report = json.loads(BASELINE.read_text())
    _require(baseline_report.get("dev_only") is True and baseline_report.get("heldout_run") is False
             and baseline_report.get("complete") is True, "expected complete dev baseline")
    baseline_rows = membership(baseline_report["rows"], dev, "baseline")
    v1_path = ROOT / "benchmarks/results/longmemeval_dev_answers_v1.json"
    v1_report = json.loads(v1_path.read_text())
    _require(v1_report.get("dev_only") is True and v1_report.get("heldout_run") is False
             and v1_report.get("complete") is True, "expected complete dev source snapshots")
    v1_rows = membership(v1_report["rows"], dev, "source snapshots")
    evidence_path = artifact_path(baseline_report["evidence_report_path"])
    _require(evidence_path.resolve().is_relative_to((ROOT / "benchmarks/results").resolve()),
             "candidate evidence must be a benchmark result")
    _require(file_digest(evidence_path) == baseline_report["evidence_report_sha256"],
             "frozen candidate fingerprint mismatch")
    evidence_rows = membership(json.loads(evidence_path.read_text())["rows"], dev, "candidate evidence")
    historical = json.loads(HISTORICAL.read_text())
    historical_rows = membership(historical["rows"], dev, "historical support")
    manifest_path = SOURCE_WORK / "manifest.json"
    _require(file_digest(manifest_path) == historical["source_manifest_sha256"],
             "frozen source manifest fingerprint mismatch")
    sessions = load_dev_sources(SOURCE_WORK, SPLIT)
    protocol = load_protocol()
    local_cap = protocol["token_cap"] - 512
    _require(local_cap > 0, "invalid local reader token cap")
    tokenizer = tiktoken.get_encoding("o200k_base")
    input_paths = {
        "baseline": BASELINE, "v1_snapshots": v1_path, "candidate_evidence": evidence_path,
        "historical_support": HISTORICAL, "split": SPLIT, "source_manifest": manifest_path,
        "protocol": ROOT / reader.PROTOCOL, "preregistration": ROOT / reader.PREREGISTRATION,
        "closure_code": Path(__file__), "closure_tests": Path(__file__).with_name("test_longmemeval_session_closure.py"),
        "source_loader": Path(__file__).with_name("longmemeval_dev_sources.py"),
        "user_first_runner": Path(__file__).with_name("longmemeval_user_first.py"),
        "reader_budget": Path(__file__).with_name("longmemeval_reader_budget.py"),
        "source_payload": Path(__file__).with_name("longmemeval_answer_evidence.py"),
        "reader_protocol_code": Path(__file__).with_name("longmemeval_reader_api.py"),
        "protocol_checker": Path(__file__).with_name("check_longmemeval_answers.py"),
    }
    output = Path(output).resolve()
    _require(output.is_relative_to((ROOT / "benchmarks/results").resolve()),
             "proof output must be a benchmark result")
    _require(output not in {path.resolve() for path in input_paths.values()}, "proof would overwrite an input")
    inputs = {name: {"path": str(path.resolve().relative_to(ROOT)), "sha256": file_digest(path)}
              for name, path in input_paths.items()}
    rows = []
    for qid in sorted(dev):
        question = baseline_rows[qid]
        original = evidence_rows[qid]["candidates"]
        closed = closure_candidates(sessions[qid], original)
        decisions = {}
        for name, candidates in (("original", original), ("closure", closed)):
            groups = source_groups(qid, sessions[qid], candidates, [v1_rows[qid], question])
            arm = cap_sources_user_first(
                {"source_groups": groups}, lambda retained: serialize(question, retained, protocol),
                lambda text: len(tokenizer.encode(text)), local_cap)
            failed = arm["reader_cap"]["selection_failure"]
            prompt = "" if failed else serialize(question, arm["source_groups"], protocol)
            users = [source for group in arm["source_groups"] for source in group["sources"]
                     if source["role"] == "user"]
            preserved = [group["frozen_selected_id_sha256"] for group in arm["source_groups"]]
            decisions[name] = {
                "candidate_aliases": candidates,
                "candidate_sha256": digest(canonical(candidates)),
                "preserved_candidate_ids": preserved,
                "preserved_user_source_ids": [source["source_id"] for source in users],
                "preserved_user_occurrences": [[source["session_index"], source["turn_index"]]
                                                for source in users],
                "reader_input_sha256": digest(prompt),
                "input_tokens": arm["reader_cap"]["input_tokens"],
                "uncapped_input_tokens": arm["reader_cap"]["uncapped_input_tokens"],
                "selection_failure": failed,
                "dropped_group_ids": arm["reader_cap"]["dropped_group_ids"],
            }
        support = historical_rows[qid]
        _require(type(support.get("scored")) is bool, "invalid support scoring flag")
        gold = {digest(tid) for tid in support["ids"] if "answer" in tid}
        for name, decision in decisions.items():
            retained_gold = gold.intersection(decision["preserved_candidate_ids"])
            decision.update(gold_turn_count=len(gold), retained_gold_turn_count=len(retained_gold),
                            full_support=(gold <= set(decision["preserved_candidate_ids"])
                                          and not decision["selection_failure"]) if support["scored"] else None)
        rows.append({"qid": qid, "scored": support["scored"],
                     "source_sessions_sha256": digest(canonical(sessions[qid])),
                     "original": decisions["original"], "closure": decisions["closure"]})
    scored = [row for row in rows if row["scored"]]
    regressions = [row["qid"] for row in scored if row["original"]["full_support"]
                   and not row["closure"]["full_support"]]
    summary = {
        "n_dev": len(rows), "n_scored": len(scored),
        "original_complete": sum(row["original"]["full_support"] for row in scored),
        "closure_complete": sum(row["closure"]["full_support"] for row in scored),
        "improvements": [row["qid"] for row in scored if row["closure"]["full_support"]
                         and not row["original"]["full_support"]],
        "regressions": regressions,
        "remaining_incomplete": [row["qid"] for row in scored if not row["closure"]["full_support"]],
        "max_input_tokens": {name: max(row[name]["input_tokens"] for row in rows)
                             for name in ("original", "closure")},
        "selection_failures": {name: sum(row[name]["selection_failure"] for row in rows)
                               for name in ("original", "closure")},
    }
    for item in inputs.values():
        same(file_digest(ROOT / item["path"]), item["sha256"], "proof input changed during measurement")
    result = {
        "schema_version": 1, "dev_only": True, "heldout_run": False, "complete": True,
        "new_model_or_judge_calls": 0, "raw_corpus_read": False,
        "sources": {"release_sha256": SOURCE_SHA256,
                    "sessions": sum(map(len, sessions.values())),
                    "user_turns": sum(len(source["turns"]) for records in sessions.values() for source in records)},
        "policy": {"seed_turns": SEED_TURNS, "opening_turns_per_seeded_session": OPENING_TURNS,
                   "candidate_limit": MAX_CANDIDATES, "token_cap": protocol["token_cap"],
                   "local_token_cap": local_cap, "tokenizer": "o200k_base",
                   "tokenizer_version": version("tiktoken")},
        "limitations": [
            "Exploratory repeated-use dev coverage; no held-out or answer accuracy claim.",
            "Full support means labeled user-turn coverage after the reader cap, not semantic sufficiency.",
            "Source identities can repeat; each retained alias preserves all matching observations.",
            "Labels are joined only after candidate generation and reader source selection.",
        ],
        "inputs": inputs, "rows": rows, "summary": summary,
    }
    reader.write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[2] / "benchmarks/results/longmemeval_session_closure_v1.json")
    result = offline_proof(parser.parse_args().output)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
