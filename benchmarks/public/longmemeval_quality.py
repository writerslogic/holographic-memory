# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Diagnose existing LongMemEval dev answers offline, without reading the corpus or making calls."""

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

from check_longmemeval_answers import (
    ARMS,
    ROOT,
    artifact_path,
    file_digest,
    recompute_scores,
    report_systems,
    require,
    same,
    wilson,
)

OUTCOMES = (
    "correct",
    "selection_failure",
    "reader_failure",
    "judge_failure",
    "judge_rejected_abstention",
    "judge_rejected_answer",
)


def index_dev_rows(rows, dev, name):
    require(isinstance(rows, list) and len(rows) == 100, f"{name} must contain 100 dev rows")
    indexed = {}
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("qid"), str), f"invalid {name} row")
        qid = row["qid"]
        require(qid in dev and qid not in indexed, f"{name} contains non-dev or duplicate question")
        require(isinstance(row.get("qtype"), str) and row["qtype"], f"invalid {name} question type")
        indexed[qid] = row
    require(set(indexed) == dev, f"{name} does not cover frozen dev membership")
    return indexed


def validate_report(report, split):
    require(isinstance(report, dict) and isinstance(split, dict), "invalid report or split")
    require(type(report.get("schema_version")) is int and report["schema_version"] == 1 and report.get("dev_only") is True
            and report.get("heldout_run") is False and report.get("complete") is True,
            "expected a complete dev-only answer report")
    for name, count in (("dev", 100), ("heldout", 400)):
        ids = split.get(name)
        require(isinstance(ids, list) and len(ids) == count
                and all(isinstance(qid, str) and qid for qid in ids)
                and len(set(ids)) == count, f"invalid frozen {name} split")
    dev = set(split["dev"])
    require(not dev.intersection(split["heldout"]), "overlapping dev and held-out split")
    return index_dev_rows(report.get("rows"), dev, "answer report")


def outcome(arm, scores):
    if arm["reader"]["status"] == "selection_failure":
        return "selection_failure"
    if arm["reader"]["status"] != "ok":
        return "reader_failure"
    if arm["judge"]["status"] != "ok":
        return "judge_failure"
    if scores["correct"]:
        return "correct"
    return "judge_rejected_abstention" if scores["abstain"] else "judge_rejected_answer"


def aggregate(records):
    n = len(records)
    correct = sum(record["correct"] for record in records)
    counts = Counter(record["outcome"] for record in records)
    abstentions = [record for record in records if record["abstain"] is True]
    scored = [record for record in records if record["selected_evidence_complete"] is not None]
    claim_count = sum(record["citation_total_claims"] for record in records)
    valid_claims = sum(record["citation_valid_claims"] for record in records)
    return {
        "n": n,
        "correct_count": correct,
        "answer_accuracy": correct / n,
        "answer_accuracy_ci95": wilson(correct, n),
        "outcomes": {name: counts[name] for name in OUTCOMES},
        "abstention": {
            "expected_answerable": sum(not record["expected_abstention"] for record in records),
            "expected_unanswerable": sum(record["expected_abstention"] for record in records),
            "on_answerable": sum(not record["expected_abstention"] for record in abstentions),
            "on_unanswerable": sum(record["expected_abstention"] for record in abstentions),
            "judge_accepted": sum(record["correct"] for record in abstentions),
            "judge_rejected": sum(record["outcome"] == "judge_rejected_abstention" for record in abstentions),
        },
        "selected_evidence": {
            "scored": len(scored),
            "complete": sum(record["selected_evidence_complete"] for record in scored),
            "incomplete": sum(not record["selected_evidence_complete"] for record in scored),
            "incorrect_with_complete": sum(not record["correct"] and record["selected_evidence_complete"]
                                           for record in scored),
            "incorrect_with_incomplete": sum(not record["correct"] and not record["selected_evidence_complete"]
                                             for record in scored),
        },
        "citation_integrity": {
            "exact_quote_claims": valid_claims,
            "total_claims": claim_count,
            "exact_quote_fraction": valid_claims / claim_count if claim_count else None,
        },
        "questions_with_dropped_source_groups": sum(record["dropped_source_groups"] > 0 for record in records),
    }


def diagnose(report, split, evidence):
    rows = validate_report(report, split)
    require(isinstance(evidence, dict), "invalid evidence report")
    evidence_rows = index_dev_rows(evidence.get("rows"), set(split["dev"]), "evidence report")
    systems = report_systems(report)
    records = []
    for qid in sorted(rows):
        row, support = rows[qid], evidence_rows[qid]
        require(row["qtype"] == support["qtype"] and type(support.get("scored")) is bool,
                "invalid evidence question type or scoring flag")
        require(isinstance(support.get("arms"), dict) and set(ARMS) <= set(support["arms"]),
                "missing evidence arm")
        require(set(row["systems"]) == set(systems), "reader systems differ across questions")
        for system in systems:
            arms = row["systems"][system]["arms"]
            require(set(arms) == set(ARMS), "missing or unexpected answer arm")
            for name in ARMS:
                arm = arms[name]
                complete = support["arms"][name]["complete"]
                require(type(complete) is bool, "invalid evidence completeness flag")
                scores = recompute_scores(arm)
                same(arm["scores"], scores, "stored answer score mismatch")
                dropped = arm["reader_cap"]["dropped_group_ids"]
                require(isinstance(dropped, list) and all(isinstance(value, str) for value in dropped)
                        and len(set(dropped)) == len(dropped), "invalid dropped source groups")
                records.append({
                    "qid": qid, "qtype": row["qtype"], "system": system, "arm": name,
                    "outcome": outcome(arm, scores), "correct": scores["correct"],
                    "expected_abstention": "_abs" in qid,
                    "abstain": scores["abstain"] if arm["reader"]["status"] == "ok" else None,
                    "reader_status": arm["reader"]["status"], "judge_status": arm["judge"]["status"],
                    "selected_evidence_complete": complete if support["scored"] else None,
                    "dropped_source_groups": len(dropped),
                    "citation_valid_claims": scores["citation_valid_claims"],
                    "citation_total_claims": scores["citation_total_claims"],
                })
    summaries = {}
    for system in systems:
        summaries[system] = {}
        for name in ARMS:
            selected = [record for record in records if record["system"] == system and record["arm"] == name]
            summary = aggregate(selected)
            summary["by_question_type"] = {
                qtype: aggregate([record for record in selected if record["qtype"] == qtype])
                for qtype in sorted({record["qtype"] for record in selected})
            }
            summaries[system][name] = summary
    return {
        "schema_version": 1,
        "dev_only": True,
        "heldout_run": False,
        "new_model_or_judge_calls": 0,
        "corpus_read": False,
        "limitations": [
            "Repeated-use exploratory dev results; no held-out or generalization claim.",
            "Existing official judge verdicts are reused, not independently judged again.",
            "Expected abstention uses the frozen _abs question label; judge acceptance is separate.",
            "Evidence completeness is labeled user-turn coverage before the reader token cap, not semantic sufficiency.",
            "Unscored evidence is unknown; incomplete evidence is not proof the answer cannot be inferred.",
            "Exact quotation validity does not establish claim entailment or answer correctness.",
            "Selected support and answer outcomes are associations, not retrieval-versus-reasoning causal diagnoses.",
        ],
        "systems": summaries,
        "failures": [record for record in records if not record["correct"]],
        "abstention_label_judge_disagreements": [
            record for record in records if record["abstain"] is True and not record["expected_abstention"]
            and record["correct"]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path,
                        default=ROOT / "benchmarks/results/longmemeval_dev_answers_v3.json", nargs="?")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmarks/results/longmemeval_dev_quality_v3.json")
    args = parser.parse_args()
    split_path = Path(__file__).with_name("longmemeval_split.json")
    report = json.loads(args.report.read_text())
    split = json.loads(split_path.read_text())
    validate_report(report, split)
    evidence_path = artifact_path(report["evidence_report_path"])
    require(evidence_path.resolve().is_relative_to((ROOT / "benchmarks/results").resolve()),
            "evidence must be a benchmark result, not a raw corpus")
    require(file_digest(evidence_path) == report["evidence_report_sha256"], "evidence fingerprint mismatch")
    evidence = json.loads(evidence_path.read_text())
    require(file_digest(split_path) == evidence["split_sha256"], "frozen split fingerprint mismatch")
    result = diagnose(report, split, evidence)
    result["inputs"] = {name: {"path": str(path.resolve().relative_to(ROOT)), "sha256": file_digest(path)}
                        if path.resolve().is_relative_to(ROOT) else {"path": str(path.resolve()), "sha256": file_digest(path)}
                        for name, path in (("answers", args.report), ("evidence", evidence_path),
                                           ("split", split_path), ("diagnostic", Path(__file__)),
                                           ("score_checker", Path(__file__).with_name("check_longmemeval_answers.py")))}
    require(args.output.resolve() not in {artifact_path(item["path"]).resolve() for item in result["inputs"].values()},
            "output would replace an input")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.close()
            temporary.replace(args.output)
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps({system: {name: {key: value[key] for key in ("correct_count", "n", "outcomes")}
                              for name, value in arms.items()} for system, arms in result["systems"].items()}))


if __name__ == "__main__":
    main()
