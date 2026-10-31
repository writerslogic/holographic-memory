# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Dev-only lossless evidence completion against the frozen full-turn selector."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from longmemeval_answer_evidence import _groups_for_question

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "bindings/python/warm-sdk/writerslogic_warm_sdk/evidence.py"
SOURCE_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("warm_source_evidence", MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.home() / ".cache/warm-bench/downloads/longmemeval_s_cleaned.json")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/memory_competition_v1/report.json")
    args = parser.parse_args()
    work = args.output.parent
    protocol = work / "protocol.json"
    baseline_path = ROOT / "benchmarks/results/evidence_budget_dev.json"
    baseline_check = ROOT / "benchmarks/results/evidence_budget_summary.json"
    historical_path = ROOT / "benchmarks/results/evidence_completeness_dev.json"
    split_path = ROOT / "benchmarks/public/longmemeval_split.json"
    assert protocol.is_file(), "freeze protocol before measurement"
    assert sha(args.source) == SOURCE_SHA256
    assert sha(baseline_path) == json.loads(baseline_check.read_text())["report_sha256"]
    baseline = json.loads(baseline_path.read_text())
    assert sha(historical_path) == baseline["input_sha256"]
    assert sha(split_path) == baseline["split_sha256"]
    split = json.loads(split_path.read_text())
    dev = set(split["dev"])
    assert len(dev) == 100 and len(split["heldout"]) == 400
    historical = {row["qid"]: row for row in json.loads(historical_path.read_text())["rows"]}
    raw_dev = {question["question_id"]: question for question in json.loads(args.source.read_text())
               if question["question_id"] in dev}
    assert set(historical) == set(raw_dev) == dev
    evidence = load_module()
    rows = []
    for frozen in baseline["rows"]:
        old = historical[frozen["qid"]]
        groups, _ = _groups_for_question(raw_dev[frozen["qid"]], set(old["ids"]))
        ranked = list(dict.fromkeys(old["baseline_ranking"]))[:64]
        selected = frozen["arms"]["operand"]["selected"]
        assert set(selected) <= set(ranked)
        reconstructed = json.dumps({"v": 2, "q": [groups[item] for item in selected], "i": []},
                                   ensure_ascii=False, separators=(",", ":"))
        assert reconstructed == frozen["operand"]["packet"]
        candidates = [groups[item] for item in ranked]
        seed = [ranked.index(item) for item in selected]
        result = evidence.complete_evidence(candidates, seed, frozen["budget"])
        retained_ids = [ranked[index] for index in result.selected]
        decoded = evidence.decode_source_packet(result.packet)
        assert decoded == [[tuple(source) for source in groups[item]] for item in retained_ids]
        base_packet = evidence.encode_source_packet([groups[item] for item in selected])
        # Scoring occurs after source-only selection; labels never enter the codec or completer.
        gold = {item for item in old["ids"] if "answer" in item}
        rows.append({"qid": frozen["qid"], "qtype": frozen["qtype"], "scored": frozen["scored"],
                     "budget_bytes": frozen["budget"], "baseline_selected": selected,
                     "baseline_packet_bytes": len(reconstructed.encode()),
                     "factored_baseline_bytes": len(base_packet.encode()),
                     "selected": retained_ids, "packet": result.packet,
                     "packet_bytes": result.packet_bytes,
                     "packet_sha256": hashlib.sha256(result.packet.encode()).hexdigest(),
                     "baseline_complete": gold <= set(selected),
                     "complete": gold <= set(retained_ids)})
    scored = [row for row in rows if row["scored"]]
    report = {"schema_version": 1, "dev_only": True, "heldout_run": False,
              "protocol_sha256": sha(protocol), "source_sha256": sha(args.source),
              "split_sha256": sha(split_path), "baseline_sha256": sha(baseline_path),
              "historical_sha256": sha(historical_path), "codec_sha256": sha(MODULE),
              "producer_sha256": sha(__file__), "candidate_limit": 64,
              "rows": rows, "summary": {"scored_questions": len(scored),
                  "baseline_complete": sum(row["baseline_complete"] for row in scored),
                  "complete": sum(row["complete"] for row in scored),
                  "gains": [row["qid"] for row in scored if row["complete"] and not row["baseline_complete"]],
                  "losses": [row["qid"] for row in scored if row["baseline_complete"] and not row["complete"]],
                  "same_selection_bytes_ratio": sum(row["factored_baseline_bytes"] for row in scored) /
                                                sum(row["baseline_packet_bytes"] for row in scored),
                  "additional_groups": sum(len(row["selected"]) - len(row["baseline_selected"]) for row in scored)},
              "answer_accuracy_measured": False, "reader_token_budget_matched": False,
              "competitor_superiority_established": False}
    work.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1, allow_nan=False) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
