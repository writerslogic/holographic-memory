# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///
"""Independently check the optimized SQLite control against original witnesses."""

import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from check_document_competition import check


def main(campaign):
    initial = check(campaign)
    folder = campaign / "rank_optimized"
    protocol = json.loads((folder / "protocol.json").read_text())
    rows = json.loads((folder / "measurements.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    assert protocol["query"] == "SELECT rowid, rank FROM docs WHERE docs MATCH ? ORDER BY rank LIMIT ?"
    assert protocol["candidate_binary_sha256"] == hashlib.sha256((campaign / "candidate-probe").read_bytes()).hexdigest()
    assert 1 <= protocol["rounds"] <= 3 and len(rows) == 4 * protocol["rounds"]
    assert {(row["dataset"], row["round"], row["arm"]) for row in rows} == {
        (dataset, repetition, arm) for dataset in ("scifact", "nfcorpus")
        for repetition in range(protocol["rounds"]) for arm in ("candidate", "sqlite_rank")}
    pairs = 0
    for dataset, split in [("scifact", "train"), ("nfcorpus", "dev")]:
        inputs = json.loads((campaign / f"{dataset}-{split}-input.json").read_text())
        identifiers = {row["id"] for row in inputs["documents"]}
        qrels = json.loads((campaign / f"{dataset}-{split}-qrels.json").read_text())
        baseline = json.loads(gzip.decompress((campaign / f"{dataset}-0-baseline.json.gz").read_bytes()))
        reference = {row["id"]: row["results"] for row in baseline["rows"]}
        times, quality = {}, {}
        for measurement in (row for row in rows if row["dataset"] == dataset):
            raw_path = folder / measurement["artifact"]
            assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == measurement["sha256"]
            raw = json.loads(gzip.decompress(raw_path.read_bytes()))
            assert not raw["ingest"] and raw["documents"] == len(identifiers)
            assert raw["open_ns"] == measurement["open_ns"] > 0
            assert len(raw["rows"]) == len(qrels) and {row["id"] for row in raw["rows"]} == set(qrels)
            assert 0 < measurement["peak_process_rss_bytes"] <= 8 * 1024 ** 3
            assert measurement["load_gate_met"] == (measurement["load_before"][0] < 3 and measurement["load_after"][0] < 3)
            arm = measurement["arm"]
            latency, ndcg, recall = [], [], []
            for row in raw["rows"]:
                assert isinstance(row["elapsed_ns"], int) and row["elapsed_ns"] > 0
                latency.append(row["elapsed_ns"])
                if arm == "candidate":
                    assert row["results"] == reference[row["id"]]
                    pairs += 1
                ids = [hit["documentId"] for hit in row["results"]]
                assert len(ids) <= 100 and len(ids) == len(set(ids)) and set(ids) <= identifiers
                expected = qrels[row["id"]]
                grades = sorted(expected.values(), reverse=True)
                ideal = sum(grade / math.log2(i + 2) for i, grade in enumerate(grades[:10]))
                observed = sum(expected.get(key, 0) / math.log2(i + 2) for i, key in enumerate(ids[:10]))
                ndcg.append(observed / ideal if ideal else 0.0)
                positive = {key for key, grade in expected.items() if grade > 0}
                recall.append(len(set(ids) & positive) / len(positive) if positive else 0.0)
            quality[arm] = {"ndcg@10": statistics.mean(ndcg), "recall@100": statistics.mean(recall)}
            times.setdefault(arm, []).extend(latency)
        produced = summary["results"][dataset]
        assert produced["quality"] == quality and produced["queries"] == len(qrels)
        for arm, values in times.items():
            values.sort()
            actual = produced["latency"][arm]
            assert actual["mean_ns"] == statistics.mean(values)
            for label, q in [("p95_ns", .95), ("p99_ns", .99)]:
                index = (len(values) - 1) * q
                lo, hi = int(index), min(int(index) + 1, len(values) - 1)
                assert actual[label] == values[lo] + (values[hi] - values[lo]) * (index - lo)
        assert produced["sqlite_over_candidate_mean"] == produced["latency"]["sqlite_rank"]["mean_ns"] / produced["latency"]["candidate"]["mean_ns"]
    assert summary["load_gate_met"] == all(row["load_gate_met"] for row in rows)
    result = {"initial": initial, "optimized_checked_runs": len(rows),
              "optimized_exact_candidate_query_pairs": pairs, "quality_reconstructed": True,
              "load_gate_met": summary["load_gate_met"]}
    (folder / "checked.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
