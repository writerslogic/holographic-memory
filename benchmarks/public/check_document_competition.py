# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///
"""Reconstruct identities, quality, exact parity, and timing summaries from raw rows."""

import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def check(folder):
    protocol = json.loads((folder / "protocol.json").read_text())
    datasets = json.loads((folder / "datasets.json").read_text())
    for arm, expected in protocol["binary_sha256"].items():
        assert hashlib.sha256((folder / f"{arm}-probe").read_bytes()).hexdigest() == expected
    for dataset, manifest in datasets.items():
        assert manifest["split"] == protocol["datasets"][dataset]
        for name, expected in manifest["produced_sha256"].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
    measurements = json.loads((folder / "measurements.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    assert protocol["datasets"] == {"scifact": "train", "nfcorpus": "dev"}
    assert 1 <= protocol["rounds"] <= 5
    assert len(measurements) == 6 * protocol["rounds"]
    expected_keys = {(dataset, arm, repetition) for dataset in protocol["datasets"]
                     for arm in ("baseline", "candidate", "sqlite") for repetition in range(protocol["rounds"])}
    assert {(row["dataset"], row["arm"], row["round"]) for row in measurements} == expected_keys
    all_pairs = 0
    for dataset, split in protocol["datasets"].items():
        inputs = json.loads((folder / f"{dataset}-{split}-input.json").read_text())
        qrels = json.loads((folder / f"{dataset}-{split}-qrels.json").read_text())
        documents = {row["id"]: row for row in inputs["documents"]}
        assert len(documents) == len(inputs["documents"])
        assert {row["id"] for row in inputs["queries"]} == set(qrels)
        reference, timings = {}, {}
        quality = {}
        for measurement in (row for row in measurements if row["dataset"] == dataset):
            artifact = folder / measurement["artifact"]
            assert hashlib.sha256(artifact.read_bytes()).hexdigest() == measurement["sha256"]
            raw = json.loads(gzip.decompress(artifact.read_bytes()))
            assert raw["documents"] == len(documents)
            assert len(raw["rows"]) == len(qrels)
            assert {row["id"] for row in raw["rows"]} == set(qrels)
            assert raw["ingest"] == measurement["ingest"] == (measurement["round"] == 0)
            assert raw["open_ns"] == measurement["open_ns"] > 0
            assert 0 < measurement["peak_process_rss_bytes"] <= 8 * 1024 ** 3
            assert measurement["load_gate_met"] == (measurement["load_before"][0] < 3 and measurement["load_after"][0] < 3)
            arm = measurement["arm"]
            timings.setdefault(arm, []).extend(row["elapsed_ns"] for row in raw["rows"])
            ndcg, recall = [], []
            for row in raw["rows"]:
                assert isinstance(row["elapsed_ns"], int) and row["elapsed_ns"] > 0
                results = row["results"]
                ids = [hit["documentId"] for hit in results]
                assert len(ids) <= 100 and len(ids) == len(set(ids)) and set(ids) <= set(documents)
                if arm in ("baseline", "candidate"):
                    if row["id"] in reference:
                        assert results == reference[row["id"]]
                    else:
                        reference[row["id"]] = results
                    assert results == sorted(results, key=lambda hit: (-hit["score"], hit["documentId"]))
                    for hit in results:
                        source = documents[hit["documentId"]]
                        assert hit["id"] == f"hms:chunk:{len(hit['documentId'].encode())}:{hit['documentId']}:0"
                        assert hit["version"] == "1" and hit["metadata"] == source["metadata"]
                        assert hit["text"] is None and hit["semanticScore"] is None and hit["sourceUri"] is None
                        assert hit["startByte"] == 0 and hit["endByte"] == len(source["text"].encode())
                        assert hit["lexicalScore"] > 0 and math.isfinite(hit["score"])
                relevant = qrels[row["id"]]
                positive = {key for key, grade in relevant.items() if grade > 0}
                recall.append(len(set(ids) & positive) / len(positive) if positive else 0.0)
                dcg = sum(relevant.get(key, 0) / math.log2(i + 2) for i, key in enumerate(ids[:10]))
                ideal = sum(grade / math.log2(i + 2) for i, grade in enumerate(sorted(relevant.values(), reverse=True)[:10]))
                ndcg.append(dcg / ideal if ideal else 0.0)
            quality[arm] = {"ndcg@10": statistics.mean(ndcg), "recall@100": statistics.mean(recall)}
        produced = summary["results"][dataset]
        assert produced["quality"] == quality
        assert produced["queries"] == len(qrels)
        assert produced["exact_warm_query_pairs"] == len(qrels) * protocol["rounds"]
        all_pairs += produced["exact_warm_query_pairs"]
        for arm, values in timings.items():
            assert produced["latency"][arm]["mean_ns"] == statistics.mean(values)
            values.sort()
            for label, q in [("p95_ns", .95), ("p99_ns", .99)]:
                index = (len(values) - 1) * q
                lo, hi = int(index), min(int(index) + 1, len(values) - 1)
                expected = values[lo] + (values[hi] - values[lo]) * (index - lo)
                assert produced["latency"][arm][label] == expected
        latency = produced["latency"]
        assert produced["baseline_over_candidate_mean"] == latency["baseline"]["mean_ns"] / latency["candidate"]["mean_ns"]
        assert produced["sqlite_over_candidate_mean"] == latency["sqlite"]["mean_ns"] / latency["candidate"]["mean_ns"]
    assert summary["load_gate_met"] == all(row["load_gate_met"] for row in measurements)
    return {"checked_runs": len(measurements), "exact_warm_query_pairs": all_pairs,
            "quality_reconstructed": True, "load_gate_met": summary["load_gate_met"]}


if __name__ == "__main__":
    folder = Path(sys.argv[1]).resolve()
    report = check(folder)
    (folder / "checked.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
