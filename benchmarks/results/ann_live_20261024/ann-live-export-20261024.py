# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
import csv
from collections import defaultdict, deque
import json
import hashlib
import math
import re
import sys
from pathlib import Path


def metric(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if not value:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def main() -> None:
    csv_path, manifest_path, out_path = map(Path, sys.argv[1:4])
    manifest = json.loads(manifest_path.read_text())
    datasets = manifest.get("datasets") or [manifest["dataset"]]
    if not isinstance(datasets, list) or not all(isinstance(dataset, str) for dataset in datasets):
        raise ValueError("manifest dataset names must be strings")
    points = []
    metadata_path = csv_path.with_name("raw-point-metadata.json")
    raw_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    identities = defaultdict(deque)
    if raw_metadata is not None:
        if len({point["file"] for point in raw_metadata}) != len(raw_metadata):
            raise ValueError("raw result file identities must be unique")
        for point in raw_metadata:
            identities[(point["algorithm"], point["parameters"], point["qps"], point["recall_at_10"])].append(point)
    with csv_path.open(newline="") as source:
        for row in csv.DictReader(source):
            if row["dataset"] not in datasets:
                continue
            if row["algorithm"] not in {"hms", "scann", "faiss-ivfpqfs", "glass"}:
                continue
            recall = metric(row, "k-nn")
            qps = metric(row, "qps")
            if recall is None or not 0 <= recall <= 1 or qps is None or qps <= 0:
                raise ValueError(f"missing or invalid recall/QPS: {row['dataset']} {row['parameters']}")
            points.append(
                {
                    "dataset": row["dataset"],
                    "algorithm": row["algorithm"],
                    "parameters": row["parameters"],
                    "recall_at_10": recall,
                    "qps": qps,
                    "epsilon_recall": metric(row, "epsilon"),
                    "large_epsilon_recall": metric(row, "largeepsilon"),
                    "relative_distance_error": metric(row, "rel"),
                    "p50_ms": metric(row, "p50"),
                    "p95_ms": metric(row, "p95"),
                    "p99_ms": metric(row, "p99"),
                    "p999_ms": metric(row, "p999"),
                    "build_s": metric(row, "build"),
                    "index_size_kb": metric(row, "indexsize"),
                    "candidates": metric(row, "candidates"),
                    "distance_computations": metric(row, "distcomps"),
                    "query_size_kb": metric(row, "queriessize"),
                }
            )
            if raw_metadata is not None:
                identity = (row["algorithm"], row["parameters"], qps, recall)
                if not identities[identity]:
                    raise ValueError(f"official CSV row has no unused raw result identity: {identity}")
                native = identities[identity].popleft()
                points[-1]["raw_result"] = native["file"]
                if "query_ef" in native:
                    points[-1]["query_ef"] = native["query_ef"]
                if row["algorithm"] == "scann":
                    points[-1]["query_arguments"] = [int(value) for value in Path(native["file"]).stem.rsplit("_", 2)[-2:]]
    if raw_metadata is not None and any(identities.values()):
        raise ValueError("some raw result identities are absent from the official CSV")
    expected_algorithms = {
        "glove-100-angular": "scann",
        "sift-128-euclidean": "faiss-ivfpqfs",
        "fashion-mnist-784-euclidean": "glass",
    }
    matched_degrees = manifest.get("matched_hms_degrees")
    if matched_degrees is not None:
        if not matched_degrees or any(type(degree) is not int or degree not in (32, 48, 64) for degree in matched_degrees):
            raise ValueError("matched_hms_degrees must be a nonempty subset of 32/48/64")
    for point in points:
        matched = True
        if point["algorithm"] == "hms" and matched_degrees is not None:
            degree = re.search(r"degree=(32|48|64),", point["parameters"])
            if not degree:
                raise ValueError(f"Missing HMS degree provenance: {point['parameters']}")
            matched = int(degree[1]) in matched_degrees
        point["same_container_comparison"] = matched
    standings = {}
    for dataset in datasets:
        comparator = expected_algorithms[dataset]
        dataset_points = [p for p in points if p["dataset"] == dataset]
        expected = {"hms": 108, comparator: {"scann": 84, "faiss-ivfpqfs": 96, "glass": 198}[comparator]}
        for algorithm, count in expected.items():
            observed = [p for p in dataset_points if p["algorithm"] == algorithm]
            key = "raw_result" if raw_metadata is not None else "parameters"
            if len(observed) != count or len({p[key] for p in observed}) != count:
                raise ValueError(f"{dataset}/{algorithm}: expected {count} unique points, found {len(observed)}")
        standings[dataset] = {}
        for target in (0.9, 0.95, 0.99):
            hms = max(
                (p for p in dataset_points if p["algorithm"] == "hms" and p["same_container_comparison"] and p["recall_at_10"] >= target),
                key=lambda p: p["qps"],
                default=None,
            )
            competitor = max(
                (p for p in dataset_points if p["algorithm"] == comparator and p["recall_at_10"] >= target),
                key=lambda p: p["qps"],
                default=None,
            )
            standings[dataset][str(target)] = {
                "hms": hms,
                "competitor": competitor,
                "hms_over_competitor_qps": (
                    hms["qps"] / competitor["qps"] if hms and competitor else None
                ),
            }
    output = {"manifest": manifest, "standings": standings, "points": points,
              "input_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
              "input_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "raw_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest() if raw_metadata is not None else None}
    out_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
