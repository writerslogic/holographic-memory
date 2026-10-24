import csv
import json
import math
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
    expected_algorithms = {
        "glove-100-angular": "scann",
        "sift-128-euclidean": "faiss-ivfpqfs",
        "fashion-mnist-784-euclidean": "glass",
    }
    standings = {}
    for dataset in datasets:
        comparator = expected_algorithms[dataset]
        dataset_points = [p for p in points if p["dataset"] == dataset]
        expected = {"hms": 108, comparator: {"scann": 84, "faiss-ivfpqfs": 96, "glass": 198}[comparator]}
        for algorithm, count in expected.items():
            observed = [p for p in dataset_points if p["algorithm"] == algorithm]
            if len(observed) != count or len({p["parameters"] for p in observed}) != count:
                raise ValueError(f"{dataset}/{algorithm}: expected {count} unique points, found {len(observed)}")
        standings[dataset] = {}
        for target in (0.9, 0.95, 0.99):
            hms = max(
                (p for p in dataset_points if p["algorithm"] == "hms" and p["recall_at_10"] >= target),
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
    output = {"manifest": manifest, "standings": standings, "points": points}
    out_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
