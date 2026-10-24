"""Regenerate conditional development diagnostics; never a competitor/leaderboard claim."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import zlib


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprints(path):
    sha = hashlib.sha256()
    crc = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            sha.update(block)
            crc = zlib.crc32(block, crc)
    return {"sha256": sha.hexdigest(), "crc32": crc}


def quantile(values, probability):
    ordered = sorted(values)
    rank = (len(ordered) - 1) * probability
    lo = math.floor(rank)
    hi = math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def interval(values, *, logarithmic=False):
    if len(values) < 2:
        return None
    rng = random.Random(539365412)
    xs = [math.log(v) for v in values] if logarithmic else values
    means = [statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(10000)]
    bounds = [quantile(means, p) for p in [0.025, 0.975]]
    return [math.exp(v) for v in bounds] if logarithmic else bounds


def load(path):
    require(path.stat().st_size <= 256 * 1024 * 1024, "result exceeds 256 MiB analyzer bound")
    with path.open() as stream:
        raw = json.load(stream)
    require(raw["schema"] == "hms-glove-development-v1" and raw["development_only"] is True, "wrong schema/scope")
    require((raw["construction_count"], raw["query_count"], raw["dim"], raw["k"]) == (65536, 256, 100, 10), "wrong geometry")
    require(1 <= raw["repeats"] <= 1000 and 1 <= len(raw["efs"]) <= 16 and raw["repeats"] * len(raw["efs"]) <= 1000, "wrong repeat grid")
    require(len(set(raw["efs"])) == len(raw["efs"]) and all(10 <= ef <= 4096 for ef in raw["efs"]), "wrong ef grid")
    paths = {name: Path(raw["input"]) / name for name in ["construction.f32", "development.f32", "declaration.json"]}
    paths["index"] = Path(raw["index"])
    bindings = {name: fingerprints(p) for name, p in paths.items()}
    for name, binding in bindings.items():
        require(binding["crc32"] == raw["binding_crc32"][name], f"{name} bytes changed since execution")
    require(paths["index"].stat().st_size == raw["serialized_index_bytes"], "index length mismatch")
    declaration = json.loads(paths["declaration.json"].read_text())
    require(declaration == raw["input_declaration"] and declaration["official_test_arrays_read"] is False, "declaration mismatch")
    for name in ["construction.f32", "development.f32"]:
        require(bindings[name]["sha256"] == declaration["files"][name], "prepared matrix SHA mismatch")
    truth = raw["truth"]
    require(len(truth) == 256 and all(len(row) == 10 and len(set(row)) == 10 and all(type(v) is int and 0 <= v < 65536 for v in row) for row in truth), "invalid exact IDs")
    cells = {}
    for trial in raw["trials"]:
        cell = (trial["ef"], trial["repeat"])
        require(cell not in cells and cell[0] in raw["efs"] and 0 <= cell[1] < raw["repeats"], "duplicate/unexpected trial")
        require(sorted(trial["query_order"]) == list(range(256)), "query order is not a permutation")
        require(len(trial["latencies_ns"]) == 256 and all(type(v) is int and v > 0 for v in trial["latencies_ns"]), "invalid latency")
        require(type(trial["batch_ns"]) is int and trial["batch_ns"] >= sum(trial["latencies_ns"]), "invalid batch duration")
        require(len(trial["distance_evaluations"]) == 256 and all(type(v) is int and v >= 0 for v in trial["distance_evaluations"]), "invalid work counts")
        ids = trial["ids"]
        require(len(ids) == 256 and all(len(row) == 10 and len(set(row)) == 10 and all(type(v) is int and 0 <= v < 65536 for v in row) for row in ids), "invalid result IDs")
        trial["recall_at_10"] = statistics.fmean(len(set(a) & set(b)) / 10 for a, b in zip(ids, truth))
        trial["batch_qps"] = 256e9 / trial["batch_ns"]
        trial["query_only_qps"] = 256e9 / sum(trial["latencies_ns"])
        cells[cell] = trial
    require(len(cells) == len(raw["efs"]) * raw["repeats"], "incomplete curve/repeats")
    summary = {}
    for ef in raw["efs"]:
        trials = [cells[(ef, repeat)] for repeat in range(raw["repeats"])]
        latencies = [v for trial in trials for v in trial["latencies_ns"]]
        summary[str(ef)] = {
            "recall_at_10": statistics.fmean(t["recall_at_10"] for t in trials),
            "batch_qps_by_repeat": [t["batch_qps"] for t in trials],
            "batch_qps_mean": statistics.fmean(t["batch_qps"] for t in trials),
            "query_only_qps_by_repeat": [t["query_only_qps"] for t in trials],
            "query_only_qps_mean": statistics.fmean(t["query_only_qps"] for t in trials),
            "query_only_qps_mean_conditional_bootstrap95": interval([t["query_only_qps"] for t in trials]),
            "latency_ns": {f"p{p:g}": quantile(latencies, p / 100) for p in [50, 95, 99, 99.9]},
            "mean_distance_evaluations": statistics.fmean(v for t in trials for v in t["distance_evaluations"]),
            "ordered_ids_equal_across_repeats": all(t["ids"] == trials[0]["ids"] for t in trials),
        }
    return raw, cells, {"raw": str(path.resolve()), "raw_sha256": fingerprints(path)["sha256"], "bindings": bindings, "logical_index_bytes": raw["logical_index_bytes"], "serialized_index_bytes": raw["serialized_index_bytes"], "curve": summary}


def compare(left, right):
    a, acells, _ = left
    b, bcells, _ = right
    for key in ["construction_count", "query_count", "dim", "k", "rerank", "efs", "repeats", "order_seed", "warmup_passes_per_ef", "truth", "truth_method", "residency", "timing_scope", "input_declaration"]:
        require(a[key] == b[key], f"unmatched {key}")
    for name in ["construction.f32", "development.f32", "declaration.json"]:
        require(a["binding_crc32"][name] == b["binding_crc32"][name], "unmatched fixtures")
    results = {}
    for ef in a["efs"]:
        ratios, batch_ratios, deltas = [], [], []
        identity, equal_work = True, True
        for repeat in range(a["repeats"]):
            x, y = acells[(ef, repeat)], bcells[(ef, repeat)]
            require(x["query_order"] == y["query_order"], "unmatched query order")
            ratios.append(y["query_only_qps"] / x["query_only_qps"])
            batch_ratios.append(y["batch_qps"] / x["batch_qps"])
            deltas.append(y["recall_at_10"] - x["recall_at_10"])
            identity &= y["ids"] == x["ids"]
            equal_work &= y["distance_evaluations"] == x["distance_evaluations"]
        results[str(ef)] = {
            "query_only_qps_right_over_left_by_repeat": ratios,
            "query_only_qps_ratio_geomean": math.exp(statistics.fmean(map(math.log, ratios))),
            "query_only_qps_ratio_conditional_paired_bootstrap95": interval(ratios, logarithmic=True),
            "batch_qps_ratio_geomean": math.exp(statistics.fmean(map(math.log, batch_ratios))),
            "recall_difference": statistics.fmean(deltas),
            "ordered_ids_equal": identity, "distance_evaluations_equal": equal_work,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--disabled", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "analysis output exists")
    require((args.candidate is None) == (args.disabled is None), "candidate and disabled must be supplied together")
    loaded = {name: load(path) for name, path in [("baseline", args.baseline), ("candidate", args.candidate), ("disabled", args.disabled)] if path is not None}
    result = {
        "schema": "hms-glove-development-analysis-v1", "development_only": True,
        "analyzer_sha256": fingerprints(Path(__file__))["sha256"],
        "uncertainty_scope": "10000 fixed-seed percentile bootstrap resamples of whole repeat IDs, paired by matching query order for ratios. Repeats share a process/index/hardware, so intervals are conditional diagnostics and exclude independent process/build/hardware uncertainty. Query observations are never treated as independent repeat samples; pooled tail quantiles are descriptive only. Separately timed processes can drift; pairing alone does not establish causality.",
        "binding_scope": "SHA256 of raw outputs and current files whose CRC32 matches execution. Caller must additionally preserve SHA-bound native source/binary, commands, environment, resource receipts and arm execution order.",
        "memory_scope": "Logical native index bytes and serialized file bytes only; no RSS/USS/peak memory inference.",
        "runs": {name: data[2] for name, data in loaded.items()},
        "comparisons": {},
    }
    if args.candidate is not None:
        result["comparisons"] = {f"{right}_over_{left}": compare(loaded[left], loaded[right]) for left, right in [("baseline", "candidate"), ("baseline", "disabled"), ("disabled", "candidate")]}
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
