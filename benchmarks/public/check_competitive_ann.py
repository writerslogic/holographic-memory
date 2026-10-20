# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = "==3.13.9"
# dependencies = ["numpy==2.2.6"]
# ///
"""Independent raw-source cosine oracle and frozen ANN selection checker."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


K = 10
TARGETS = (0.90, 0.95, 0.99)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def verified_inputs(path):
    info = json.loads((path / "inputs.json").read_text())
    dim, rows = info["dim"], info["database_rows"]
    if type(dim) is not int or type(rows) is not int or dim not in (100, 256) or rows not in (10000, 100000, 250000):
        raise ValueError("unregistered input shape")
    if info.get("benchmark_test_opened") is not False:
        raise ValueError("invalid source policy")
    source = Path(info["source_train_path"])
    if source.name != "train.f32" or sha(source) != info["source_train_sha256"]:
        raise ValueError("source training corpus changed")
    for name, digest in info["files"].items():
        if Path(name).name != name or sha(path / name) != digest:
            raise ValueError("input hash or path changed")
    corpus = np.memmap(source, dtype="<f4", mode="r").reshape(-1, dim)
    ids_by_phase = {}
    for name, count in (("train", rows), ("calibration", 80), ("validation", 120)):
        ids = np.load(path / f"{name}_original_ids.npy", allow_pickle=False)
        if ids.dtype.kind not in "iu" or ids.shape != (count,) or len(np.unique(ids)) != count or np.any(ids < 0) or np.any(ids >= len(corpus)):
            raise ValueError("invalid source row ids")
        raw = np.fromfile(path / f"{name}_raw.f32", dtype="<f4").reshape(-1, dim)
        if not np.array_equal(raw, corpus[ids]) or not np.isfinite(raw).all():
            raise ValueError("rows differ from original source ids")
        expected = raw.astype(np.float64)
        norms = np.linalg.norm(expected, axis=1)
        expected /= np.where(norms > 0, norms, 1)[:, None]
        stored = np.fromfile(path / f"{name}.f32", dtype="<f4").reshape(-1, dim)
        if not np.array_equal(stored, expected.astype("<f4")):
            raise ValueError("normalized rows differ from raw source oracle")
        if name != "train" and np.any(norms == 0):
            raise ValueError("zero cosine query")
        ids_by_phase[name] = set(ids.tolist())
    if any(ids_by_phase[left] & ids_by_phase[right] for left, right in (("train", "calibration"), ("train", "validation"), ("calibration", "validation"))):
        raise ValueError("train/calibration/validation leakage")
    clusters = json.loads((path / "clusters.json").read_text())
    for name, digest in clusters["files"].items():
        if Path(name).name != name or sha(path / name) != digest:
            raise ValueError("clustering artifact changed")
    return info


def normalized_raw(path, name, dim):
    values = np.fromfile(path / f"{name}_raw.f32", dtype="<f4").reshape(-1, dim).astype(np.float64)
    norms = np.sqrt(np.sum(values * values, axis=1))
    values /= np.where(norms > 0, norms, 1)[:, None]
    return values


def exact_truth(train, queries, k=K):
    best_ids = np.empty((len(queries), 0), dtype=np.int64)
    best_values = np.empty((len(queries), 0), dtype=np.float64)
    for start in range(0, len(train), 8192):
        block = train[start:start + 8192]
        values = np.dot(queries, block.T)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite cosine oracle")
        ids = np.broadcast_to(np.arange(start, start + len(block)), values.shape)
        values = np.concatenate((best_values, values), axis=1)
        ids = np.concatenate((best_ids, ids), axis=1)
        # Exact score ties are ordered by original database id.
        order = np.lexsort((ids, -values), axis=1)[:, :k]
        best_values = np.take_along_axis(values, order, axis=1)
        best_ids = np.take_along_axis(ids, order, axis=1)
    return best_ids, best_values


def score_predictions(predictions, train, queries, truth_ids, truth_values):
    labels = np.asarray(predictions)
    if labels.dtype.kind not in "iu" or labels.shape != (len(queries), K):
        raise ValueError("invalid prediction shape or type")
    if np.any(labels < -1) or np.any(labels >= len(train)):
        raise ValueError("prediction id outside database")
    if any(len(set(row[row >= 0].tolist())) != np.count_nonzero(row >= 0) for row in labels):
        raise ValueError("duplicate prediction id")
    present = labels >= 0
    similarities = np.einsum("qd,qkd->qk", queries, train[np.maximum(labels, 0)], optimize=False)
    similarities[~present] = -np.inf
    canonical = np.array([len(set(found) & set(truth)) / K for found, truth in zip(labels.tolist(), truth_ids.tolist(), strict=True)])
    boundary = np.sum(similarities >= truth_values[:, K - 1, None] - 1e-12, axis=1) / K
    return {"recall_at_10": float(np.mean(canonical)), "boundary_recall_at_10": float(np.mean(boundary)),
            "query_recall": canonical.tolist(), "boundary_query_recall": boundary.tolist(),
            "missing_result_count": int(np.count_nonzero(~present))}


def checked_result(result, phase, info, path, train, queries, ids, values):
    if result.get("schema_version") != 1 or result["dim"] != info["dim"] or result["database_rows"] != info["database_rows"] or result["query_rows"] != len(queries):
        raise ValueError("result does not match registered input")
    if result.get("phase", phase) != phase:
        raise ValueError("wrong result phase")
    if result["index_bytes"] <= 0 or type(result["index_bytes"]) is not int:
        raise ValueError("invalid serialized bytes")
    index_path = Path(result["index_path"])
    if index_path.resolve().parent != path.resolve():
        raise ValueError("index is outside campaign directory")
    if index_path.stat().st_size != result["index_bytes"] or sha(index_path) != result["index_sha256"]:
        raise ValueError("serialized index changed")
    configs = []
    for config_id, config in enumerate(result["configs"]):
        score = score_predictions(config["predictions"], train, queries, ids, values)
        elapsed = np.asarray(config["elapsed_ns"])
        if elapsed.dtype.kind not in "iu" or elapsed.ndim != 2 or elapsed.shape[1] != len(queries) or elapsed.shape[0] not in (1, 3, 5) or np.any(elapsed <= 0):
            raise ValueError("invalid per-query timing")
        params = {key: value for key, value in config.items() if key in ("ef", "nprobe", "rerank")}
        scanned = config.get("scanned_counts")
        if scanned is not None:
            counts = np.asarray(scanned)
            if counts.dtype.kind not in "iu" or counts.shape != (len(queries),) or np.any(counts > len(train)):
                raise ValueError("invalid scanned counts")
            nlist = json.loads((path / "clusters.json").read_text())["nlist"]
            if params.get("nprobe") == nlist and np.any(counts != len(train)):
                raise ValueError("all-probe search did not scan full database")
        configs.append({"config_id": config_id, "search": params, **score,
                        "median_ns": float(np.median(elapsed)), "p95_ns": float(np.quantile(elapsed, 0.95)),
                        "load_gate_met": config.get("load_gate_met", False)})
    return {"system": result["system"], "build": result.get("build", {key: result[key] for key in ("bits", "residual_bits") if key in result}),
            "index_bytes": result["index_bytes"], "index_sha256": result["index_sha256"],
            "build_ns": result.get("build_ns"), "peak_rss_bytes": result.get("peak_rss_bytes"), "configs": configs}


def choose(rows, target):
    eligible = [(config["median_ns"], row["index_bytes"], row_id, config["config_id"])
                for row_id, row in enumerate(rows) for config in row["configs"] if config["recall_at_10"] >= target]
    if not eligible:
        return None
    _, _, row_id, config_id = min(eligible)
    return {"row_id": row_id, "config_id": config_id}


def calibrate(args):
    path = Path(args.input)
    info = verified_inputs(path)
    train = normalized_raw(path, "train", info["dim"])
    queries = normalized_raw(path, "calibration", info["dim"])
    ids, values = exact_truth(train, queries)
    rows = []
    for filename in sorted(path.glob("*-calibration.json")):
        result = json.loads(filename.read_text())
        row = checked_result(result, "calibration", info, path, train, queries, ids, values)
        row.update({"result_path": str(filename), "result_sha256": sha(filename)})
        rows.append(row)
    if not rows:
        raise ValueError("no calibration results")
    selections = {}
    for system in sorted({row["system"] for row in rows}):
        subset_ids = [i for i, row in enumerate(rows) if row["system"] == system]
        subset = [rows[i] for i in subset_ids]
        selections[system] = {}
        for target in TARGETS:
            picked = choose(subset, target)
            if picked is not None:
                picked["row_id"] = subset_ids[picked["row_id"]]
            selections[system][str(target)] = picked
    nlist = json.loads((path / "clusters.json").read_text())["nlist"]
    anchors = []
    for row_id, row in enumerate(rows):
        eligible = [config for config in row["configs"] if config["search"].get("nprobe") == nlist]
        if not eligible:
            max_ef = max(config["search"].get("ef", 0) for config in row["configs"])
            eligible = [config for config in row["configs"] if config["search"].get("ef", 0) == max_ef]
        best = min(eligible, key=lambda config: (-config["recall_at_10"], config["median_ns"], config["config_id"]))
        anchors.append({"row_id": row_id, "config_id": best["config_id"],
                        "purpose": "all-probe maximum calibration quality" if "nprobe" in best["search"] else "maximum graph search beam"})
        if info["database_rows"] >= 100000 and row["system"] == "hms_ivf":
            capacity = [config for config in row["configs"] if config["search"] == {"nprobe": nlist, "rerank": 1024}]
            if len(capacity) != 1:
                raise ValueError("missing preregistered maximum-capacity anchor")
            if capacity[0]["config_id"] != best["config_id"]:
                anchors.append({"row_id": row_id, "config_id": capacity[0]["config_id"],
                                "purpose": "all-probe rerank1024 capacity anchor, independent of calibration recall ties"})
    write(path / "calibration-check.json", {"schema_version": 1, "inputs_sha256": sha(path / "inputs.json"),
          "checker_sha256": sha(Path(__file__)), "oracle": "float64 exact cosine of original raw train rows, score ties by database id",
          "selection": "fastest descriptive calibration median meeting strict recall target; bytes break timing ties; no validation results opened",
          "timing_claim_eligible": False, "timing_limit": "HMS native query-loop versus competitor Python binding call overhead is not matched",
          "rows": rows, "selections": selections, "quality_anchors": anchors})
    print(json.dumps(selections, indent=2), flush=True)


def validate(args):
    import competitive_ann as runner

    with runner.timing_lock():
        validate_selected(args, runner)


def validate_selected(args, runner):
    path = Path(args.input)
    info = verified_inputs(path)
    frozen_path = path / "calibration-check.json"
    frozen_hash = sha(frozen_path)
    frozen = json.loads(frozen_path.read_text())
    if frozen["inputs_sha256"] != sha(path / "inputs.json"):
        raise ValueError("frozen calibration uses different inputs")
    selected = {}
    for targets in frozen["selections"].values():
        for choice in targets.values():
            if choice is not None:
                selected.setdefault(choice["row_id"], set()).add(choice["config_id"])
    for choice in frozen["quality_anchors"]:
        selected.setdefault(choice["row_id"], set()).add(choice["config_id"])
    for row_id, config_ids in selected.items():
        row = frozen["rows"][row_id]
        result_path = Path(row["result_path"])
        if sha(result_path) != row["result_sha256"]:
            raise ValueError("calibration result changed after freeze")
        original = json.loads(result_path.read_text())
        name = result_path.name.removesuffix("-calibration.json")
        searches = [row["configs"][i]["search"] for i in sorted(config_ids)]
        completed_path = path / f"{name}-validation.json"
        completed_receipt = path / f"{name}-validation-execution.json"
        if completed_path.exists() and completed_receipt.exists():
            receipt = json.loads(completed_receipt.read_text())
            completed = json.loads(completed_path.read_text())
            index_path = Path(completed["index_path"])
            actual_searches = [{key: config[key] for key in ("ef", "nprobe", "rerank") if key in config} for config in completed["configs"]]
            if receipt["exit_code"] != 0 or completed["build"] != row["build"] or completed["system"] != row["system"] or completed["index_sha256"] != row["index_sha256"] or sha(index_path) != row["index_sha256"] or any(search not in actual_searches for search in searches):
                raise ValueError("completed validation worker differs from frozen selection")
            continue
        if row["system"] == "hms_ivf":
            if args.hms_binary is None:
                raise ValueError("selected HMS validation requires --hms-binary")
            probes = sorted({p["nprobe"] for p in searches})
            reranks = sorted({p["rerank"] for p in searches})
            command = [args.hms_binary, "--train", str(path / "train.f32"), "--queries", str(path / "validation.f32"),
                       "--centroids", str(path / "centroids.f32"), "--assignments", str(path / "assignments.u32"),
                       "--dim", str(info["dim"]), "--bits", str(row["build"]["bits"]), "--residual-bits", str(row["build"]["residual_bits"]),
                       "--nprobe", ",".join(map(str, probes)), "--rerank", ",".join(map(str, reranks)),
                       "--repeats", str(args.repeats), "--out", str(path / f"{name}-validation.json"),
                       "--index-out", str(path / f"{name}-validation.index")]
        else:
            command = runner.worker_command(row["system"], "worker", "--input", path, "--system", row["system"],
                      "--phase", "validation", "--build", json.dumps(row["build"]), "--search", json.dumps(searches),
                      "--index", original["index_path"], "--repeats", args.repeats, "--out", path / f"{name}-validation.json")
        receipt = runner.execute(command, path / f"{name}-validation-execution.json")
        if row["system"] == "hms_ivf":
            result_path = path / f"{name}-validation.json"
            candidate = json.loads(result_path.read_text())
            index_path = path / f"{name}-validation.index"
            candidate.update({"system": "hms_ivf", "phase": "validation", "build": row["build"],
                              "index_path": str(index_path), "index_sha256": sha(index_path),
                              "peak_rss_bytes": receipt["external_peak_rss_bytes"],
                              "binary_sha256": runner.sha(Path(args.hms_binary)),
                              "measurement": original["measurement"]})
            runner.write(result_path, candidate, exclusive=False)
        if sha(frozen_path) != frozen_hash:
            raise ValueError("frozen selection changed during validation")


def report(args):
    path = Path(args.input)
    info = verified_inputs(path)
    frozen_path = path / "calibration-check.json"
    frozen = json.loads(frozen_path.read_text())
    train = normalized_raw(path, "train", info["dim"])
    queries = normalized_raw(path, "validation", info["dim"])
    ids, values = exact_truth(train, queries)
    validated = {}
    for filename in sorted(path.glob("*-validation.json")):
        result = json.loads(filename.read_text())
        row = checked_result(result, "validation", info, path, train, queries, ids, values)
        row.update({"result_path": str(filename), "result_sha256": sha(filename)})
        validated[filename.name.removesuffix("-validation.json")] = row
    selected = []
    for system, targets in frozen["selections"].items():
        for target, choice in targets.items():
            if choice is None:
                selected.append({"system": system, "target": float(target), "status": "calibration_target_missed"})
                continue
            calibration = frozen["rows"][choice["row_id"]]
            config = calibration["configs"][choice["config_id"]]
            name = Path(calibration["result_path"]).name.removesuffix("-calibration.json")
            validation = validated[name]
            matches = [candidate for candidate in validation["configs"] if candidate["search"] == config["search"]]
            if len(matches) != 1 or validation["build"] != calibration["build"] or validation["index_sha256"] != calibration["index_sha256"]:
                raise ValueError("validation differs from frozen build/search/index")
            checked = matches[0]
            selected.append({"system": system, "target": float(target), "build": validation["build"], "search": checked["search"],
                             "calibration_recall": config["recall_at_10"], "validation_recall": checked["recall_at_10"],
                             "validation_boundary_recall": checked["boundary_recall_at_10"], "median_ns": checked["median_ns"],
                             "p95_ns": checked["p95_ns"], "index_bytes": validation["index_bytes"],
                             "status": "validation_target_met" if checked["recall_at_10"] >= float(target) else "validation_target_missed"})
    anchors = []
    for choice in frozen["quality_anchors"]:
        calibration = frozen["rows"][choice["row_id"]]
        config = calibration["configs"][choice["config_id"]]
        name = Path(calibration["result_path"]).name.removesuffix("-calibration.json")
        validation = validated[name]
        matches = [candidate for candidate in validation["configs"] if candidate["search"] == config["search"]]
        if len(matches) != 1 or validation["index_sha256"] != calibration["index_sha256"]:
            raise ValueError("quality anchor differs from frozen configuration")
        anchors.append({"system": calibration["system"], "build": calibration["build"], "search": config["search"],
                        "calibration_recall": config["recall_at_10"], "validation_recall": matches[0]["recall_at_10"],
                        "validation_boundary_recall": matches[0]["boundary_recall_at_10"], "index_bytes": validation["index_bytes"],
                        "purpose": choice["purpose"]})
    write(path / "independent-check.json", {"schema_version": 1, "dataset": info["dataset"], "database_rows": info["database_rows"],
          "inputs_sha256": sha(path / "inputs.json"), "frozen_calibration_sha256": sha(frozen_path), "checker_sha256": sha(Path(__file__)),
          "independent_raw_source_oracle": True, "validation_rows": len(queries), "timing_claim_eligible": False,
          "timing_limit": frozen["timing_limit"], "selected": selected, "quality_anchors": anchors, "all_validation_rows": validated})
    print(json.dumps(selected, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("calibrate", "validate", "report"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--hms-binary")
    parser.add_argument("--repeats", type=int, choices=(1, 3, 5), default=3)
    args = parser.parse_args()
    {"calibrate": calibrate, "validate": validate, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
