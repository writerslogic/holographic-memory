# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Independently check split isolation, resource caps and paired bootstrap results."""

import argparse
import itertools
import json
import math
import statistics
import zlib
from pathlib import Path


def mix(a, b):
    mask = (1 << 64) - 1
    z = (a * 0x9E3779B97F4A7C15 + b) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return z ^ (z >> 31)


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)


def interval(values, indices):
    draws = sorted(statistics.mean(values[i] for i in sample) for sample in indices)
    return [draws[49], draws[1949]]


def check(path):
    root = Path(__file__).resolve().parents[2]
    report = json.loads(path.read_text())
    assert report["schema_version"] == 1
    for key, source in (("operator", "src/core/phase_path.rs"),
                        ("harness", "src/bin/phase-path-bench.rs"),
                        ("geometry", "src/core/manifold.rs")):
        assert report["source_crc32"][key] == zlib.crc32((root / source).read_bytes()), source
    args = report["args"]
    calibration_seeds = list(range(args["seed"], args["seed"] + args["train_seeds"]))
    heldout_seeds = list(range(args["seed"] + 10000, args["seed"] + 10000 + args["seeds"]))
    assert set(calibration_seeds).isdisjoint(heldout_seeds)
    expected_conditions = set(itertools.product(("coincident", "paths"), args["depths"], args["facts"], args["noise"]))
    assert len(report["rows"]) == len(expected_conditions)
    grid = [{"distance": d, "spin": s} for d in (0.0, 0.5, 2.0) for s in (0, 1, 4)]
    names = ["commutative_sanity", "positional_flat", "positional_curved", "monomial_flat",
             "monomial_metric", "monomial_connection", "monomial_both"]
    indices = [[mix(args["seed"] ^ 0xF02, sample * args["seeds"] + j) % args["seeds"]
                for j in range(args["seeds"])] for sample in range(2000)]
    passes = []
    ordered_gains = []
    for row in report["rows"]:
        condition = (row["layout"], row["depth"], row["facts"], row["coordinate_noise"])
        assert condition in expected_conditions
        expected_conditions.remove(condition)
        arms = row["arms"]
        selections = row["calibration"]
        assert [arm["name"] for arm in arms] == names
        assert [selection["arm"] for selection in selections] == names
        best_means = []
        for arm, selection in zip(arms, selections):
            trials = selection["trials"]
            assert [trial["setting"] for trial in trials] == grid
            for trial in trials:
                assert [run["seed"] for run in trial["runs"]] == calibration_seeds
                close(trial["mean_accuracy"], statistics.mean(run["accuracy"] for run in trial["runs"]))
            best = max(trials, key=lambda trial: trial["mean_accuracy"])
            best_means.append(best["mean_accuracy"])
            assert arm["selected"] == best["setting"]
            assert [run["seed"] for run in arm["runs"]] == heldout_seeds
            close(arm["accuracy"], statistics.mean(run["accuracy"] for run in arm["runs"]))
            for run in list(itertools.chain.from_iterable(t["runs"] for t in trials)) + arm["runs"]:
                assert run["queries"] == min(row["facts"], args["queries"])
                assert 0 <= run["correct"] + run["reverse_errors"] <= run["queries"]
                close(run["accuracy"], run["correct"] / run["queries"])
                assert 32 <= run["dimensions"] <= 4096
                assert 0 < run["model_bytes"] <= args["budget"]
                assert run["witness_bytes"] > 0
                assert all(math.isfinite(run[key]) and run[key] >= 0
                           for key in ("query_p95_us", "build_ms"))
        baseline = 2 if best_means[2] > best_means[1] else 1
        assert row["baseline"] == names[baseline]
        for i in range(args["seeds"]):
            assert len({arm["runs"][i]["fixture_crc32"] for arm in arms}) == 1
            assert len({arm["runs"][i]["witness_bytes"] for arm in arms}) == 1
        for key, a, b in (("delta_vs_positional", 6, baseline),
                          ("metric_delta_given_connection", 6, 5),
                          ("connection_delta_given_metric", 6, 4)):
            values = [ra["accuracy"] - rb["accuracy"] for ra, rb in zip(arms[a]["runs"], arms[b]["runs"])]
            close(row[key]["mean"], statistics.mean(values))
            for actual, expected in zip(row[key]["ci95"], interval(values, indices)):
                close(actual, expected)
        controls = row["exact_control"]
        assert [run["seed"] for run in controls] == heldout_seeds
        for run in controls:
            assert 0 <= run["correct"] <= run["queries"] == min(row["facts"], args["queries"])
            close(run["accuracy"], run["correct"] / run["queries"])
        if min(row[key]["ci95"][0] for key in ("delta_vs_positional", "metric_delta_given_connection", "connection_delta_given_metric")) > 0:
            passes.append(condition)
        if arms[baseline]["accuracy"] > arms[0]["accuracy"]:
            ordered_gains.append(condition)
    assert not expected_conditions
    return {"conditions": len(report["rows"]), "heldout_seeds": args["seeds"],
            "mechanism_passes": passes, "positional_beats_commutative_conditions": len(ordered_gains),
            "paired_intervals_recomputed": 3 * len(report["rows"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    print(json.dumps(check(parser.parse_args().report), indent=2))
