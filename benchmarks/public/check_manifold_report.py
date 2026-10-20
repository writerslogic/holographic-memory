# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check source fingerprints, split isolation, selection and paired result accounting."""

import argparse
import json
import math
import statistics
import zlib
from pathlib import Path


def check(path):
    root = Path(__file__).resolve().parents[2]
    report = json.loads(path.read_text())
    for name, relative in (
        ("kernel", "src/core/manifold.rs"),
        ("harness", "src/bin/manifold-memory-bench.rs"),
    ):
        assert report["source_crc32"][name] == zlib.crc32((root / relative).read_bytes()), (
            "source fingerprint mismatch", relative
        )
    args = report["args"]
    calibration = set(range(args["seed"], args["seed"] + args["train_seeds"]))
    heldout = list(range(args["seed"] + 10_000, args["seed"] + 10_000 + args["seeds"]))
    assert calibration.isdisjoint(heldout)
    assert len(report["rows"]) == 2 * len(args["facts"]) * len(args["noise"])
    passes = []
    max_residual = 0.0
    for row in report["rows"]:
        arms = row["arms"]
        assert len(arms) == len(row["calibration"]) == 4
        for arm, selection in zip(arms, row["calibration"]):
            assert arm["name"] == selection["arm"]
            trials = selection["trials"]
            assert len(trials) == 2 * len(args["bandwidths"])
            for trial in trials:
                assert len(trial["seed_accuracies"]) == args["train_seeds"]
                assert math.isclose(trial["mean_accuracy"], statistics.mean(trial["seed_accuracies"]), abs_tol=1e-12)
            best = max(trials, key=lambda trial: trial["mean_accuracy"])
            assert arm["selected"] == best["setting"]
            assert [run["seed"] for run in arm["runs"]] == heldout
            assert math.isclose(arm["accuracy"], statistics.mean(run["accuracy"] for run in arm["runs"]), abs_tol=1e-12)
            for run in arm["runs"]:
                assert run["queries"] == min(row["facts"], args["queries"])
                assert 0 <= run["correct"] <= run["queries"]
                assert run["accuracy"] == run["correct"] / run["queries"]
                assert math.isfinite(run["relative_residual"]) and run["relative_residual"] <= 1e-6
                max_residual = max(max_residual, run["relative_residual"])
        for index in range(args["seeds"]):
            assert len({arm["runs"][index]["persistent_bytes"] for arm in arms}) == 1
        for arm in arms:
            delta = arm["accuracy"] - arms[0]["accuracy"]
            assert math.isclose(delta, arm["delta_vs_flat"], abs_tol=1e-12)
            lo, hi = arm["paired_delta_ci95"]
            assert -1 <= lo <= hi <= 1
        metric = row["metric_delta_given_curved_transport"]
        transport = row["transport_delta_given_curved_metric"]
        assert math.isclose(metric["mean"], arms[3]["accuracy"] - arms[2]["accuracy"], abs_tol=1e-12)
        assert math.isclose(transport["mean"], arms[3]["accuracy"] - arms[1]["accuracy"], abs_tol=1e-12)
        if min(arms[3]["paired_delta_ci95"][0], metric["ci95"][0], transport["ci95"][0]) > 0:
            passes.append({"layout": row["layout"], "facts": row["facts"], "noise": row["coordinate_noise_std"]})
    return {"conditions": len(report["rows"]), "heldout_seeds": args["seeds"],
            "mechanism_passes": passes, "max_relative_residual": max_residual}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    print(json.dumps(check(parser.parse_args().report), indent=2))
