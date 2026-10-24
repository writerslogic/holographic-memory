# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Independently validate the experiment grid and recompute its frozen selection."""
import argparse
import json
import math
from pathlib import Path
from statistics import median


def verify(data):
    assert data["stage"] == "complete", "incomplete experiment"
    assert data["measurement"] == "development"
    assert data["rounds"] == 3 and data["query_threads"] == 1 and data["k"] == 100
    assert data["training_rows"] == 32768 and data["validation_rows"] == 1000
    assert len(data["query_affinity"]) == 1
    assert len(data["rows"]) == 309
    assert len(data["builds_s"]) == 7
    expected = set()
    ef = [64, 80, 100, 128, 160, 200, 256, 320, 400, 512, 640, 800, 1000, 1500, 2000] if data["metric"] == "ip" else [40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400, 512, 640, 800, 1000]
    rr = [200, 400, 1000] if data["metric"] == "ip" else [100, 200, 400]
    for e in ef:
        for r in rr:
            expected.add(("graph", "graph-R64", (e, r)))
    for c in [256, 1024]:
        for rank in [16, 32, 64]:
            for p in [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]:
                for r in [100, 200, 400, 800]:
                    expected.add(("lowrank", f"lowrank-C{c}-r{rank}", (p, r)))
    actual = {(r["arm"], r["build"], tuple(r["args"])) for r in data["rows"]}
    assert actual == expected
    for row in data["rows"]:
        assert 0 <= row["val_recall"] <= 1
        assert len(row["qps"]) == 3 and all(math.isfinite(q) and q > 0 for q in row["qps"])
    selected = {}
    for arm in ["graph", "lowrank"]:
        eligible = [i for i, r in enumerate(data["rows"]) if r["arm"] == arm and r["val_recall"] >= .95]
        selected[arm] = min(eligible, key=lambda i: (-median(data["rows"][i]["qps"]), data["rows"][i]["build"], data["rows"][i]["args"])) if eligible else None
    assert selected == data["selected"]
    assert all(0 <= row["before"] < 3 and 0 <= row["after"] < 3 for row in data["loads"])
    summary = {"schema_version": 1, "source": data["source"], "verified_cells": len(actual), "tuning_batches": 3 * len(actual), "selected": {arm: data["rows"][i] if i is not None else None for arm, i in selected.items()}, "official_test_read": data["official_test_read"]}
    if any(i is None for i in selected.values()):
        assert not data["official_test_read"]
        summary["verdict"] = "FAIL: no eligible development configuration"
    else:
        assert data["official_test_read"]
        graph, lowrank = (data["confirmation"][arm] for arm in ["graph", "lowrank"])
        assert len(graph["qps"]) == len(lowrank["qps"]) == 3
        ratios = [b / a for a, b in zip(graph["qps"], lowrank["qps"])]
        assert ratios == data["paired_speedups"]
        summary.update(confirmation=data["confirmation"], paired_speedups=ratios, verdict="PASS" if min(graph["recall"], lowrank["recall"]) >= .95 and min(ratios) >= 1 else "FAIL")
    assert summary["verdict"] == data["verdict"]
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify(json.loads(args.result.read_text()))
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    else:
        print(text, end="")
