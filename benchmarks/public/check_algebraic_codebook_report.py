# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Check budgets, coupled fixtures, bootstrap evidence and explicit decoding oracles."""

import argparse
import itertools
import json
import math
import statistics
import zlib
from pathlib import Path

from check_phase_path_report import close, mix


KINDS = ("unpacked_random", "packed_random", "regenerated_random", "walsh")
CORRUPTIONS = ("clean", "phase_5_percent", "erasure_10_percent")


def interval(values, samples):
    draws = sorted(sum(values[i] for i in sample) / len(values) for sample in samples)
    return [draws[49], draws[1949]]


def records(seed, facts, labels):
    stride = (mix(seed ^ 0xA01, 0) & 4095) | 1
    offset = mix(seed ^ 0xA02, 0) & 4095
    return [((stride * i + offset) & 4095, mix(seed ^ 0xA03, i) % labels)
            for i in range(facts)]


def fixture_crc(seed, facts, labels):
    data = b"".join(key.to_bytes(2, "little") + bytes([value])
                    for key, value in records(seed, facts, labels))
    return zlib.crc32(data)


def memory(kind, dim, labels, facts, layout):
    result = {
        "model_metadata": layout["model_metadata"],
        "field": dim * 16,
        "key_buffers": (28 * dim + 1) // 2,
        "value_buffers": {"unpacked_random": 2 * dim * labels,
                          "packed_random": (dim * labels + 1) // 2}.get(kind, 0),
        "witnesses": facts * layout["witness"],
        "workspace_metadata": layout["workspace_metadata"],
        "cue": dim * 2,
        "raw": dim * 16,
        "scores": labels * 8,
    }
    result["model_bytes"] = sum(result[k] for k in
                                ("model_metadata", "field", "key_buffers", "value_buffers", "witnesses"))
    result["query_workspace_bytes"] = sum(result[k] for k in
                                          ("workspace_metadata", "cue", "raw", "scores"))
    result["total_bytes"] = result["model_bytes"] + result["query_workspace_bytes"]
    return result


def explicit_decode_oracle(row, args):
    seed = args["seed"]
    dim, labels, facts = args["dim"], row["labels"], row["facts"]
    data = records(seed, facts, labels)
    count = min(facts, args["queries"])
    shift = mix(seed ^ 0xA04, 0) % facts
    queries = [(shift + i * facts // count) % facts for i in range(count)]
    lut = [complex(math.cos(math.tau * p / 16), math.sin(math.tau * p / 16))
           for p in range(16)]

    def phase(seed, row, d):
        return mix(mix(seed ^ 0xC0DE, row), d) & 15

    keys = [[phase(seed ^ 0xB02, r, d) for d in range(dim)] for r in range(28)]

    def key_phase(key, d):
        return (keys[key & 15][d] + keys[16 + ((key >> 4) & 7)][(d + 1) % dim]
                + keys[16 + ((key >> 7) & 7)][(d + 2) % dim]
                + keys[24 + (key >> 10)][d]) & 15

    checked = 0
    for arm in (row["arms"][0], row["arms"][3]):
        walsh = arm["kind"] == "walsh"
        values = [[8 * ((label & (d % labels)).bit_count() % 2) if walsh
                   else phase(seed ^ 0xB01, label, d) for d in range(dim)]
                  for label in range(labels)]
        field = [sum(lut[(key_phase(key, d) + values[value][d]) & 15]
                     for key, value in data) for d in range(dim)]
        predicted = []
        for index in queries:
            key = data[index][0]
            raw = [z * lut[key_phase(key, d)].conjugate() for d, z in enumerate(field)]
            scores = [sum((z * lut[p].conjugate()).real for z, p in zip(raw, code))
                      for code in values]
            predicted.append(max(range(labels), key=lambda label: scores[label]))
        run = arm["runs"][0]
        assert run["correct"] == sum(p == data[i][1] for p, i in zip(predicted, queries))
        assert run["prediction_crc32"] == zlib.crc32(b"".join(p.to_bytes(2, "little") for p in predicted))
        checked += 1
    return checked


def check(path):
    root = Path(__file__).resolve().parents[2]
    report = json.loads(path.read_text())
    assert report["schema_version"] == 1 and report["experiment"] == "implicit-algebraic-codebook"
    for key, source in (("codebook", "src/core/algebraic_codebook.rs"),
                        ("harness", "src/bin/algebraic-codebook-bench.rs"),
                        ("transform", "src/core/qgraph/kernels.rs"),
                        ("hash", "src/core/entangled.rs")):
        assert report["source_crc32"][key] == zlib.crc32((root / source).read_bytes()), source
    args, layout = report["args"], report["layout"]
    seeds = list(range(args["seed"], args["seed"] + args["seeds"]))
    for previous in ("manifold_curvature_synthetic.json", "phase_path_synthetic.json"):
        previous_path = root / "benchmarks/results" / previous
        if previous_path.exists():
            old = json.loads(previous_path.read_text())["args"]
            old_seeds = set(range(old["seed"], old["seed"] + old["train_seeds"]))
            old_seeds.update(range(old["seed"] + 10000, old["seed"] + 10000 + old["seeds"]))
            assert set(seeds).isdisjoint(old_seeds), previous
    samples = {tag: [[mix(args["seed"] ^ tag, i * args["seeds"] + j) % args["seeds"]
                     for j in range(args["seeds"])] for i in range(2000)]
               for tag in (0xD01, 0xD02)}
    conditions = set(itertools.product(("fixed_dimension", "fixed_total_budget"),
                                       args["labels"], CORRUPTIONS, args["facts"]))
    assert len(report["rows"]) == len(conditions)
    secondary_passes, by_condition, oracle_runs = [], {}, 0
    for row in report["rows"]:
        mode, labels, name, facts = row["mode"], row["labels"], row["corruption"]["name"], row["facts"]
        key = mode, labels, name, facts
        assert key in conditions
        conditions.remove(key)
        by_condition[key] = row
        expected_corruption = {"name": name, "phase_replacement": 0.05 if name == "phase_5_percent" else 0.0,
                               "erasure": 0.1 if name == "erasure_10_percent" else 0.0}
        assert row["corruption"] == expected_corruption
        assert [arm["kind"] for arm in row["arms"]] == list(KINDS)
        for arm in row["arms"]:
            kind = arm["kind"]
            assert [run["seed"] for run in arm["runs"]] == seeds
            accuracies = [run["accuracy"] for run in arm["runs"]]
            close(arm["accuracy"], statistics.mean(accuracies))
            for actual, expected in zip(arm["accuracy_ci95"], interval(accuracies, samples[0xD01])):
                close(actual, expected)
            for run in arm["runs"]:
                dim = run["dimensions"]
                assert 1 <= dim <= 8192 and (kind != "walsh" or dim % labels == 0)
                assert run["memory"] == memory(kind, dim, labels, facts, layout)
                if mode == "fixed_dimension":
                    assert dim == args["dim"]
                else:
                    assert run["memory"]["total_bytes"] <= args["budget"]
                    step = labels if kind == "walsh" else 1
                    assert dim + step > 8192 or memory(kind, dim + step, labels, facts, layout)["total_bytes"] > args["budget"]
                assert run["fixture_crc32"] == fixture_crc(run["seed"], facts, labels)
                assert 0 <= run["correct"] <= run["queries"] == min(facts, args["queries"])
                close(run["accuracy"], run["correct"] / run["queries"])
                assert all(math.isfinite(run[k]) and run[k] >= 0 for k in
                           ("query_p95_us", "build_ms", "update_roundtrip_us", "update_max_abs_error"))
                assert run["query_p95_us"] > 0 and run["update_max_abs_error"] < 1e-9
        for i in range(args["seeds"]):
            runs = [arm["runs"][i] for arm in row["arms"]]
            assert len({run["fixture_crc32"] for run in runs}) == 1
            if mode == "fixed_dimension":
                assert len({run["trace_crc32"] for run in runs[:3]}) == 1
                assert len({run["prediction_crc32"] for run in runs[:3]}) == 1
        assert [c["control"] for c in row["contrasts"]] == list(KINDS[:3])
        for i, contrast in enumerate(row["contrasts"]):
            candidate, control = row["arms"][3]["runs"], row["arms"][i]["runs"]
            differences = [a["accuracy"] - b["accuracy"] for a, b in zip(candidate, control)]
            ratios = [a["query_p95_us"] / b["query_p95_us"] for a, b in zip(candidate, control)]
            for stat, values in (("accuracy_delta", differences), ("p95_ratio", ratios)):
                close(contrast[stat], statistics.mean(values))
                for actual, expected in zip(contrast[stat + "_ci95"], interval(values, samples[0xD02])):
                    close(actual, expected)
        if mode == "fixed_total_budget" and row["arms"][3]["accuracy_ci95"][0] >= 0.95 and all(
                c["accuracy_delta_ci95"][0] >= -0.005 and c["p95_ratio_ci95"][1] <= 0.5 for c in row["contrasts"]):
            secondary_passes.append(key)
        controls = row["exact_control"]
        assert [r["seed"] for r in controls] == seeds
        for control, candidate in zip(controls, row["arms"][3]["runs"]):
            dim = candidate["dimensions"]
            payload = layout["packed_codes"] + 8 * dim + layout["value_codebook"] + layout["vec"] + facts * layout["witness"]
            workspace = layout["vec"] + 2 * dim
            assert control["dimensions"] == dim and control["payload_bytes"] == payload
            assert control["query_workspace_bytes"] == workspace and control["total_bytes"] == payload + workspace
            assert 0 <= control["correct"] <= control["queries"] == min(facts, args["queries"])
            close(control["accuracy"], control["correct"] / control["queries"])
            assert math.isfinite(control["query_p95_us"]) and control["query_p95_us"] > 0
        if mode == "fixed_dimension" and name == "clean" and facts == args["facts"][0] and facts <= 16:
            oracle_runs += explicit_decode_oracle(row, args)
    assert not conditions
    capacities, primary_passes = [], []
    for labels, corruption in itertools.product(args["labels"], CORRUPTIONS):
        capacity = {}
        for i, kind in enumerate(KINDS):
            last = 0
            for facts in args["facts"]:
                row = by_condition["fixed_total_budget", labels, corruption, facts]
                if row["arms"][i]["accuracy_ci95"][0] < 0.95:
                    break
                last = facts
            capacity[kind] = last
        candidate_load = capacity["walsh"]
        eligible = [(i, kind) for i, kind in enumerate(KINDS[:3]) if capacity[kind] > 0]
        latency_passes = True
        if candidate_load and eligible:
            candidate_runs = by_condition["fixed_total_budget", labels, corruption, candidate_load]["arms"][3]["runs"]
            for i, kind in eligible:
                control_runs = by_condition["fixed_total_budget", labels, corruption, capacity[kind]]["arms"][i]["runs"]
                ratios = [a["query_p95_us"] / b["query_p95_us"] for a, b in zip(candidate_runs, control_runs)]
                latency_passes &= interval(ratios, samples[0xD02])[1] <= 1.0
        if eligible and candidate_load >= 2 * max(capacity[kind] for _, kind in eligible) and latency_passes:
            primary_passes.append((labels, corruption))
        capacities.append({"labels": labels, "corruption": corruption, "lower_ci_95_capacity": capacity})
    return {"conditions": len(report["rows"]), "seeds": args["seeds"],
            "bootstrap_intervals_recomputed": 10 * len(report["rows"]),
            "explicit_score_oracle_runs": oracle_runs,
            "primary_capacity_passes": primary_passes, "secondary_decoder_passes": secondary_passes,
            "capacities": capacities}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    print(json.dumps(check(parser.parse_args().report), indent=2))
