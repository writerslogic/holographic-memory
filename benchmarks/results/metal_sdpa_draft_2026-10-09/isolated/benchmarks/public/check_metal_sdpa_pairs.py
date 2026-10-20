# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///
"""Independently check real-input, full-reader paired timing evidence."""
import argparse
import copy
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def checked_bytes(item):
    data = Path(item["path"]).read_bytes()
    require(len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"],
            "artifact byte or fingerprint mismatch")
    return data


def check(report, allow_incomplete):
    require(report["schema_version"] == 1 and report["rounds"] == 11
            and report["load_threshold"] == 3 and report["warmups_per_arm"] == 1,
            "timing condition changed")
    require(report["complete"] or allow_incomplete, "timing study incomplete")
    registration = json.loads(checked_bytes(report["preregistration"]))
    checked_bytes(report["producer"])
    settings = registration["control"]
    require(report["request"] == settings["initial_request"], "selected request changed")
    require(report["binaries"]["control"]["sha256"] == settings["binary"]["sha256"],
            "control fingerprint changed")
    for binary in report["binaries"].values():
        checked_bytes(binary)
    request = report["request"]
    chunk = next(c for c in settings["completed_real_dev_chunks"] if c["chunk"] == request["chunk"])
    for name in ("input", "manifest", "native_output"):
        record = chunk[name]
        data = Path(record["path"]).read_bytes()
        require(len(data) == record["size"] and hashlib.sha256(data).hexdigest() == record["sha256"],
                "registered source evidence differs")
    jobs = json.loads(Path(chunk["manifest"]["path"]).read_text())["jobs"]
    index = next(i for i, job in enumerate(jobs) if job["cache_key"] == request["cache_key"])
    prompt = json.loads(Path(chunk["input"]["path"]).read_text())[index]
    expected = json.loads(Path(chunk["native_output"]["path"]).read_text())[index]
    require(json.loads(checked_bytes(report["input"])) == [prompt]
            and hashlib.sha256(prompt.encode()).hexdigest() == request["input_sha256"],
            "timed input is not the registered completed dev input")
    require(report["expected_text_sha256"] == hashlib.sha256(expected.encode()).hexdigest(),
            "reference prediction fingerprint changed")
    expected_order = [(True, 0, "control"), (True, 0, "candidate")]
    for round_index in range(11):
        order = ("control", "candidate") if round_index % 2 == 0 else ("candidate", "control")
        expected_order.extend((False, round_index, arm) for arm in order)
    runs = report["runs"]
    require(len(runs) <= len(expected_order) and (not report["complete"] or len(runs) == 24),
            "paired round coverage differs")
    for record, condition in zip(runs, expected_order, strict=False):
        require((record["warmup"], record["round"], record["arm"]) == condition,
                "warmup or alternating round order differs")
        output = json.loads(checked_bytes(record["output"]))
        checked_bytes(record["stdout"])
        lines = checked_bytes(record["stderr"]).splitlines()
        telemetry = [json.loads(line) for line in lines if line.startswith(b"{")]
        require(telemetry == [record["native"]], "native timing record changed")
        native = record["native"]
        for name in ("run_secs", "load_secs", "secs"):
            require(type(native[name]) in (int, float) and math.isfinite(native[name])
                    and native[name] >= 0, "invalid measured seconds")
        require(native["run_secs"] > 0
                and math.isclose(native["secs"], native["run_secs"] + native["load_secs"], abs_tol=1e-8),
                "complete loop interval inconsistent")
        require(type(record["wall_seconds"]) in (int, float) and math.isfinite(record["wall_seconds"])
                and record["wall_seconds"] >= native["secs"], "invalid wall timer interval")
        require(native["items"] == 1 and native["batch"] == 1 and native["stage"] == "chat"
                and native["device"].startswith("Metal("), "native query condition differs")
        gated = native["load_gate_met"] is True and native["load_at_gate"] < 3
        require(record["load_gate_met"] == gated and record["preliminary"] == (not gated)
                and (report.get("functional_only") or record["load_before"][0] < 3),
                "gate label inconsistent")
        if report.get("functional_only"):
            require(report["timing_claim"] is False and native["preliminary"] is True
                    and native["timing_claim"] is False and not report["complete"],
                    "functional diagnostic asserts a timing claim")
        require(record["same_text"] == (output == [expected]), "prediction agreement changed")
        command = record["argv"]
        require(command[0] == report["binaries"][record["arm"]]["path"]
                and command[1:5] == ["lme-model", "--stage", "chat", "--model"],
                "native command differs")
        pairs = dict(zip(command[4::2], command[5::2], strict=True))
        expected_pairs = {"--model": settings["model_directory"], "--revision": settings["revision"],
                          "--batch": "1", "--input": report["input"]["path"],
                          "--out": record["output"]["path"]}
        if not report.get("functional_only"):
            expected_pairs.update({"--max-load": "3", "--max-wait-secs": "0"})
        require(pairs == expected_pairs,
                "model, cap, batch, gate or input invocation differs")
    require(math.isfinite(report["lock_elapsed_seconds"])
            and 0 <= report["lock_elapsed_seconds"] < report["lock_max_seconds"] <= 840,
            "timing lock hold exceeded cap")
    summary = {"complete": report["complete"], "validated_runs": len(runs),
               "same_text_runs": sum(r["same_text"] for r in runs), "speed_kill_passed": False,
               "full_arm_equivalence_established": False}
    if report["complete"]:
        require(report["failure"] is None and all(r["same_text"] and r["load_gate_met"] for r in runs),
                "complete study has a failed condition")
        values = {arm: [r["native"]["run_secs"] for r in runs if not r["warmup"] and r["arm"] == arm]
                  for arm in ("control", "candidate")}
        medians = {arm: statistics.median(seconds) for arm, seconds in values.items()}
        require(report["median_run_seconds"] == medians, "median prediction differs")
        deltas = [math.log(a / b) for a, b in zip(values["control"], values["candidate"], strict=True)]
        generator = random.Random(20261009)
        samples = sorted(statistics.mean(deltas[generator.randrange(11)] for _ in range(11))
                         for _ in range(2000))
        lower, upper = samples[49], samples[1950]
        summary.update({"median_run_seconds": medians,
                        "paired_mean_speed_ratio": math.exp(statistics.mean(deltas)),
                        "paired_speed_ratio_ci95": [math.exp(lower), math.exp(upper)],
                        "speed_kill_passed": lower > 0,
                        "scope": "one preregistered real dev query; full-arm equivalence unestablished"})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--tamper-tests", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    summary = check(report, args.allow_incomplete)
    if args.tamper_tests:
        def mutate(field, value):
            changed = copy.deepcopy(report)
            changed["runs"][0][field] = value
            return changed
        corruptions = [mutate("wall_seconds", -1), mutate("same_text", not report["runs"][0]["same_text"])]
        changed = copy.deepcopy(report)
        changed["input"]["bytes"] += 1
        corruptions.append(changed)
        changed = copy.deepcopy(report)
        changed["binaries"]["control"]["sha256"] = "0" * 64
        corruptions.append(changed)
        for corruption in corruptions:
            try:
                check(corruption, args.allow_incomplete)
            except (ValueError, KeyError, TypeError):
                continue
            raise ValueError("tamper was accepted")
        summary["tamper_tests_rejected"] = len(corruptions)
    summary["checker_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary["report_sha256"] = hashlib.sha256(args.report.read_bytes()).hexdigest()
    print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()
