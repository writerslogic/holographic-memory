#!/usr/bin/env python3
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded fresh-process comparison of small phasor-table widening thresholds."""

import datetime
import hashlib
import json
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "target/holography-phase-substrate/threshold"
EVIDENCE = ROOT / "benchmarks/results/holography_phase_substrate_v1"


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    source = (ROOT / "src/core/phase_hvec.rs").read_text()
    probe = (ROOT / "benchmarks/research/phase_substrate/probe.rs").read_text()
    assert source.count("if n <= 256 {") == 1
    result = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "serialization_scope": "this task only; host exclusivity is not asserted",
        "case": {"n": 257, "dim": 16384},
        "sample_ms": 10,
        "repeats": 5,
        "fresh_processes_per_threshold": 7,
        "probe_sha256": hashlib.sha256(probe.encode()).hexdigest(),
        "variants": {},
        "runs": [],
    }
    for threshold in [256, 512]:
        variant = source.replace("if n <= 256 {", f"if n <= {threshold} {{")
        source_path = TARGET / f"phase_{threshold}.rs"
        source_path.write_text(variant)
        driver = probe.replace('"../../../src/core/phase_hvec.rs"', json.dumps(str(source_path)))
        driver = driver.replace('"../../results/holography_phase_substrate_v1/baseline_phase_hvec.rs"', json.dumps(str(EVIDENCE / "baseline_phase_hvec.rs")))
        driver_path = TARGET / f"probe_{threshold}.rs"
        driver_path.write_text(driver)
        binary = TARGET / f"probe_{threshold}"
        command = ["rustc", "--edition=2021", "-C", "opt-level=3", "-C", "lto=true", "-C", "codegen-units=1", "-C", "strip=debuginfo", str(driver_path), "-o", str(binary)]
        subprocess.run(command, cwd=ROOT, check=True)
        result["variants"][threshold] = {"source_sha256": hashlib.sha256(variant.encode()).hexdigest(), "compile_command": command}
    for trial in range(7):
        for threshold in ([256, 512] if trial % 2 == 0 else [512, 256]):
            lines = subprocess.check_output([str(TARGET / f"probe_{threshold}"), "257", "16384", "10", "5", str(trial)], cwd=ROOT, text=True).splitlines()
            records = [json.loads(line) for line in lines]
            result["runs"].append({"threshold": threshold, **records[0], "timings": records[1:]})
    result["summary"] = []
    for threshold in [256, 512]:
        for operation in ["bundle2", "bundle8", "bundle32"]:
            records = [timing for run in result["runs"] if run["threshold"] == threshold for timing in run["timings"] if timing["operation"] == operation]
            result["summary"].append({"threshold": threshold, "operation": operation, "baseline_ns": statistics.median([ns for timing in records for ns in timing["baseline_ns"]]), "current_ns": statistics.median([ns for timing in records for ns in timing["current_ns"]]), "current_per_process_medians_ns": [statistics.median(timing["current_ns"]) for timing in records]})
    result["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    output = EVIDENCE / "threshold_diagnosis.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
