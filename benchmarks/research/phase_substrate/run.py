#!/usr/bin/env python3
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare phase-vector kernels in fresh processes, retaining every timing sample."""

import argparse
import datetime
import hashlib
import json
import platform
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "benchmarks/results/holography_phase_substrate_v1"


def command(args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def parse_case(value):
    try:
        n, dim = map(int, value.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("case must be N:DIM") from error
    if not 2 <= n <= 65536 or dim <= 0:
        raise argparse.ArgumentTypeError("case requires 2 <= N <= 65536 and positive DIM")
    return n, dim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, choices=["exploratory-concurrent", "serial-validation"])
    parser.add_argument("--sample-ms", type=float, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cold-trials", type=int, default=3)
    parser.add_argument("--case", dest="cases", type=parse_case, action="append", help="limit to N:DIM; repeat for multiple cases")
    parser.add_argument("--output", type=Path, default=EVIDENCE / "exploratory.json")
    args = parser.parse_args()
    if args.sample_ms <= 0 or args.repeats <= 0 or args.cold_trials <= 0:
        parser.error("timing and repetition counts must be positive")
    cases = args.cases or [(n, dim) for n in [2, 16, 256, 257, 65536] for dim in [64, 512, 4096, 16384]] + [(65536, 65536)]

    binary = ROOT / "target/holography-phase-substrate/probe"
    binary.parent.mkdir(parents=True, exist_ok=True)
    compile_command = [
        "rustc", "--edition=2021", "-C", "opt-level=3", "-C", "lto=true",
        "-C", "codegen-units=1", "-C", "strip=debuginfo",
        "benchmarks/research/phase_substrate/probe.rs", "-o", str(binary),
    ]
    subprocess.run(compile_command, cwd=ROOT, check=True)

    sources = [
        "src/core/phase_hvec.rs",
        "benchmarks/results/holography_phase_substrate_v1/baseline_phase_hvec.rs",
        "benchmarks/research/phase_substrate/probe.rs",
        "benchmarks/research/phase_substrate/run.py",
    ]
    metadata = {
        "label": args.label,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "head": command(["git", "rev-parse", "HEAD"]),
        "rustc": command(["rustc", "--version"]),
        "cargo": command(["cargo", "--version"]),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "compile_command": compile_command,
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources},
        "sample_ms": args.sample_ms,
        "repeats": args.repeats,
        "cold_trials": args.cold_trials,
        "cases": cases,
        "fixture_creation_timed": False,
        "warm_timings_include_result_allocation_and_drop": True,
        "cold_timings_include_result_allocation_but_not_drop": True,
        "phase_vector_storage": "unchanged Vec<u16>; two bytes per phase",
        "max_global_lookup_payload_bytes": 1_578_928,
        "concurrent_work_warning": args.label == "exploratory-concurrent",
        "serialization_scope": "this task only; host exclusivity is not asserted",
    }
    if platform.system() == "Darwin":
        metadata["hardware"] = {
            key: command(["sysctl", "-n", key])
            for key in ["machdep.cpu.brand_string", "hw.physicalcpu", "hw.memsize"]
        }

    result = {"metadata": metadata, "runs": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".partial.json")
    for trial in range(args.cold_trials):
        for n, dim in cases:
            lines = command([str(binary), str(n), str(dim), str(args.sample_ms), str(args.repeats), str(trial)]).splitlines()
            records = [json.loads(line) for line in lines]
            result["runs"].append({**records[0], "timings": records[1:]})
            partial.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Finished fresh-process trial {trial + 1}/{args.cold_trials}", flush=True)

    result["summary"] = []
    for n, dim in cases:
        runs = [run for run in result["runs"] if run["n"] == n and run["dim"] == dim]
        row = {"n": n, "dim": dim, "operations": {}}
        for operation in ["similarity", "bind", "unbind", "bundle1", "bundle2", "bundle8", "bundle32"]:
            timings = [timing for run in runs for timing in run["timings"] if timing["operation"] == operation]
            old = statistics.median([ns for timing in timings for ns in timing["baseline_ns"]])
            new = statistics.median([ns for timing in timings for ns in timing["current_ns"]])
            row["operations"][operation] = {"baseline_ns": old, "current_ns": new, "speedup": old / new}
        for operation in ["similarity", "bundle8"]:
            old = statistics.median([run[f"cold_{operation}_baseline_ns"] for run in runs])
            new = statistics.median([run[f"cold_{operation}_current_ns"] for run in runs])
            row["operations"][f"cold_{operation}"] = {"baseline_ns": old, "current_ns": new, "speedup": old / new}
        result["summary"].append(row)
    result["metadata"]["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    partial.write_text(json.dumps(result, indent=2) + "\n")
    partial.replace(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
