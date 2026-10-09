# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///
"""Measure complete native reader calls on one preregistered, completed dev input."""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest(path)}


def save(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--functional-only", action="store_true")
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    registration = json.loads(args.preregistration.read_text())
    control = registration["control"]
    chosen = control["initial_request"]
    chunk = next(c for c in control["completed_real_dev_chunks"] if c["chunk"] == chosen["chunk"])
    for name in ("input", "manifest", "native_output"):
        source = Path(chunk[name]["path"])
        if digest(source) != chunk[name]["sha256"]:
            raise ValueError("preregistered input evidence fingerprint differs")
    jobs = json.loads(Path(chunk["manifest"]["path"]).read_text())["jobs"]
    position = next(i for i, job in enumerate(jobs) if job["cache_key"] == chosen["cache_key"])
    prompt = json.loads(Path(chunk["input"]["path"]).read_text())[position]
    expected = json.loads(Path(chunk["native_output"]["path"]).read_text())[position]
    if hashlib.sha256(prompt.encode()).hexdigest() != chosen["input_sha256"]:
        raise ValueError("preregistered prompt fingerprint differs")
    input_path = directory / "input.json"
    input_path.write_text(json.dumps([prompt], ensure_ascii=False) + "\n")
    binaries = {"control": Path(control["binary"]["path"]), "candidate": args.candidate.resolve()}
    if digest(binaries["control"]) != control["binary"]["sha256"]:
        raise ValueError("frozen binary fingerprint differs")
    report = {"schema_version": 1, "complete": False, "preregistration": artifact(args.preregistration),
              "producer": artifact(Path(__file__)), "input": artifact(input_path),
              "binaries": {k: artifact(p) for k, p in binaries.items()}, "request": chosen,
              "expected_text_sha256": hashlib.sha256(expected.encode()).hexdigest(),
              "warmups_per_arm": 1, "rounds": 11, "load_threshold": 3,
              "pilot_only": args.pilot_only or args.functional_only,
              "timing_claim": not args.functional_only, "functional_only": args.functional_only,
              "lock_max_seconds": 840, "runs": [], "failure": None}
    report_path = directory / "report.json"
    save(report_path, report)
    acquired_at = time.monotonic()
    def run(arm, round_index, warmup):
        if time.monotonic() - acquired_at > 720:
            raise RuntimeError("lock time budget leaves no safe next complete call")
        load = os.getloadavg()
        gate_deadline = min(time.monotonic() + 120, acquired_at + 720)
        while not args.functional_only and load[0] >= 3 and time.monotonic() < gate_deadline:
            time.sleep(5)
            load = os.getloadavg()
        if not args.functional_only and load[0] >= 3:
            raise RuntimeError("load gate unmet; no timing call started")
        prefix = directory / f"{'warmup' if warmup else 'round'}_{round_index:02d}_{arm}"
        out = prefix.with_suffix(".output.json")
        command = [str(binaries[arm]), "lme-model", "--stage", "chat", "--model",
                   control["model_directory"], "--revision", control["revision"], "--batch", "1",
                   "--input", str(input_path), "--out", str(out)]
        if not args.functional_only:
            command.extend(["--max-load", "3", "--max-wait-secs", "0"])
        started = time.monotonic()
        result = subprocess.run(command, capture_output=True, timeout=120, check=False)
        wall = time.monotonic() - started
        stdout, stderr = prefix.with_suffix(".stdout.log"), prefix.with_suffix(".stderr.log")
        stdout.write_bytes(result.stdout)
        stderr.write_bytes(result.stderr)
        telemetry = [json.loads(line) for line in result.stderr.splitlines() if line.startswith(b"{")]
        if result.returncode != 0 or len(telemetry) != 1 or not out.is_file():
            raise RuntimeError("native process did not produce one successful timing record")
        native = telemetry[0]
        if args.functional_only:
            native.update({"preliminary": True, "timing_claim": False})
            stderr.write_bytes(b"\n".join(json.dumps(native, allow_nan=False).encode()
                                          if line.startswith(b"{") else line
                                          for line in result.stderr.splitlines()) + b"\n")
        prediction = json.loads(out.read_text())
        gated = native["load_gate_met"] is True and native["load_at_gate"] < 3
        entry = {"arm": arm, "round": round_index, "warmup": warmup, "argv": command,
                 "wall_seconds": wall, "native": native, "load_before": load,
                 "load_gate_met": gated, "preliminary": not gated,
                 "output": artifact(out), "stdout": artifact(stdout), "stderr": artifact(stderr),
                 "same_text": prediction == [expected]}
        report["runs"].append(entry)
        save(report_path, report)
        print(json.dumps({"arm": arm, "round": round_index, "warmup": warmup,
                          "run_seconds": native["run_secs"], "same_text": entry["same_text"],
                          "load_gate_met": gated}), flush=True)
        if not gated and not args.functional_only:
            raise RuntimeError("native load gate unmet; record is preliminary")
        if not entry["same_text"]:
            raise RuntimeError("generated text drift; candidate cannot enter frozen weak arm")
    try:
        for arm in ("control", "candidate"):
            run(arm, 0, True)
        if args.pilot_only or args.functional_only:
            report["lock_elapsed_seconds"] = time.monotonic() - acquired_at
            return
        for round_index in range(11):
            order = ("control", "candidate") if round_index % 2 == 0 else ("candidate", "control")
            for arm in order:
                run(arm, round_index, False)
        elapsed = time.monotonic() - acquired_at
        if elapsed >= 840:
            raise RuntimeError("timing lock duration limit exceeded")
        report["complete"] = True
        report["lock_elapsed_seconds"] = elapsed
        report["median_run_seconds"] = {
            arm: statistics.median(r["native"]["run_secs"] for r in report["runs"]
                                   if not r["warmup"] and r["arm"] == arm) for arm in binaries}
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        report["failure"] = str(error)
        report["lock_elapsed_seconds"] = time.monotonic() - acquired_at
        raise
    finally:
        save(report_path, report)


if __name__ == "__main__":
    main()
