# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///
"""Suspend the frozen worker for two short checks, always resuming its existing state."""
import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/Volumes/A/holographic-memory")
DRAFT = Path("/Volumes/A/hms-metal-sdpa")
PARENT = 62480
FROZEN = "/Volumes/A/.hms-target-gpubatch/release/public-bench"
REGISTRATION = DRAFT / "benchmarks/results/preregistration_metal_sdpa_2026-10-09.json"
JOURNAL = ROOT / "benchmarks/results/weak_reader_suspension_pilot_2026-10-09.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def save(value):
    pending = JOURNAL.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(JOURNAL)


def main():
    global JOURNAL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--functional-only", action="store_true")
    args = parser.parse_args()
    if args.functional_only:
        JOURNAL = ROOT / "benchmarks/results/weak_reader_suspension_functional_2026-10-09.json"
    active = subprocess.check_output(["ps", "-axo", "pid,ppid,comm"], text=True).splitlines()
    children = []
    for line in active[1:]:
        pid, parent, command = line.strip().split(None, 2)
        if int(parent) == PARENT and command == FROZEN:
            children.append(int(pid))
    if len(children) != 1:
        raise ValueError("expected exactly one active frozen native child")
    if any("rustc" in line or "cargo" in line for line in active):
        raise ValueError("a build remains active; reader was not suspended")
    with Path("/Volumes/A/.hms-target/timing.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    if JOURNAL.exists():
        raise ValueError("pilot journal already exists")
    journal = {"schema_version": 1, "status": "preparing", "parent_pid": PARENT,
               "native_pid": children[0], "started_at_utc": now(), "resumed_at_utc": None,
               "frozen_binary_sha256": hashlib.sha256(Path(FROZEN).read_bytes()).hexdigest(),
               "reason": "real-dev output and runtime pilots of isolated Metal attention candidate",
               "timing_claim": False, "frozen_prompt_or_generation_parameters_changed": False,
               "checks": []}
    save(journal)
    stopped = []
    began = time.monotonic()
    try:
        for pid in (PARENT, children[0]):
            os.kill(pid, signal.SIGSTOP)
            stopped.append(pid)
        journal["status"] = "suspended_waiting_for_load_gate"
        save(journal)
        deadline = time.monotonic() + 180
        while not args.functional_only and os.getloadavg()[0] >= 3 and time.monotonic() < deadline:
            print(json.dumps({"waiting_for_load_gate": os.getloadavg()[0]}), flush=True)
            time.sleep(10)
        if not args.functional_only and os.getloadavg()[0] >= 3:
            raise RuntimeError("load gate remained unmet; no GPU check started")
        for name, candidate in (
            ("unmodified_control", "/Volumes/A/.hms-target-metal-sdpa/release/public-bench-control"),
            ("sdpa", "/Volumes/A/.hms-target-metal-sdpa/release/public-bench-sdpa"),
        ):
            prefix = "functional_" if args.functional_only else "pilot_"
            directory = DRAFT / "target/sdpa-validation" / (prefix + name)
            command = ["/Volumes/A/.hms-target/timed.sh", "uv", "run", "--locked", "--script",
                       str(DRAFT / "benchmarks/public/metal_sdpa_pairs.py"), "--preregistration",
                       str(REGISTRATION), "--candidate", candidate, "--directory", str(directory),
                       "--pilot-only"]
            if args.functional_only:
                command.append("--functional-only")
            journal["status"] = "running_" + name
            save(journal)
            log = DRAFT / "target/sdpa-validation" / (prefix + name + ".log")
            with log.open("wb") as output:
                process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                           start_new_session=True, cwd=DRAFT)
                try:
                    returncode = process.wait(timeout=800)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=15)
                    raise RuntimeError("short-lock pilot time cap exceeded") from None
            path = directory / "report.json"
            result = json.loads(path.read_text()) if path.exists() else None
            journal["checks"].append({"name": name, "returncode": returncode, "report_path": str(path),
                                      "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                                      if path.exists() else None,
                                      "output_agreement": all(r["same_text"] for r in result["runs"])
                                      if result and len(result["runs"]) == 2 else False})
            save(journal)
            print(json.dumps(journal["checks"][-1]), flush=True)
            if returncode != 0:
                raise RuntimeError("pilot rejected; frozen worker will resume")
        journal["status"] = "pilots_completed"
    except Exception as error:
        journal["status"] = "pilot_stopped"
        journal["error"] = str(error)
        raise
    finally:
        resumed = []
        for pid in reversed(stopped):
            try:
                os.kill(pid, signal.SIGCONT)
                resumed.append(pid)
            except ProcessLookupError:
                pass
        journal["resumed_pids"] = resumed
        journal["resumed_at_utc"] = now()
        journal["suspended_elapsed_seconds"] = time.monotonic() - began
        save(journal)
        print(json.dumps({"resumed_pids": resumed, "status": journal["status"]}), flush=True)


if __name__ == "__main__":
    main()
