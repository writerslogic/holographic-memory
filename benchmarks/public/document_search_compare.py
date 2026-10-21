# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///
"""Paired persistent-document scan/postings comparison with exact output checks."""

import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import signal
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/results/superiority_v2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, q):
    values = sorted(values)
    at = (len(values) - 1) * q
    low = int(at)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (at - low)


def time_arguments(system):
    if system == "Darwin":
        return ["/usr/bin/time", "-l"]
    if system == "Linux":
        return ["/usr/bin/time", "-v"]
    raise ValueError(f"unsupported timing platform: {system}")


def peak_rss_bytes(log, system):
    pattern, scale = {
        "Darwin": (r"(\d+)\s+maximum resident set size", 1),
        "Linux": (r"Maximum resident set size \(kbytes\):\s*(\d+)", 1024),
    }[system]
    match = re.search(pattern, log)
    if match is None or int(match.group(1)) <= 0:
        raise ValueError("missing or invalid OS peak RSS counter")
    return int(match.group(1)) * scale


def run_timed(command, log, timeout):
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def terminate(_signal, _frame):
    raise SystemExit("comparison terminated; active process group cleaned up")


def main():
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--out", type=Path, default=OUT / "document_paired")
    parser.add_argument("--lock-path", type=Path, default=Path("./.hms-target/timing.lock"))
    args = parser.parse_args()
    if not 1 <= args.rounds <= 11:
        raise ValueError("rounds outside budget")
    folder = args.out
    folder.mkdir(exist_ok=False)
    binaries = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "src").rglob("*.rs"))}
    binary_hashes = {key: sha(path) for key, path in binaries.items()}
    runs, cells = [], {}
    started = time.monotonic()
    lock_path = args.lock_path
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for n in (1000, 10000):
            for repetition in range(args.rounds):
                outputs, measurements = {}, {}
                order = ("baseline", "candidate") if repetition % 2 == 0 else ("candidate", "baseline")
                for arm in order:
                    remaining = 850 - (time.monotonic() - started)
                    if remaining <= 0:
                        raise RuntimeError("900-second timing-lock budget nearly exhausted")
                    raw = folder / f"{n}-{repetition}-{arm}.json"
                    log_path = raw.with_suffix(".log")
                    load_before = os.getloadavg()
                    command = [*time_arguments(platform.system()), str(binaries[arm]), "--documents", str(n), "--out", str(raw)]
                    with log_path.open("w") as log:
                        run_timed(command, log, min(120, remaining))
                    report = json.loads(raw.read_text())
                    keys = [(row["mode"], row["filtered"], row["query"], row["repetition"]) for row in report["rows"]]
                    if len(keys) != 576 or len(set(keys)) != 576 or report["documents"] != n:
                        raise ValueError("invalid benchmark row identities")
                    outputs[arm] = {key: row["results"] for key, row in zip(keys, report["rows"])}
                    by_cell = {}
                    for row in report["rows"]:
                        by_cell.setdefault((row["mode"], row["filtered"]), []).append(row["elapsed_ns"])
                    measurements[arm] = {key: {"mean":statistics.mean(v),"p50":quantile(v,.5),"p95":quantile(v,.95),"p99":quantile(v,.99)} for key, v in by_cell.items()}
                    with raw.open("rb") as source, gzip.open(raw.with_suffix(".json.gz"), "wb") as target:
                        target.write(source.read())
                    raw.unlink()
                    peak = peak_rss_bytes(log_path.read_text(), platform.system())
                    load_after = os.getloadavg()
                    runs.append({"n":n,"round":repetition,"arm":arm,"order":list(order),
                        "command":command,"load_before":load_before,"load_after":load_after,
                        "load_gate_met":load_before[0]<3 and load_after[0]<3,
                        "peak_process_rss_bytes":peak,
                        "ingest_seconds":report["ingest_seconds"],"artifact":raw.with_suffix(".json.gz").name,
                        "sha256":sha(raw.with_suffix(".json.gz"))})
                if outputs["baseline"] != outputs["candidate"]:
                    raise ValueError(f"exact result mismatch at N={n}, round={repetition}")
                for key in measurements["baseline"]:
                    series = cells.setdefault((n, *key), {metric:[] for metric in ("mean","p50","p95","p99")})
                    for metric in series:
                        series[metric].append(measurements["baseline"][key][metric]/measurements["candidate"][key][metric])
                print(f"N={n} round={repetition}: exact parity", flush=True)
    if sources != {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT / "src").rglob("*.rs"))}:
        raise ValueError("source changed during experiment")
    if binary_hashes != {key:sha(path) for key,path in binaries.items()}:
        raise ValueError("executable changed during experiment")
    summary = []
    for (n, mode, filtered), metrics in cells.items():
        rng = random.Random(24301)
        ratios = metrics["mean"]
        boot = [statistics.median(rng.choices(ratios,k=len(ratios))) for _ in range(5000)]
        summary.append({"n":n,"mode":mode,"filtered":filtered,
            "paired_baseline_over_candidate":metrics,
            "median_ratios":{key:statistics.median(value) for key,value in metrics.items()},
            "qps_ratio_ci95_exploratory":[quantile(boot,.025),quantile(boot,.975)],
            "timing_gate_met":all(r["load_gate_met"] for r in runs if r["n"]==n)})
    result={"exact_output_parity":True,"queries_compared":576*args.rounds*2,
        "hardware":platform.platform(),"machine":platform.machine(),"binary_sha256":binary_hashes,
        "source_sha256":sources,"rounds":args.rounds,"runs":runs,"cells":summary,
        "serving_rss_isolated":False,"not_competitor_superiority":True,
        "interpretation":"synthetic initial comparison; shared-host load and no multiplicity correction, no primary-contract claim"}
    (folder/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
