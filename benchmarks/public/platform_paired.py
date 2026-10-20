# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Paired kernel comparison of two `public-bench` binaries on one host (platform area).

  uv run python benchmarks/public/platform_paired.py \
      --arm base=/path/public-bench-base --arm new=/path/public-bench-new \
      (an arm may add environment: --arm vnni=/path/public-bench@HMS_QGRAPH_KERNEL=vnni) \
      --data ~/.cache/hms-bench/glove-100-angular --cache /path/glove.graph \
      --threads 1,4 --ef 256,768 --repeats 7 --out benchmarks/results/platform_aarch64_glove_v1.json

Each arm is one `ann-qgraph` process per search thread (the harness times one query at a time on
one thread, so T concurrent processes over the same cached graph are T search threads). The
arms are alternated round by round through the harness's `--sync` hand-shake, so a round of one
arm never overlaps a round of the other, and the arm order flips every round to cancel drift.
The ratio is the median of the per-round new/base ratios of the summed QPS, reported with the
minimum, the maximum and the number of rounds in which new beat base. Recall and exact
evaluations per query must match between arms (the kernels are bit-exact, so the search visits
the same candidates); a mismatch is recorded and fails the run. Load averages before every
round are recorded; the harness's own load gate is disabled because this host is shared.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CONFIG = [
    "--index", "vertex", "--vertex-bits", "4", "--residual", "--residual-bits", "8",
    "--degree", "32", "--build-ef", "200", "--align-rows", "--reorder", "--id-bytes", "3",
    "--rerank", "64", "--holdout", "2000", "--max-load", "100000", "--max-wait-secs", "0",
]


def wait_for(path: Path, procs: list[subprocess.Popen], timeout: float) -> None:
    end = time.time() + timeout
    while not path.exists():
        for p in procs:
            if p.poll() not in (None, 0):
                raise SystemExit(f"a harness process exited {p.returncode} before {path.name}")
        if time.time() > end:
            raise SystemExit(f"timed out waiting for {path}")
        time.sleep(0.05)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="name=path[@KEY=VALUE[,KEY=VALUE]]: a public-bench binary and optional environment")
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True, help="graph cache shared by every arm (built by the first run if absent)")
    ap.add_argument("--threads", default="1")
    ap.add_argument("--ef", default="256,768")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    arms: dict[str, str] = {}
    envs: dict[str, dict[str, str]] = {}
    for spec in a.arm:
        name, rest = spec.split("=", 1)
        path, _, env = rest.partition("@")
        arms[name] = path
        envs[name] = dict(kv.split("=", 1) for kv in env.split(",") if kv)
    if len(arms) != 2:
        sys.exit("exactly two arms")
    names = list(arms)
    report = {"schema_version": 1, "date": time.strftime("%Y-%m-%d"), "host": os.uname().machine,
              "config": CONFIG, "ef": a.ef, "repeats": a.repeats, "tag": a.tag, "arms": arms, "arm_env": envs, "threads": {}}
    for threads in [int(t) for t in a.threads.split(",")]:
        work = Path(tempfile.mkdtemp(prefix="platform-paired-"))
        procs: dict[str, list[subprocess.Popen]] = {n: [] for n in names}
        for n in names:
            for t in range(threads):
                cmd = [arms[n], "ann-qgraph", "--data", a.data, "--out", str(work / f"{n}-{t}.json"),
                       "--graph-cache", a.cache, "--sync", str(work / f"{n}-{t}"), "--ef", a.ef,
                       "--repeats", str(a.repeats), *CONFIG]
                procs[n].append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, env=os.environ | envs[n],
                                                 stderr=open(work / f"{n}-{t}.log", "w")))
            # The first process of the first arm builds the cache if it is missing; wait for it
            # to be ready before the others read the file.
            wait_for(work / f"{n}-0.ready", procs[n], 7200)
        for n in names:
            for t in range(threads):
                wait_for(work / f"{n}-{t}.ready", procs[n], 7200)
        loads = []
        for rep in range(a.repeats):
            order = names if rep % 2 == 0 else names[::-1]
            loads.append(os.getloadavg()[0])
            for n in order:
                for t in range(threads):
                    (work / f"{n}-{t}.go.{rep}").write_text("")
                for t in range(threads):
                    wait_for(work / f"{n}-{t}.done.{rep}", procs[n], 3600)
        for n in names:
            for p in procs[n]:
                if p.wait() != 0:
                    raise SystemExit(f"{n} exited {p.returncode}")
        results = {n: [json.loads((work / f"{n}-{t}.json").read_text()) for t in range(threads)] for n in names}
        rows = []
        mismatch = []
        for i, row in enumerate(results[names[0]][0]["sweep"]):
            per_arm = {}
            for n in names:
                runs = [r["sweep"][i]["qps_runs"] for r in results[n]]
                per_arm[n] = [sum(col) for col in zip(*runs)]
                same = {(round(r["sweep"][i]["recall_at_10"], 6), round(r["sweep"][i]["mean_exact_evals_per_query"], 3))
                        for r in results[n]}
                if len(same) != 1:
                    mismatch.append((n, row["params"]["ef"], "threads disagree"))
                per_arm[n + "_key"] = next(iter(same))
            if per_arm[names[0] + "_key"] != per_arm[names[1] + "_key"]:
                mismatch.append((row["params"]["ef"], per_arm[names[0] + "_key"], per_arm[names[1] + "_key"]))
            ratios = [x / y for x, y in zip(per_arm[names[1]], per_arm[names[0]])]
            rows.append({
                "ef": row["params"]["ef"], "recall_at_10": per_arm[names[0] + "_key"][0],
                "evals_per_query": per_arm[names[0] + "_key"][1],
                f"qps_{names[0]}_rounds": [round(x, 1) for x in per_arm[names[0]]],
                f"qps_{names[1]}_rounds": [round(x, 1) for x in per_arm[names[1]]],
                f"qps_{names[0]}_median": round(statistics.median(per_arm[names[0]]), 1),
                f"qps_{names[1]}_median": round(statistics.median(per_arm[names[1]]), 1),
                "ratio_median_of_rounds": round(statistics.median(ratios), 4),
                "ratio_min": round(min(ratios), 4), "ratio_max": round(max(ratios), 4),
                "rounds_new_faster": sum(r > 1 for r in ratios), "rounds": len(ratios),
            })
        report["threads"][str(threads)] = {"rows": rows, "load_1m_before_round": [round(x, 1) for x in loads],
                                           "result_mismatch": mismatch}
        if mismatch:
            Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
            raise SystemExit(f"search results differ between arms: {mismatch}")
    Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["threads"], indent=1)[:3000])


if __name__ == "__main__":
    main()
