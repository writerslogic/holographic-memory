"""Usage: /Volumes/A/.hms-target/timed.sh uv run python -I benchmarks/public/batching_timing.py <plan.json> <out.json>

Paired A/B timing of `public-bench lme-model` arms inside one timing lock: every round runs each
arm once, in plan order, so arms alternate. Before each run the 1-minute load average must drop
below `max_load`; the wait shares one `max_wait_secs` budget across the whole invocation, and a
run that starts above the gate is recorded with load_gate_met false. Each run is wrapped in
/usr/bin/time -l for peak memory.

plan.json: {"rounds": 2, "max_load": 8, "max_wait_secs": 900, "out_dir": "...",
            "arms": [{"name": "...", "argv": ["/path/public-bench", "lme-model", ...]}]}
An arm's argv gets `--out <out_dir>/<name>_r<round>.out` appended."""
import json
import re
import subprocess
import sys
import time


def load1():
    v = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout
    return [float(x) for x in v.strip("{} \n").split()]


plan = json.load(open(sys.argv[1]))
deadline = time.monotonic() + plan["max_wait_secs"]
runs = []
for r in range(plan["rounds"]):
    for arm in plan["arms"]:
        while load1()[0] >= plan["max_load"] and time.monotonic() < deadline:
            time.sleep(5)
        before = load1()
        out = f"{plan['out_dir']}/{arm['name']}_r{r}.out"
        t = time.monotonic()
        p = subprocess.run(["/usr/bin/time", "-l", *arm["argv"], "--out", out],
                           capture_output=True, text=True)
        wall = time.monotonic() - t
        line = next((json.loads(x) for x in p.stderr.splitlines() if x.startswith("{")), {})
        peak = re.search(r"(\d+)\s+peak memory footprint", p.stderr)
        rss = re.search(r"(\d+)\s+maximum resident set size", p.stderr)
        run = {"arm": arm["name"], "round": r, "exit": p.returncode, "wall_secs": wall,
               "load_avg_before": before, "load_avg_after": load1(),
               "load_gate_met": before[0] < plan["max_load"], "public_bench": line,
               "peak_memory_footprint_bytes": int(peak.group(1)) if peak else None,
               "max_rss_bytes": int(rss.group(1)) if rss else None, "out": out}
        if p.returncode != 0:
            run["stderr_tail"] = p.stderr[-2000:]
        runs.append(run)
        print(json.dumps({k: run[k] for k in ("arm", "round", "exit", "wall_secs", "load_avg_before")}),
              flush=True)
json.dump({"plan": plan, "runs": runs}, open(sys.argv[2], "w"), indent=1)
