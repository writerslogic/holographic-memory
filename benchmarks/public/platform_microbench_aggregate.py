# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Aggregate repeated `dot_kernel_microbench` runs into one results file (platform area).

  cargo test --release --lib dot_kernel_microbench --no-run   # note the test binary
  for i in 1 2 3 4 5 6; do WARM_PLATFORM_BENCH_OUT=run$i.json <test-binary> dot_kernel_microbench --ignored; done
  uv run python benchmarks/public/platform_microbench_aggregate.py run*.json \
      --host "Apple M4, macOS" --out benchmarks/results/platform_aarch64_microbench_v1.json

Per (dim, kernel, variant): the minimum and median over runs of the per-run best-of-7 ns per
call, cold (random rows, prefetched batches) and hot (L1-resident rows). `ratio_to_incumbent`
is the variant's minimum over the incumbent's minimum for the same (dim, kernel, column), where
the incumbent is `--incumbent` (below 1 is faster); it is only a ranking aid, the kill rule is
in docs/performance/platforms.md.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--host", required=True)
    ap.add_argument("--incumbent", default="sdot", help="variant the others are ranked against")
    ap.add_argument("--load", default="", help="comma-separated 1-minute load before each run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cells: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: {"cold": [], "hot": []})
    for path in a.runs:
        for r in json.loads(Path(path).read_text())["results"]:
            key = (r["dim"], r["kernel"], r["variant"])
            if r.get("ns_cold") is not None:
                cells[key]["cold"].append(r["ns_cold"])
            if r.get("ns_hot") is not None:
                cells[key]["hot"].append(r["ns_hot"])
    rows = []
    for (dim, kernel, variant), v in cells.items():
        row = {"dim": dim, "kernel": kernel, "variant": variant, "runs": len(v["hot"])}
        for col in ("cold", "hot"):
            if v[col]:
                row[f"ns_{col}_min"] = round(min(v[col]), 3)
                row[f"ns_{col}_median"] = round(statistics.median(v[col]), 3)
        rows.append(row)
    base = {(r["dim"], r["kernel"]): r for r in rows if r["variant"] == a.incumbent}
    for r in rows:
        b = base.get((r["dim"], r["kernel"]))
        if b and r["variant"] != a.incumbent:
            for col in ("cold", "hot"):
                if f"ns_{col}_min" in r and f"ns_{col}_min" in b:
                    r[f"ratio_to_incumbent_{col}"] = round(r[f"ns_{col}_min"] / b[f"ns_{col}_min"], 3)
    out = {
        "schema_version": 1, "category": "platform_kernels", "date": time.strftime("%Y-%m-%d"),
        "host": a.host, "incumbent": a.incumbent, "runs": len(a.runs),
        "load_1m_before_runs": [float(x) for x in a.load.split(",") if x],
        "protocol": "dot_kernel_microbench (src/core/qgraph/kernels.rs): 400k random rows, 64-row prefetched batches (cold) "
                    "and 64 L1-resident rows (hot); per run best of 7 passes of 2^20 calls; fn-pointer calls for all variants",
        "rows": rows,
    }
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
