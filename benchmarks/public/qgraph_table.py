# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26"]
# ///
"""Prints the "Quantized graph index" tables of docs/PUBLIC-BENCHMARKS.md from the results files,
so the prose is written from the files only.

  qgraph_table.py <dataset>...        reads benchmarks/results/public_qgraph_<dataset>.json
  qgraph_table.py --file <path>...    reads the given merged files

Per system: QPS at recall@10 0.90 and 0.95 (interpolated on the Pareto frontier as evaluate.py
does), index bytes, build seconds, whether every timed row met the load gate, the load range, and
the ratio HMS / system for QPS (above 1: HMS faster) and system / HMS for bytes (above 1: HMS
smaller), so that a loss shows as a ratio below 1 with its margin.
"""

import json
import sys
from pathlib import Path

from evaluate import hms_label

RESULTS = Path(__file__).resolve().parents[1] / "results"


def series_info(r: dict) -> dict:
    info: dict[str, dict] = {}
    for row in r["competitors"]:
        if row["index"] == "IndexHNSWFlat":
            key = f"faiss HNSW M={row['params']['M']}"
        elif row["library"] == "hnswlib":
            key = "hnswlib M=16"
        else:
            key = row.get("series") or (None if row["library"] == "faiss" else f"{row['library']} {row['index']}")
        if key is None:
            continue
        i = info.setdefault(key, {"bytes": row["index_bytes"], "build": row["build_secs"], "gate": True, "loads": []})
        i["gate"] &= bool(row["load_gate_met"])
        i["loads"] += list(row["load_1m_before_runs"])
    for h in r["hms"]:
        info[hms_label(h)] = {"bytes": h["index_bytes"], "build": h["build"]["build_secs"],
                              "gate": all(x["load_gate_met"] for x in h["sweep"]),
                              "loads": [x for s in h["sweep"] for x in s["load_1m_before_runs"]]}
    return info


def table(r: dict) -> str:
    info = series_info(r)
    at = r["qps_at_recall"]
    hms_key = next((k for k in at if k.startswith("HMS VGraph (8-bit")), None)
    hms = at.get(hms_key, {})
    hms_bytes = info.get(hms_key, {}).get("bytes")
    out = ["| System | QPS @ 0.90 | QPS @ 0.95 | HMS/sys @ 0.90 | HMS/sys @ 0.95 | Index bytes | sys/HMS bytes | Build s | Gate met | Load 1m |",
           "|---|---|---|---|---|---|---|---|---|---|"]

    def q(x):
        return None if x is None else x.get("qps")

    def num(x, f="{:,.0f}"):
        return "not reached" if x is None else f.format(x)

    def ratio(a, b):
        return "" if a is None or b is None or b == 0 else f"{a / b:.2f}"

    for key, v in at.items():
        i = info.get(key, {})
        q90, q95 = q(v.get("0.90")), q(v.get("0.95"))
        loads = i.get("loads") or []
        out.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            key, num(q90), num(q95), ratio(q(hms.get("0.90")), q90), ratio(q(hms.get("0.95")), q95),
            num(i.get("bytes")), ratio(i.get("bytes"), hms_bytes), num(i.get("build"), "{:,.0f}"),
            "yes" if i.get("gate") else "no", f"{min(loads):.0f}-{max(loads):.0f}" if loads else ""))
    for e in r.get("extra_systems", []):
        if e["status"] != "ran":
            out.append(f"| {e['system']} | {e['status']}: {e['reason']} | | | | | | | | |")
    return "\n".join(out)


def main() -> None:
    args = sys.argv[1:]
    files = [Path(a) for a in args[1:]] if args and args[0] == "--file" else [RESULTS / f"public_qgraph_{a}.json" for a in args]
    for f in files:
        r = json.loads(f.read_text())
        print(f"\n{r['dataset']['name']} ({f.name})\n")
        print(table(r))


if __name__ == "__main__":
    main()
