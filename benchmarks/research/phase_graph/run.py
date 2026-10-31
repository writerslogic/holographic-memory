#!/usr/bin/env python3
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compile actual before/after graph sources with their actual hash mixer."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import re
import statistics
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]


def command(args):
    return subprocess.check_output(args, cwd=ROOT, text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="12c80dec")
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for label in ("before", "after"):
        sources[label] = {}
        for name in ("phase_graph", "entangled"):
            path = f"src/core/{name}.rs"
            sources[label][name] = (
                command(["git", "show", f"{args.baseline}:{path}"])
                if label == "before"
                else (ROOT / path).read_text()
            )
    environment = {
        "baseline": command(["git", "rev-parse", args.baseline]).strip(),
        "platform": platform.platform(),
        "rustc": command(["rustc", "-Vv"]).strip(),
        "driver_sha256": hashlib.sha256(Path(__file__).with_name("bench.rs").read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "compiler_flags": ["--edition=2021", "-C", "opt-level=3", "-C", "lto=true", "-C", "codegen-units=1"],
        "source_sha256": {
            label: {name: hashlib.sha256(value.encode()).hexdigest() for name, value in files.items()}
            for label, files in sources.items()
        },
        "note": "Standalone actual graph/hash; FastMap uses std HashMap in both builds. Query loops do not access this membership map. Task-local scheduling; unrelated host jobs may remain. Not an end-to-end search benchmark.",
    }
    rows = []
    with tempfile.TemporaryDirectory(prefix="warm-phase-graph-") as directory:
        build = Path(directory)
        for label, files in sources.items():
            graph = build / f"{label}.rs"
            graph.write_text(files["phase_graph"])
            mixer = re.search(r"pub fn hash_u64\(a: u64, b: u64\) -> u64 \{.*?\n\}", files["entangled"], re.S)
            if mixer is None:
                raise RuntimeError("hash_u64 source format changed")
            wrapper = build / f"{label}_driver.rs"
            wrapper.write_text(
                "mod core { pub mod hash { pub type FastMap<K,V> = std::collections::HashMap<K,V>; }"
                + "pub mod entangled {" + mixer.group() + "}"
                + f"#[allow(dead_code)] #[path={json.dumps(str(graph))}] pub mod phase_graph; }}"
                + f"include!({json.dumps(str(Path(__file__).with_name('bench.rs')))});"
            )
            command(["rustc", *environment["compiler_flags"], str(wrapper), "-o", str(build / label)])
        for facts, dim, n in ((128, 512, 16), (512, 1024, 256), (2048, 1024, 256)):
            for trial in range(args.trials):
                for label in (("before", "after") if trial % 2 == 0 else ("after", "before")):
                    row = json.loads(command([str(build / label), str(facts), str(dim), str(n)]))
                    row.update(variant=label, trial=trial)
                    rows.append(row)
                    with (args.output / "cases.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
    summary = []
    for shape in sorted({(r["facts"], r["dim"], r["n_phases"]) for r in rows}):
        selected = [r for r in rows if (r["facts"], r["dim"], r["n_phases"]) == shape]
        medians = {
            label: {metric: statistics.median(r[metric] for r in selected if r["variant"] == label)
                    for metric in ("build_ns", "query_ns", "score_ns", "forward_correct", "backward_correct", "three_hop_correct", "mean_reverse_score")}
            for label in ("before", "after")
        }
        summary.append({"shape": shape, "medians": medians,
                        "speedup": {key: medians["before"][key] / medians["after"][key] for key in ("build_ns", "query_ns", "score_ns")}})
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
