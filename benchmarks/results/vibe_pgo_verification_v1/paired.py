# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


HASHES = {
    "v3": "5d6f39deb003bed7fef3c6f06fca3d03ef989e7026c3f020779fbd87afe7c662",
    "pgo": "1b9119dced3bcf73b61b9fa91af324de5150d9d6b1c25e98eed1b413ad496968",
    "native": "a958edc6373e8a0fd65ffda8ad1bce993ccc8d993cba5f0e2b020daac02c5713",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    assert len(manifest["runs"]) == 18
    run_load_max = max(max(r["load_before"], r["load_after"]) for r in manifest["runs"])
    rows = []
    for dataset in ("glove", "sift"):
        runs = {
            (r, arm): json.loads((folder / f"{dataset}-r{r}-{arm}.json").read_text())
            for r in range(3) for arm in HASHES
        }
        for run in runs.values():
            assert run["k"] == 10 and run["n_queries"] == 10000 and run["repeats"] == 1
            assert run["build"]["threads"] == 1
            assert run["index_file"]["loaded"]
            assert len(run["sweep"]) == 3
        for i, ef in enumerate((64, 128, 256)):
            base = [runs[r, "v3"]["sweep"][i] for r in range(3)]
            for arm in ("pgo", "native"):
                candidate = [runs[r, arm]["sweep"][i] for r in range(3)]
                for b, c in zip(base, candidate):
                    assert b["params"] == c["params"]
                    assert b["params"]["ef"] == ef
                    assert b["params"]["rerank"] == 100
                    assert b["recall_at_10"] == c["recall_at_10"]
                    assert b["mean_exact_evals_per_query"] == c["mean_exact_evals_per_query"]
                ratios = [c["qps_single_thread"] / b["qps_single_thread"] for b, c in zip(base, candidate)]
                loads = [v for cell in base + candidate for v in cell["load_1m_before_runs"]]
                rows.append(dict(dataset=dataset, ef=ef, arm=arm, ratios=ratios,
                                 median_ratio=statistics.median(ratios),
                                 recall_at_10=base[0]["recall_at_10"],
                                 mean_exact_evals_per_query=base[0]["mean_exact_evals_per_query"],
                                 load_max=max(max(loads), run_load_max),
                                 eligible=max(max(loads), run_load_max) < 3
                                          and all(cell["load_gate_met"] for cell in base + candidate)))
    pgo = [row for row in rows if row["arm"] == "pgo"]
    return dict(rows=rows, passes_development_bar=all(r["eligible"] and r["median_ratio"] >= 1.03 for r in pgo),
                production_adoption=False, limitations=["Older pinned source 99e84d92; current-source and wheel validation remain required.",
                                                       "k=10 development comparison, not the VIBE k=100 protocol or a leader comparison."])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--glove", type=Path)
    parser.add_argument("--sift", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        print(json.dumps(summarize(args.out), indent=2))
        return
    if not all((args.root, args.glove, args.sift)):
        parser.error("--root, --glove and --sift are required for a run")
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = dict(source="99e84d92f72a42e6f54b2cc48fb30fc86cdfa99c", binary_sha256=HASHES,
                    started_unix=time.time(), cpu="Intel Core Ultra 9 185H", threads=1,
                    affinity=[0], rounds=3, queries=10000, count=10, max_load=3, runs=[])
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if os.getloadavg()[0] >= 3:
        raise RuntimeError("load gate failed before hashing inputs")
    for arm, expected in HASHES.items():
        assert digest(args.root / f"t-{arm}/release/public-bench") == expected, arm
    manifest["inputs"] = {}
    for name, path in (("glove", args.glove), ("sift", args.sift)):
        manifest["inputs"][name] = {p.name: digest(p) for p in path.iterdir() if p.is_file()}
        manifest["inputs"][name]["saved_index"] = digest(args.root / f"ab/idx-{name}")
        manifest["inputs"][name]["saved_index_metadata"] = digest(args.root / f"ab/idx-{name}.json")
    manifest["profile_sha256"] = digest(args.root / "merged.profdata")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = os.environ | {"RAYON_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    for r, order in enumerate((("v3", "pgo", "native"), ("pgo", "native", "v3"), ("native", "v3", "pgo"))):
        for dataset, data in (("glove", args.glove), ("sift", args.sift)):
            for arm in order:
                load = os.getloadavg()[0]
                if load >= 3:
                    raise RuntimeError(f"load gate failed: {load}")
                stem = f"{dataset}-r{r}-{arm}"
                command = ["taskset", "-c", "0", f"t-{arm}/release/public-bench", "ann-qgraph",
                           "--data", str(data), "--out", str(args.out / f"{stem}.json"),
                           "--load-index", f"ab/idx-{dataset}", "--index", "vertex", "--residual",
                           "--align-rows", "--reorder", "--quantized-search", "--degree", "32",
                           "--build-ef", "200", "--ef", "64,128,256", "--rerank", "100",
                           "--count", "10", "--queries", "10000", "--repeats", "1",
                           "--max-load", "3", "--max-wait-secs", "0"]
                started = time.time()
                with (args.out / f"{stem}.log").open("w") as log:
                    subprocess.run(command, cwd=args.root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=90, check=True)
                load_after = os.getloadavg()[0]
                manifest["runs"].append(dict(stem=stem, command=command, started_unix=started,
                                             ended_unix=time.time(), load_before=load, load_after=load_after))
                (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                output = json.loads((args.out / f"{stem}.json").read_text())
                if load_after >= 3 or any(not cell["load_gate_met"] or
                                         max(cell["load_1m_before_runs"]) >= 3
                                         for cell in output["sweep"]):
                    raise RuntimeError(f"load gate failed after {stem}")
    (args.out / "summary.json").write_text(json.dumps(summarize(args.out), indent=2) + "\n")


if __name__ == "__main__":
    main()
