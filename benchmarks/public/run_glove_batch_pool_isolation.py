import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import run_glove_development as query
import run_glove_local_diagnosis as base
import run_glove_prepared_diagnosis as monitor


ROOT = base.ROOT.with_name("vector-glove-batch-pool-ablation-v2")
PREVIOUS = base.ROOT.with_name("vector-glove-batch-pool-ablation-v1")
BASELINE = base.ROOT.with_name("vector-glove-query-diagnosis-v1")
VERTEX = "src/core/qgraph/vertex.rs"
CANDIDATE_SHA = "cf05a410e46491f6e141c199c8613dee4c3cd65711ae407a1bcc76e4b6e5b50e"
BASELINE_BINARY_SHA = "3262dbaed9bdf7b6f4520993b9088057349bd3efb7cd7c133f007a5cfe2a61a5"
ORDERS = [["baseline", "disabled", "enabled"], ["enabled", "baseline", "disabled"], ["disabled", "enabled", "baseline"]]
FILES = ["Cargo.toml", "Cargo.lock", "benchmarks/public/vector_glove_development.rs", "benchmarks/public/analyze_glove_development.py", "benchmarks/public/analyze_glove_batch_pool.py", "benchmarks/public/run_glove_batch_pool_ablation.py", "benchmarks/public/analyze_glove_batch_pool_isolation.py", "benchmarks/public/run_glove_batch_pool_isolation.py", "benchmarks/public/run_glove_development.py", "benchmarks/public/run_glove_local_diagnosis.py", "benchmarks/public/run_glove_prepared_diagnosis.py"]


def copy(source, relative):
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return base.sha(target)


def verify_native(native, candidate_sha):
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base.TREE, text=True).strip() != base.SOURCE:
        raise RuntimeError("wrong baseline revision")
    changed = subprocess.check_output(["git", "diff", "--name-only", "--", "src", "Cargo.lock"], cwd=base.TREE, text=True).splitlines()
    if changed != [VERTEX]:
        raise RuntimeError("candidate must change vertex.rs only")
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "src"], cwd=base.TREE, text=True)
    if untracked:
        raise RuntimeError("untracked native source is not admitted")
    for name, original_sha in native.items():
        expected = candidate_sha if name == VERTEX else original_sha
        if base.sha(base.TREE / name) != expected:
            raise RuntimeError(f"native source SHA mismatch: {name}")


def main(args):
    os.umask(0o077)
    ROOT.mkdir()
    deadline = time.monotonic() + 900
    query.ROOT = ROOT
    monitor.ROOT = ROOT
    native = json.loads((base.ROOT / "source-manifest.json").read_text())["native"]
    baseline_manifest = json.loads((BASELINE / "source-manifest.json").read_text())
    previous_complete = json.loads((PREVIOUS / "complete.json").read_text())
    previous_plan = json.loads((PREVIOUS / "plan.json").read_text())
    previous_compile = json.loads((PREVIOUS / "compile.receipt.json").read_text())
    for name in ["plan.json", "candidate-binary", "compile.receipt.json"]:
        if base.sha(PREVIOUS / name) != previous_complete["sha256"][name]:
            raise RuntimeError("preserved v1 evidence SHA mismatch")
    if previous_compile["returncode"] != 0 or previous_compile["error"] is not None:
        raise RuntimeError("preserved v1 compile did not succeed")
    tests = json.loads(args.tests_receipt.read_text())
    if tests.get("returncode") != 0 or tests.get("error") is not None:
        raise RuntimeError("test receipt does not record successful completion")
    verify_native(native, args.candidate_sha256)
    for name in ["Cargo.toml", "Cargo.lock", "benchmarks/public/vector_glove_development.rs", "benchmarks/public/analyze_glove_development.py", "benchmarks/public/run_glove_development.py", "benchmarks/public/run_glove_local_diagnosis.py", "benchmarks/public/run_glove_prepared_diagnosis.py"]:
        if base.sha(base.TREE / name) != baseline_manifest["harness"][name]:
            raise RuntimeError(f"baseline harness changed: {name}")
    if base.sha(BASELINE / "vector-glove-development") != BASELINE_BINARY_SHA:
        raise RuntimeError("baseline binary SHA mismatch")
    declaration = json.loads((query.INPUT / "declaration.json").read_text())
    for name, expected in declaration["files"].items():
        if Path(name).name != name or base.sha(query.INPUT / name) != expected:
            raise RuntimeError("prepared input SHA mismatch")
    frozen = {}
    for name in FILES:
        frozen[f"harness/{name}"] = copy(base.TREE / name, f"harness/{name}")
    for name in native:
        frozen[f"candidate-source/{name}"] = copy(base.TREE / name, f"candidate-source/{name}")
    frozen["candidate-tests.receipt.json"] = copy(args.tests_receipt, "candidate-tests.receipt.json")
    frozen["baseline-source-manifest.json"] = copy(base.ROOT / "source-manifest.json", "baseline-source-manifest.json")
    frozen["baseline-harness-manifest.json"] = copy(BASELINE / "source-manifest.json", "baseline-harness-manifest.json")
    frozen["baseline.sample.txt"] = copy(BASELINE / "sampled.sample.txt", "baseline.sample.txt")
    frozen["baseline.sample.receipt.json"] = copy(BASELINE / "sampled.receipt.json", "baseline.sample.receipt.json")
    frozen["baseline-binary"] = copy(BASELINE / "vector-glove-development", "baseline-binary")
    frozen["v1-compile.receipt.json"] = copy(PREVIOUS / "compile.receipt.json", "v1-compile.receipt.json")
    frozen["v1-plan.json"] = copy(PREVIOUS / "plan.json", "v1-plan.json")
    rustc = subprocess.check_output(["rustc", "--version", "--verbose"], text=True)
    if rustc != baseline_manifest["rustc"]:
        raise RuntimeError("toolchain changed from baseline")
    env = {key: os.environ[key] for key in ["PATH", "HOME", "TMPDIR", "RUSTUP_HOME", "CARGO_HOME", "DEVELOPER_DIR", "SDKROOT"] if key in os.environ}
    env.update({"CARGO_TARGET_DIR": str(base.TREE / "target"), "CARGO_BUILD_JOBS": "1", "RAYON_NUM_THREADS": "1", "CARGO_PROFILE_RELEASE_DEBUG": "1", "CARGO_PROFILE_RELEASE_STRIP": "none", "HMS_QGRAPH_PREFETCH": "0", "HMS_QGRAPH_BATCH_EST": "0", "HMS_QGRAPH_FRONTIER": "1", "HMS_QGRAPH_BLOCK_SCREEN": "0", "HMS_QGRAPH_RERANK_Z": "0"})
    declared_env = {key: value for key, value in env.items() if key.startswith(("CARGO_", "RAYON_", "HMS_"))}
    plan = {
        "schema": "hms-glove-batch-pool-ablation-plan-v2", "development_only": True,
        "source": base.SOURCE, "candidate_vertex_sha256": args.candidate_sha256,
        "candidate_mechanism": "Same8-key batch-merge mechanism, API and buffers asv1; dispatch once per query to const-specialized loops to address measured disabled-arm regression; no threshold tuning",
        "reason_for_isolation": "Preservedv1 enabled/S improvement did not excuse disabled/S regression; defaultcontrol gates declared before v2 data",
        "baseline_binary_sha256": BASELINE_BINARY_SHA, "tests_receipt_sha256": frozen["candidate-tests.receipt.json"],
        "review_scope": "Parent explicitly reviews the exact candidate SHA and successful test evidence before invoking this script; receipt alone is not independent proof of source/test binding",
        "input": str(query.INPUT), "index": str(query.INDEX),
        "input_declaration_sha256": base.sha(query.INPUT / "declaration.json"), "index_sha256": base.sha(query.INDEX),
        "block_orders": ORDERS, "efs": [10, 20, 40, 80, 120, 160, 240, 400, 640], "repeats_per_process": 7,
        "k": 10, "rerank": 64, "seed": 539365412, "knob_mode": "env",
        "arms": {"baseline": {"binary": "baseline-binary", "HMS_QGRAPH_BATCH_POOL": "0"}, "disabled": {"binary": "candidate-binary", "HMS_QGRAPH_BATCH_POOL": "0"}, "enabled": {"binary": "candidate-binary", "HMS_QGRAPH_BATCH_POOL": "1"}},
        "sample_after_timings": {"arm": "enabled", "ef": 240, "repeats": 1000, "duration_s": 5, "interval_ms": 1},
        "primary_gate": {"ef": 80, "recall_floor": 0.95, "minimum_geomean_qps_ratio": 1.10, "bootstrap_lower_strictly_above": 1.0, "contrasts": ["enabled_over_baseline", "enabled_over_disabled"]},
        "default_control_gate": {"ef": 80, "contrast": "disabled_over_baseline", "minimum_geomean_qps_ratio": 0.99, "bootstrap_lower_at_least": 0.98},
        "secondary_default_control_gate": {"contrast": "disabled_over_baseline", "efs": [10, 20, 40, 80, 120, 160, 240, 400, 640], "minimum_geomean_qps_ratio": 0.98},
        "semantic_gate": "All ordered IDs and native-reported distance evaluations match exactS across every ef,query,repeat,process. Baseline/disabled timing equality is reported, not assumed.",
        "causal_gate": "Review query-only sampled call-site contrast consistent with bulkmerge mechanism; no claim from speed alone. Samples excluded from timed comparisons.",
        "uncertainty": "Collapse7 repeats per fresh process, paired whole-process-block bootstrap enumerating27 resamples;3blocks lowpower, conditionalhardware only. No query/repeat pseudoreplication.",
        "limits": {"compile_s": 600, "native_s": 180, "overall_s": 900, "compile_sampled_tree_rss_gib": 4, "native_sampled_tree_rss_gib": 2},
        "memory_scope": "Approximately200ms sampled summed child-tree RSS, not strict OS peak or unique memory; controller/source preparation/sampler excluded",
        "lock": str(Path.home() / "hms-timing.lock"), "environment": declared_env,
        "official_test_arrays_read": False, "competitive_or_leaderboard_claim": False,
        "rustc": rustc, "hardware": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize", "hw.ncpu"], text=True),
        "previous_candidate": {"root": str(PREVIOUS), "vertex_sha256": previous_plan["candidate_vertex_sha256"], "binary_sha256": previous_complete["sha256"]["candidate-binary"], "binary_bytes": (PREVIOUS / "candidate-binary").stat().st_size, "compile_receipt_sha256": frozen["v1-compile.receipt.json"], "compile_elapsed_s": previous_compile["elapsed_s"], "compile_sampled_summed_process_tree_rss_kib": previous_compile["sampled_summed_process_tree_rss_kib"]},
        "compile_cost_scope": "Report measuredv1/v2 compile elapsed time and sampled summed child-tree RSS with binary byte size; builds reuse their private target cache and are not matched clean builds. Code growth is a cost, not a success claim.",
        "frozen_source_and_harness_sha256": frozen,
    }
    base.save(ROOT / "plan.json", plan)
    lock = (Path.home() / "hms-timing.lock").open("a+")
    base.fcntl.flock(lock, base.fcntl.LOCK_EX | base.fcntl.LOCK_NB)
    try:
        with Path("/Volumes/A/hms-coordination/RUN-ALL.md").open("a") as coordination:
            coordination.write(f"\n{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} Vector batch-pool isolationv2 owner acquired local timing flock; plan {ROOT / 'plan.json'},same8-key mechanism withquery-level constdispatch, frozen defaultcontrol gates and3balanced blocks ofS/disabled/enabled full9efcurves with7withinprocess repeats; distinct sample afterward.1compilejob/1Rayon,4GiBcompile/2GiBnative sampledRSS,600/180/900s guards. No finalarrays/remoteaction/lexical work.\n")
        verify_native(native, args.candidate_sha256)
        monitor.run("compile", ["cargo", "build", "--offline", "--locked", "--release", "--features", "experimental", "--bin", "vector-glove-development"], env, 600, deadline, 4)
        verify_native(native, args.candidate_sha256)
        frozen["candidate-binary"] = copy(base.TREE / "target/release/vector-glove-development", "candidate-binary")
        compile_receipt = json.loads((ROOT / "compile.receipt.json").read_text())
        compile_cost = {"v1": plan["previous_candidate"], "v2": {"vertex_sha256": args.candidate_sha256, "binary_sha256": frozen["candidate-binary"], "binary_bytes": (ROOT / "candidate-binary").stat().st_size, "compile_receipt_sha256": base.sha(ROOT / "compile.receipt.json"), "compile_elapsed_s": compile_receipt["elapsed_s"], "compile_sampled_summed_process_tree_rss_kib": compile_receipt["sampled_summed_process_tree_rss_kib"]}, "scope": plan["compile_cost_scope"]}
        timed_runs = {}
        for block, order in enumerate(ORDERS):
            for position, arm in enumerate(order):
                name = f"block{block}-{arm}"
                settings = plan["arms"][arm]
                arm_env = dict(env, HMS_QGRAPH_BATCH_POOL=settings["HMS_QGRAPH_BATCH_POOL"])
                command = [str(ROOT / settings["binary"]), str(query.INPUT), str(query.INDEX), str(ROOT / f"{name}.json"), "--knobs", "env"]
                started = time.monotonic()
                query.query_run(name, command, arm_env, deadline)
                timed_runs[name] = {"block": block, "position": position, "arm": arm, "command": command, "started_monotonic_s": started, "finished_monotonic_s": time.monotonic(), "environment": dict(declared_env, HMS_QGRAPH_BATCH_POOL=settings["HMS_QGRAPH_BATCH_POOL"]), "binary_sha256": frozen[settings["binary"]], "result_sha256": base.sha(ROOT / f"{name}.json")}
        sample_env = dict(env, HMS_QGRAPH_BATCH_POOL="1")
        query.query_run("candidate-sampled", [str(ROOT / "candidate-binary"), str(query.INPUT), str(query.INDEX), str(ROOT / "candidate-sampled.json"), "--knobs", "env", "--efs", "240", "--repeats", "1000"], sample_env, deadline, sampling=True)
        profile = {name: {"path": name, "sha256": base.sha(ROOT / name)} for name in ["baseline.sample.txt", "baseline.sample.receipt.json", "candidate-sampled.sample.txt", "candidate-sampled.receipt.json", "candidate-sampled.json"] if (ROOT / name).exists()}
        base.save(ROOT / "execution-manifest.json", {"plan_sha256": base.sha(ROOT / "plan.json"), "frozen_files": frozen, "timed_runs": timed_runs, "profile_evidence": profile, "compile_cost": compile_cost})
        monitor.run("analyze", [sys.executable, str(base.TREE / "benchmarks/public/analyze_glove_batch_pool_isolation.py"), "--root", str(ROOT), "--out", str(ROOT / "analysis.json")], env, 180, deadline, 2)
        base.save(ROOT / "complete.json", {"status": "completed-development-isolation-v2", "acceptance": json.loads((ROOT / "analysis.json").read_text())["primary"], "competitive_claim": False, "new_paid_compute_usd": 0, "final_load": os.getloadavg(), "sha256": {str(path.relative_to(ROOT)): base.sha(path) for path in ROOT.rglob("*") if path.is_file()}})
        print(json.dumps({"status": "completed-development-isolation-v2", "root": str(ROOT)}))
    finally:
        base.fcntl.flock(lock, base.fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--tests-receipt", type=Path, required=True)
    args = parser.parse_args()
    if re.fullmatch("[0-9a-f]{64}", args.candidate_sha256) is None:
        parser.error("candidate SHA256 must be64lowercase hexadecimal characters")
    if args.candidate_sha256 != CANDIDATE_SHA:
        parser.error("candidate SHA256 differs from the declared isolation-v2 source")
    if ROOT.exists():
        raise SystemExit("Run directory already exists; preserve it and declare a new attempt")
    try:
        main(args)
    except BaseException as error:
        if ROOT.exists():
            base.save(ROOT / "failure.json", {"error": f"{type(error).__name__}: {error}", "automatic_retry": False})
        raise
