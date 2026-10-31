import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import tomllib

import run_glove_development as query
import run_glove_local_diagnosis as base
import run_glove_prepared_diagnosis as monitor


ROOT = base.ROOT.with_name("vector-glove-batch-pool-ablation-v3")
PREVIOUS = base.ROOT.with_name("vector-glove-batch-pool-ablation-v2")
BASELINE = base.ROOT.with_name("vector-glove-query-diagnosis-v1")
VERTEX = "src/core/qgraph/vertex.rs"
CANDIDATE_SHA = "d55f61e457db1e49d9dd6c13dabc9db6f376e7c90a66e04063efc2e85697cd04"
CARGO_SHA = "04f4f836c6a8bad2defb465eb5399276f67e1121538c218a3f5e31bf020dabcd"
BASELINE_BINARY_SHA = "3262dbaed9bdf7b6f4520993b9088057349bd3efb7cd7c133f007a5cfe2a61a5"
ORDERS = [["baseline", "disabled", "enabled"], ["enabled", "baseline", "disabled"], ["disabled", "enabled", "baseline"]]
FILES = ["Cargo.toml", "Cargo.lock", "examples/batched_vector_search.rs", "benchmarks/public/vector_glove_development.rs", "benchmarks/public/analyze_glove_development.py", "benchmarks/public/analyze_glove_batch_pool.py", "benchmarks/public/analyze_glove_batch_pool_feature.py", "benchmarks/public/run_glove_batch_pool_feature.py", "benchmarks/public/run_glove_development.py", "benchmarks/public/run_glove_local_diagnosis.py", "benchmarks/public/run_glove_prepared_diagnosis.py"]
FEATURES = {"baseline": "experimental", "disabled": "experimental", "enabled": "experimental,experimental-batch-pool"}


def validate_cargo(current_bytes, original_bytes):
    current = tomllib.loads(current_bytes.decode())
    original = tomllib.loads(original_bytes.decode())
    if current["features"].pop("experimental-batch-pool", None) != ["experimental"]:
        raise ValueError("new feature must depend only on experimental")
    entries = current.get("example", [])
    added = [entry for entry in entries if entry.get("name") == "batched_vector_search"]
    allowed = {"name": "batched_vector_search", "required-features": ["experimental-batch-pool"]}
    if len(added) != 1 or added[0] not in [allowed, dict(allowed, path="examples/batched_vector_search.rs")]:
        raise ValueError("unexpected new example registration")
    current["example"] = [entry for entry in entries if entry.get("name") != "batched_vector_search"]
    if current != original:
        raise ValueError("Cargo dependencies, existing features/targets, or profiles changed")
    return {"feature": {"experimental-batch-pool": ["experimental"]}, "example": added[0], "all_other_parsed_manifest_entries_equal_baseline": True}


def validate_tests(path, enabled):
    receipt = json.loads(path.read_text())
    command = receipt.get("command", [])
    if receipt.get("returncode") != 0 or receipt.get("error") is not None:
        raise ValueError("test receipt does not record successful completion")
    if command[:2] != ["cargo", "test"] or "--lib" not in command or not any("qgraph" in arg for arg in command):
        raise ValueError("a focused qgraph library-test receipt is required")
    if "--all-features" in command or "--features" not in command:
        raise ValueError("test features must be explicit")
    features = set(command[command.index("--features") + 1].replace(",", " ").split())
    allowed = [{"experimental", "experimental-batch-pool"}, {"experimental-batch-pool"}] if enabled else [{"experimental"}]
    if features not in allowed:
        raise ValueError("test receipt uses the wrong feature mode")
    return receipt


def query_run(name, command, env, deadline, sampling=False):
    if "WARM_QGRAPH_BATCH_POOL" in env:
        raise RuntimeError("runtime batch-pool environment switch is forbidden in featurev3")
    start = time.monotonic()
    sampled_peak = 0
    sample = None
    error = None
    events = {}
    with (ROOT / f"{name}.stdout").open("x") as stdout, (ROOT / f"{name}.stderr").open("x") as stderr:
        process = subprocess.Popen(command, cwd=base.TREE, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            while process.poll() is None:
                sampled_peak = max(sampled_peak, monitor.attributed_rss_tree(process.pid))
                if sampled_peak > 2 * 1024 * 1024:
                    raise RuntimeError("2 GiB sampled native process-tree RSS guard exceeded")
                if time.monotonic() > min(deadline, start + 180):
                    raise TimeoutError("180s native/1200s aggregate wall guard exceeded")
                log = (ROOT / f"{name}.stderr").read_text()
                for marker in ["QUERY_STARTED", "QUERY_FINISHED"]:
                    if marker in log and marker not in events:
                        events[marker] = time.monotonic() - start
                if sampling and sample is None and "QUERY_STARTED" in events:
                    events["sample_launched"] = time.monotonic() - start
                    with (ROOT / f"{name}.sample-launch.log").open("x") as sample_log:
                        sample = subprocess.Popen(["/usr/bin/sample", str(process.pid), "5", "1", "-file", str(ROOT / f"{name}.sample.txt")], stdout=sample_log, stderr=subprocess.STDOUT, start_new_session=True)
                time.sleep(0.2)
            if process.returncode:
                raise RuntimeError(f"{name} exited {process.returncode}")
        except BaseException as caught:
            error = f"{type(caught).__name__}: {caught}"
            raise
        finally:
            base.stop(process)
            if sample is not None:
                try:
                    sample.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    base.stop(sample)
            base.save(ROOT / f"{name}.receipt.json", {"command": command, "pid": process.pid, "returncode": process.returncode, "elapsed_s": time.monotonic() - start, "sampled_summed_process_tree_rss_kib": sampled_peak, "sampled_rss_guard_gib": 2, "rss_is_not_peak_or_unique_memory": True, "sampler_process_rss_excluded": True, "marker_observations_s": events, "marker_resolution_s": 0.2, "error": error, "sampler_returncode": None if sample is None else sample.returncode})


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
    deadline = time.monotonic() + 1200
    query.ROOT = ROOT
    monitor.ROOT = ROOT
    native = json.loads((base.ROOT / "source-manifest.json").read_text())["native"]
    baseline_manifest = json.loads((BASELINE / "source-manifest.json").read_text())
    previous_complete = json.loads((PREVIOUS / "complete.json").read_text())
    previous_plan = json.loads((PREVIOUS / "plan.json").read_text())
    previous_compile = json.loads((PREVIOUS / "compile.receipt.json").read_text())
    for name in ["plan.json", "candidate-binary", "compile.receipt.json"]:
        if base.sha(PREVIOUS / name) != previous_complete["sha256"][name]:
            raise RuntimeError("preserved v2 evidence SHA mismatch")
    if previous_compile["returncode"] != 0 or previous_compile["error"] is not None:
        raise RuntimeError("preserved v2 compile did not succeed")
    validate_tests(args.disabled_tests_receipt, False)
    validate_tests(args.enabled_tests_receipt, True)
    verify_native(native, args.candidate_sha256)
    if base.sha(base.TREE / "Cargo.toml") != CARGO_SHA:
        raise RuntimeError("Cargo SHA differs from reviewed feature manifest")
    baseline_cargo = BASELINE / "harness/Cargo.toml"
    if base.sha(baseline_cargo) != baseline_manifest["harness"]["Cargo.toml"]:
        raise RuntimeError("archived baseline Cargo SHA mismatch")
    cargo_delta = validate_cargo((base.TREE / "Cargo.toml").read_bytes(), baseline_cargo.read_bytes())
    for name in ["Cargo.lock", "benchmarks/public/vector_glove_development.rs", "benchmarks/public/analyze_glove_development.py", "benchmarks/public/run_glove_development.py", "benchmarks/public/run_glove_local_diagnosis.py", "benchmarks/public/run_glove_prepared_diagnosis.py"]:
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
    frozen["tests-disabled.receipt.json"] = copy(args.disabled_tests_receipt, "tests-disabled.receipt.json")
    frozen["tests-enabled.receipt.json"] = copy(args.enabled_tests_receipt, "tests-enabled.receipt.json")
    frozen["baseline-Cargo.toml"] = copy(baseline_cargo, "baseline-Cargo.toml")
    frozen["baseline-source-manifest.json"] = copy(base.ROOT / "source-manifest.json", "baseline-source-manifest.json")
    frozen["baseline-harness-manifest.json"] = copy(BASELINE / "source-manifest.json", "baseline-harness-manifest.json")
    frozen["baseline.sample.txt"] = copy(BASELINE / "sampled.sample.txt", "baseline.sample.txt")
    frozen["baseline.sample.receipt.json"] = copy(BASELINE / "sampled.receipt.json", "baseline.sample.receipt.json")
    frozen["baseline-binary"] = copy(BASELINE / "vector-glove-development", "baseline-binary")
    frozen["v2-compile.receipt.json"] = copy(PREVIOUS / "compile.receipt.json", "v2-compile.receipt.json")
    frozen["v2-plan.json"] = copy(PREVIOUS / "plan.json", "v2-plan.json")
    rustc = subprocess.check_output(["rustc", "--version", "--verbose"], text=True)
    if rustc != baseline_manifest["rustc"]:
        raise RuntimeError("toolchain changed from baseline")
    env = {key: os.environ[key] for key in ["PATH", "HOME", "TMPDIR", "RUSTUP_HOME", "CARGO_HOME", "DEVELOPER_DIR", "SDKROOT"] if key in os.environ}
    env.update({"CARGO_TARGET_DIR": str(base.TREE / "target"), "CARGO_BUILD_JOBS": "1", "RAYON_NUM_THREADS": "1", "CARGO_PROFILE_RELEASE_DEBUG": "1", "CARGO_PROFILE_RELEASE_STRIP": "none", "WARM_QGRAPH_PREFETCH": "0", "WARM_QGRAPH_BATCH_EST": "0", "WARM_QGRAPH_FRONTIER": "1", "WARM_QGRAPH_BLOCK_SCREEN": "0", "WARM_QGRAPH_RERANK_Z": "0"})
    declared_env = {key: value for key, value in env.items() if key.startswith(("CARGO_", "RAYON_", "WARM_"))}
    plan = {
        "schema": "warm-glove-batch-pool-ablation-plan-v3", "development_only": True,
        "source": base.SOURCE, "candidate_vertex_sha256": args.candidate_sha256, "candidate_cargo_sha256": CARGO_SHA,
        "candidate_mechanism": "Same8-key batch merge behind nondefault experimental-batch-pool compile feature; feature0 restores oldPool/SearchKnobs layout and API, feature1 enables mechanism automatically; no runtime switch and no threshold tuning",
        "reason_for_isolation": "Final separately declared compile-feature attempt afterv2 failed defaultcontrol gates; preserve prior failures and park if this attempt fails, no automaticretry or additionalvariants",
        "baseline_binary_sha256": BASELINE_BINARY_SHA, "test_receipts_sha256": {"disabled": frozen["tests-disabled.receipt.json"], "enabled": frozen["tests-enabled.receipt.json"]},
        "review_scope": "Parent explicitly reviews exact native/Cargo SHA and both feature-mode qgraph test receipts before invocation; receipts alone are not independent source/test binding proof",
        "cargo_delta": cargo_delta,
        "input": str(query.INPUT), "index": str(query.INDEX),
        "input_declaration_sha256": base.sha(query.INPUT / "declaration.json"), "index_sha256": base.sha(query.INDEX),
        "block_orders": ORDERS, "efs": [10, 20, 40, 80, 120, 160, 240, 400, 640], "repeats_per_process": 7,
        "k": 10, "rerank": 64, "seed": 539365412, "knob_mode": "env",
        "arms": {arm: {"binary": f"{arm}-binary", "features": FEATURES[arm]} for arm in ["baseline", "disabled", "enabled"]},
        "compile_order": ["disabled", "enabled"], "compile_switch": "--features experimental versus --features experimental,experimental-batch-pool; copy disabled binary before enabled compilation",
        "sample_after_timings": {"arm": "enabled", "ef": 240, "repeats": 1000, "duration_s": 5, "interval_ms": 1},
        "primary_gate": {"ef": 80, "recall_floor": 0.95, "minimum_geomean_qps_ratio": 1.10, "bootstrap_lower_strictly_above": 1.0, "contrasts": ["enabled_over_baseline", "enabled_over_disabled"]},
        "default_control_gate": {"ef": 80, "contrast": "disabled_over_baseline", "minimum_geomean_qps_ratio": 0.99, "bootstrap_lower_at_least": 0.98},
        "secondary_default_control_gate": {"contrast": "disabled_over_baseline", "efs": [10, 20, 40, 80, 120, 160, 240, 400, 640], "minimum_geomean_qps_ratio": 0.98},
        "semantic_gate": "All ordered IDs and native-reported distance evaluations match exactS across every ef,query,repeat,process. Baseline/disabled timing equality is reported, not assumed.",
        "causal_gate": "Review query-only sampled call-site contrast consistent with bulkmerge mechanism; no claim from speed alone. Samples excluded from timed comparisons.",
        "uncertainty": "Collapse7 repeats per fresh process, paired whole-process-block bootstrap enumerating27 resamples;3blocks lowpower, conditionalhardware only. No query/repeat pseudoreplication.",
        "limits": {"per_compile_s": 600, "native_s": 180, "overall_s": 1200, "compile_sampled_tree_rss_gib": 4, "native_sampled_tree_rss_gib": 2},
        "memory_scope": "Approximately200ms sampled summed child-tree RSS, not strict OS peak or unique memory; controller/source preparation/sampler excluded",
        "lock": str(Path.home() / "warm-timing.lock"), "environment": declared_env,
        "official_test_arrays_read": False, "competitive_or_leaderboard_claim": False,
        "rustc": rustc, "hardware": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize", "hw.ncpu"], text=True),
        "previous_candidate": {"root": str(PREVIOUS), "vertex_sha256": previous_plan["candidate_vertex_sha256"], "binary_sha256": previous_complete["sha256"]["candidate-binary"], "binary_bytes": (PREVIOUS / "candidate-binary").stat().st_size, "compile_receipt_sha256": frozen["v2-compile.receipt.json"], "compile_elapsed_s": previous_compile["elapsed_s"], "compile_sampled_summed_process_tree_rss_kib": previous_compile["sampled_summed_process_tree_rss_kib"]},
        "compile_cost_scope": "Measuredv2 andv3 feature0/feature1 compile elapsed time, sampled summed child-tree RSS and binarybytes; builds share private target caches and are not matched clean builds. Code growth is a cost, not a success claim.",
        "frozen_source_and_harness_sha256": frozen,
    }
    base.save(ROOT / "plan.json", plan)
    lock = (Path.home() / "warm-timing.lock").open("a+")
    base.fcntl.flock(lock, base.fcntl.LOCK_EX | base.fcntl.LOCK_NB)
    try:
        with Path("/Volumes/A/warm-coordination/RUN-ALL.md").open("a") as coordination:
            coordination.write(f"\n{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} Vector batch-pool featurev3 owner acquired local timing flock; plan {ROOT / 'plan.json'},final separately declared compile-feature attempt afterv2 defaultcontrol failure. Same8-key mechanism, two serialized candidate builds then3balanced blocks/full9efcurves/7repeats and separate sample.1compilejob/1Rayon,4GiBcompile/2GiBnative sampledRSS,600spercompile/180snative/1200soverall. No runtimebatchpoolflag/finalarrays/remoteaction/lexical work.\n")
        compile_cost = {"v2": plan["previous_candidate"], "scope": plan["compile_cost_scope"]}
        compile_runs = {}
        for arm in ["disabled", "enabled"]:
            verify_native(native, args.candidate_sha256)
            if any(base.sha(base.TREE / name) != frozen[f"harness/{name}"] for name in FILES):
                raise RuntimeError("frozen Cargo/harness changed before compilation")
            command = ["cargo", "build", "--offline", "--locked", "--release", "--features", FEATURES[arm], "--bin", "vector-glove-development"]
            started = time.monotonic()
            monitor.run(f"compile-{arm}", command, env, 600, deadline, 4)
            verify_native(native, args.candidate_sha256)
            if any(base.sha(base.TREE / name) != frozen[f"harness/{name}"] for name in FILES):
                raise RuntimeError("frozen Cargo/harness changed during compilation")
            binary_name = f"{arm}-binary"
            frozen[binary_name] = copy(base.TREE / "target/release/vector-glove-development", binary_name)
            receipt_name = f"compile-{arm}.receipt.json"
            receipt = json.loads((ROOT / receipt_name).read_text())
            compile_runs[arm] = {"command": command, "features": FEATURES[arm], "started_monotonic_s": started, "binary_copied_monotonic_s": time.monotonic(), "binary_sha256": frozen[binary_name], "vertex_sha256": args.candidate_sha256, "cargo_sha256": CARGO_SHA}
            compile_cost[f"v3_{arm}"] = {"vertex_sha256": args.candidate_sha256, "binary_sha256": frozen[binary_name], "binary_bytes": (ROOT / binary_name).stat().st_size, "compile_receipt_sha256": base.sha(ROOT / receipt_name), "compile_elapsed_s": receipt["elapsed_s"], "compile_sampled_summed_process_tree_rss_kib": receipt["sampled_summed_process_tree_rss_kib"]}
        timed_runs = {}
        for block, order in enumerate(ORDERS):
            for position, arm in enumerate(order):
                name = f"block{block}-{arm}"
                settings = plan["arms"][arm]
                arm_env = dict(env)
                command = [str(ROOT / settings["binary"]), str(query.INPUT), str(query.INDEX), str(ROOT / f"{name}.json"), "--knobs", "env"]
                started = time.monotonic()
                query_run(name, command, arm_env, deadline)
                timed_runs[name] = {"block": block, "position": position, "arm": arm, "command": command, "started_monotonic_s": started, "finished_monotonic_s": time.monotonic(), "environment": dict(declared_env), "compile_features": settings["features"], "binary_sha256": frozen[settings["binary"]], "result_sha256": base.sha(ROOT / f"{name}.json")}
        query_run("candidate-sampled", [str(ROOT / "enabled-binary"), str(query.INPUT), str(query.INDEX), str(ROOT / "candidate-sampled.json"), "--knobs", "env", "--efs", "240", "--repeats", "1000"], dict(env), deadline, sampling=True)
        profile = {name: {"path": name, "sha256": base.sha(ROOT / name)} for name in ["baseline.sample.txt", "baseline.sample.receipt.json", "candidate-sampled.sample.txt", "candidate-sampled.receipt.json", "candidate-sampled.json"] if (ROOT / name).exists()}
        base.save(ROOT / "execution-manifest.json", {"plan_sha256": base.sha(ROOT / "plan.json"), "frozen_files": frozen, "compile_runs": compile_runs, "timed_runs": timed_runs, "profile_evidence": profile, "compile_cost": compile_cost})
        monitor.run("analyze", [sys.executable, str(base.TREE / "benchmarks/public/analyze_glove_batch_pool_feature.py"), "--root", str(ROOT), "--out", str(ROOT / "analysis.json")], env, 180, deadline, 2)
        base.save(ROOT / "complete.json", {"status": "completed-development-feature-v3", "acceptance": json.loads((ROOT / "analysis.json").read_text())["primary"], "competitive_claim": False, "new_paid_compute_usd": 0, "final_load": os.getloadavg(), "sha256": {str(path.relative_to(ROOT)): base.sha(path) for path in ROOT.rglob("*") if path.is_file()}})
        print(json.dumps({"status": "completed-development-feature-v3", "root": str(ROOT)}))
    finally:
        base.fcntl.flock(lock, base.fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--disabled-tests-receipt", type=Path, required=True)
    parser.add_argument("--enabled-tests-receipt", type=Path, required=True)
    args = parser.parse_args()
    if re.fullmatch("[0-9a-f]{64}", args.candidate_sha256) is None:
        parser.error("candidate SHA256 must be64lowercase hexadecimal characters")
    if args.candidate_sha256 != CANDIDATE_SHA:
        parser.error("candidate SHA256 differs from the declared feature-v3 source")
    if ROOT.exists():
        raise SystemExit("Run directory already exists; preserve it and declare a new attempt")
    try:
        main(args)
    except BaseException as error:
        if ROOT.exists():
            base.save(ROOT / "failure.json", {"error": f"{type(error).__name__}: {error}", "automatic_retry": False})
        raise
