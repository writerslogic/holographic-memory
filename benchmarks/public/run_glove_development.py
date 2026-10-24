import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import run_glove_local_diagnosis as base
import run_glove_prepared_diagnosis as monitor


ROOT = Path("/Volumes/A/hms-coordination/benchmark-improvement-20261024/vector-glove-query-diagnosis-v1")
INPUT = base.ROOT / "input"
INDEX = base.ROOT.with_name("vector-glove-local-diagnosis-attempt5") / "plain.hms"
FILES = ["Cargo.toml", "Cargo.lock", "benchmarks/public/vector_glove_development.rs", "benchmarks/public/analyze_glove_development.py", "benchmarks/public/run_glove_development.py", "benchmarks/public/run_glove_local_diagnosis.py", "benchmarks/public/run_glove_prepared_diagnosis.py"]


def query_run(name, command, env, deadline, sampling=False):
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
                    raise TimeoutError("180s native/900s aggregate wall guard exceeded")
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


def main():
    os.umask(0o077)
    ROOT.mkdir()
    monitor.ROOT = ROOT
    plan = {
        "scope": "Exact S train-only GloVe query diagnosis; no candidate, competitor or final evaluation",
        "source": base.SOURCE, "input": str(INPUT), "index": str(INDEX),
        "input_declaration_sha256": base.sha(INPUT / "declaration.json"), "index_sha256": base.sha(INDEX),
        "sampled_run": {"efs": [240], "repeats": 100, "sampling": "5 seconds at1ms after QUERY_STARTED; native timings contaminated by sampler and are not speed evidence"},
        "unsampled_run": {"efs": [10, 20, 40, 80, 120, 160, 240, 400, 640], "repeats": 7},
        "k": 10, "rerank": 64, "knobs": "SearchKnobs::OFF", "rayon_threads": 1, "build_jobs": 1,
        "limits": {"compile_s": 600, "native_s": 180, "overall_s": 900, "compile_sampled_tree_rss_gib": 4, "native_sampled_tree_rss_gib": 2},
        "memory_scope": "Sampled child-tree RSS approximately200ms, not peak/unique memory or strict OS enforcement; controller/preparation/sampler excluded",
        "test_scope": "Only new harness binary tests; CI opt1/debug0 test profile; build+test command shares compile guard",
        "lock": str(Path.home() / "hms-timing.lock"), "official_test_arrays_read": False,
        "timing_interpretation": "Within-process repeated development queries only. Separate-process/hardware uncertainty absent. Exact truth uses exhaustive f64 cosine from original f32 rows.",
    }
    base.save(ROOT / "plan.json", plan)
    archive = ROOT / "harness"
    archive.mkdir()
    for name in FILES:
        target = archive / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base.TREE / name, target)
    lock = (Path.home() / "hms-timing.lock").open("a+")
    base.fcntl.flock(lock, base.fcntl.LOCK_EX | base.fcntl.LOCK_NB)
    try:
        deadline = time.monotonic() + 900
        with Path("/Volumes/A/hms-coordination/RUN-ALL.md").open("a") as coordination:
            coordination.write(f"\n{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} Vector development harness owner acquired local timing flock for query-only exactS diagnosis; plan {ROOT / 'plan.json'}. Existing train-only fixture/attempt5 serialized index; one compile job/Rayon thread,4GiB compile and2GiB native sampledRSS,600scompile/180snative/900stotal. No candidate, official arrays, remote action or lexical work.\n")
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base.TREE, text=True).strip() != base.SOURCE:
            raise RuntimeError("wrong source revision")
        if subprocess.check_output(["git", "diff", "--", "src", "Cargo.lock"], cwd=base.TREE):
            raise RuntimeError("native baseline changed")
        original = json.loads((base.ROOT / "source-manifest.json").read_text())
        if any(base.sha(base.TREE / name) != value for name, value in original["native"].items()):
            raise RuntimeError("native source SHA mismatch")
        declaration = json.loads((INPUT / "declaration.json").read_text())
        for name, value in declaration["files"].items():
            if Path(name).name != name or base.sha(INPUT / name) != value:
                raise ValueError("prepared input SHA mismatch")
        env = {key: os.environ[key] for key in ["PATH", "HOME", "TMPDIR", "RUSTUP_HOME", "CARGO_HOME", "DEVELOPER_DIR", "SDKROOT"] if key in os.environ}
        env.update({"CARGO_TARGET_DIR": str(base.TREE / "target"), "CARGO_BUILD_JOBS": "1", "RAYON_NUM_THREADS": "1", "CARGO_PROFILE_RELEASE_DEBUG": "1", "CARGO_PROFILE_RELEASE_STRIP": "none"})
        base.save(ROOT / "source-manifest.json", {"harness": {name: base.sha(base.TREE / name) for name in FILES}, "native_manifest_sha256": base.sha(base.ROOT / "source-manifest.json"), "native_hashes_verified": True, "rustc": subprocess.check_output(["rustc", "--version", "--verbose"], text=True), "hardware": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize", "hw.ncpu"], text=True), "load_before": os.getloadavg(), "python": sys.version, "environment": {key: value for key, value in env.items() if key.startswith(("CARGO_", "RAYON_"))}})
        monitor.run("compile", ["cargo", "build", "--offline", "--locked", "--release", "--features", "experimental", "--bin", "vector-glove-development"], env, 600, deadline, 4)
        binary = ROOT / "vector-glove-development"
        shutil.copy2(base.TREE / "target/release/vector-glove-development", binary)
        base.save(ROOT / "binary.json", {"sha256": base.sha(binary), "path": str(binary)})
        query_run("sampled", [str(binary), str(INPUT), str(INDEX), str(ROOT / "sampled.json"), "--efs", "240", "--repeats", "100"], env, deadline, sampling=True)
        query_run("curve", [str(binary), str(INPUT), str(INDEX), str(ROOT / "curve.json")], env, deadline)
        monitor.run("analyze", [sys.executable, str(base.TREE / "benchmarks/public/analyze_glove_development.py"), "--baseline", str(ROOT / "curve.json"), "--out", str(ROOT / "analysis.json")], env, 180, deadline, 2)
        test_env = dict(env, CARGO_PROFILE_TEST_OPT_LEVEL="1", CARGO_PROFILE_TEST_DEBUG="0", CARGO_PROFILE_DEV_OPT_LEVEL="1", CARGO_PROFILE_DEV_DEBUG="0")
        monitor.run("tests", ["cargo", "test", "--offline", "--locked", "--features", "experimental", "--bin", "vector-glove-development"], test_env, 600, deadline, 4)
        base.save(ROOT / "complete.json", {"status": "completed-baseline-query-diagnosis", "native_source": base.SOURCE, "competitive_claim": False, "new_paid_compute_usd": 0, "final_load": os.getloadavg(), "sha256": {str(path.relative_to(ROOT)): base.sha(path) for path in ROOT.rglob("*") if path.is_file()}})
        print(json.dumps({"status": "completed-baseline-query-diagnosis", "root": str(ROOT)}))
    finally:
        base.fcntl.flock(lock, base.fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    if ROOT.exists():
        raise SystemExit("Run directory already exists; preserve it and declare a new attempt")
    try:
        main()
    except BaseException as error:
        if ROOT.exists():
            base.save(ROOT / "failure.json", {"error": f"{type(error).__name__}: {error}", "automatic_retry": False})
        raise
