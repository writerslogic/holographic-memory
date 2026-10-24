import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import run_glove_local_diagnosis as diagnosis


PREVIOUS = diagnosis.ROOT
ROOT = None


def attributed_rss_tree(pid):
    output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss=,comm="], text=True)
    rows = []
    for line in output.splitlines():
        process, parent, rss, command = line.split(maxsplit=3)
        rows.append((int(process), int(parent), int(rss), command))
    owned = {pid}
    while True:
        expanded = owned | {process for process, parent, _, _ in rows if parent in owned}
        if expanded == owned:
            break
        owned = expanded
    children = [(process, rss, command) for process, _, rss, command in rows if process in owned]
    total = sum(rss for _, rss, _ in children)
    with (ROOT / "memory-samples.jsonl").open("a") as stream:
        stream.write(json.dumps({"monotonic_s": time.monotonic(), "root_pid": pid, "summed_rss_kib": total, "processes": children}) + "\n")
    return total


def run(name, command, env, seconds, deadline, memory_gib, sampling=False):
    start = time.monotonic()
    sample = None
    sampled_peak = 0
    error = None
    with (ROOT / f"{name}.stdout").open("x") as stdout, (ROOT / f"{name}.stderr").open("x") as stderr:
        process = subprocess.Popen(command, cwd=diagnosis.TREE, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            while process.poll() is None:
                rss = attributed_rss_tree(process.pid)
                sampled_peak = max(sampled_peak, rss)
                if rss > memory_gib * 1024 * 1024:
                    raise RuntimeError(f"{memory_gib} GiB sampled process-tree RSS guard exceeded")
                if time.monotonic() > min(deadline, start + seconds):
                    raise TimeoutError("declared diagnostic wall-time limit exceeded")
                if sampling and sample is None and "BUILD_STARTED" in (ROOT / f"{name}.stderr").read_text():
                    with (ROOT / "sample-launch.log").open("x") as sample_log:
                        sample = subprocess.Popen(["/usr/bin/sample", str(process.pid), "5", "1", "-file", str(ROOT / "build.sample.txt")], stdout=sample_log, stderr=subprocess.STDOUT, start_new_session=True)
                time.sleep(0.2)
            if process.returncode:
                raise RuntimeError(f"{name} exited {process.returncode}")
        except BaseException as caught:
            error = f"{type(caught).__name__}: {caught}"
            raise
        finally:
            diagnosis.stop(process)
            if sample is not None:
                try:
                    sample.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    diagnosis.stop(sample)
            diagnosis.save(ROOT / f"{name}.receipt.json", {
                "command": command, "pid": process.pid, "returncode": process.returncode,
                "elapsed_s": time.monotonic() - start,
                "sampled_summed_process_tree_rss_kib": sampled_peak,
                "sampled_rss_guard_gib": memory_gib,
                "rss_is_not_peak_or_unique_memory": True,
                "sampler_process_rss_excluded": True,
                "error": error, "sampler_returncode": None if sample is None else sample.returncode,
            })


def main():
    os.umask(0o077)
    ROOT.mkdir()
    diagnosis.ROOT = ROOT
    plan = json.loads((PREVIOUS / "plan.json").read_text())
    plan.update({
        "attempt": int(ROOT.name.rsplit("attempt", 1)[1]),
        "build_jobs": 1,
        "prepared_input": str(PREVIOUS / "input"),
        "previous_attempt": str(PREVIOUS),
        "compile_hypothesis": "Attempt4 excluded dev-dependencies but one rustc still reached2001248KiB; explicitly allow4GiB during compilation on32GiB host, keeping one job and release code generation/symbol settings unchanged",
        "compile_cache": "Reuse private target containing incomplete attempts1/3/4 artifacts; compile time is cache-conditioned",
        "compile_target": "experimental-only binary includes the same diagnostic example",
        "memory_limit_scope": "Sampled sum of child process-tree RSS every approximately 200ms; not strict peak or unique memory; preparation and controller excluded",
    })
    plan["limits"].pop("sampled_tree_rss_gib")
    plan["limits"].update({"compile_sampled_tree_rss_gib": 4, "native_sampled_tree_rss_gib": 2})
    diagnosis.save(ROOT / "plan.json", plan)
    archive = ROOT / "harness"
    archive.mkdir()
    for name in ["Cargo.toml", "Cargo.lock", "examples/vector_glove_diagnosis.rs", "benchmarks/public/run_glove_local_diagnosis.py", "benchmarks/public/run_glove_prepared_diagnosis.py", "benchmarks/public/vector_glove_diagnosis.rs"]:
        target = archive / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(diagnosis.TREE / name, target)
    diagnosis.rss_tree = attributed_rss_tree
    lock = (Path.home() / "hms-timing.lock").open("a+")
    diagnosis.fcntl.flock(lock, diagnosis.fcntl.LOCK_EX | diagnosis.fcntl.LOCK_NB)
    try:
        with Path("/Volumes/A/hms-coordination/RUN-ALL.md").open("a") as coordination:
            coordination.write(f"\n{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} Vector-only attempt{plan['attempt']} acquired local timing flock; plan {ROOT / 'plan.json'}, prepared attempt1 inputs reused by hash, one compile job/one Rayon thread,600s compile/180s pernative/900s total; explicit4GiB compile/unchanged2GiB native sampled child-tree RSS guards. Attempt4 single-rustc guard failure preserved;32GiB host. Baseline diagnosis only, no remote action or lexical work.\n")
        deadline = time.monotonic() + 900
        tree = diagnosis.TREE
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tree, text=True).strip() != diagnosis.SOURCE:
            raise RuntimeError("wrong baseline")
        if subprocess.check_output(["git", "diff", "--", "src", "Cargo.lock"], cwd=tree):
            raise RuntimeError("production source changed")
        data = PREVIOUS / "input"
        declaration = json.loads((data / "declaration.json").read_text())
        for name, expected in declaration["files"].items():
            if Path(name).name != name or diagnosis.sha(data / name) != expected:
                raise ValueError("prepared input hash mismatch")
        original = json.loads((PREVIOUS / "source-manifest.json").read_text())
        for name, expected in original["native"].items():
            if diagnosis.sha(tree / name) != expected:
                raise ValueError("baseline source hash mismatch")
        diagnosis.save(ROOT / "source-manifest.json", {
            "native_manifest_sha256": diagnosis.sha(PREVIOUS / "source-manifest.json"),
            "native_hashes_verified": True,
            "prepared_declaration_sha256": diagnosis.sha(data / "declaration.json"),
            "harness": {name: diagnosis.sha(tree / name) for name in ["Cargo.toml", "Cargo.lock", "examples/vector_glove_diagnosis.rs", "benchmarks/public/run_glove_local_diagnosis.py", "benchmarks/public/run_glove_prepared_diagnosis.py", "benchmarks/public/vector_glove_diagnosis.rs"]},
            "rustc": subprocess.check_output(["rustc", "--version", "--verbose"], text=True),
            "hardware": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize", "hw.ncpu"], text=True),
            "load_before": os.getloadavg(), "python": sys.version,
        })
        env = {key: os.environ[key] for key in ["PATH", "HOME", "TMPDIR", "RUSTUP_HOME", "CARGO_HOME", "DEVELOPER_DIR", "SDKROOT"] if key in os.environ}
        env.update({"CARGO_TARGET_DIR": str(tree / "target"), "CARGO_BUILD_JOBS": "1", "RAYON_NUM_THREADS": "1", "CARGO_PROFILE_RELEASE_DEBUG": "1", "CARGO_PROFILE_RELEASE_STRIP": "none"})
        run("compile", ["cargo", "build", "--verbose", "--offline", "--locked", "--release", "--features", "experimental", "--bin", "vector-glove-diagnosis"], env, 600, deadline, 4)
        binary = tree / "target/release/vector-glove-diagnosis"
        for mode in ["plain", "profiled"]:
            child_env = dict(env)
            if mode == "profiled":
                child_env["HMS_QGRAPH_BUILD_PROFILE"] = "1"
            run(mode, [str(binary), str(data), str(ROOT / f"{mode}.json"), mode], child_env, 180, deadline, 2, sampling=mode == "profiled")
        a, b = [json.loads((ROOT / f"{mode}.json").read_text()) for mode in ["plain", "profiled"]]
        if a["graph_hash"] != b["graph_hash"] or diagnosis.sha(ROOT / "plain.hms") != diagnosis.sha(ROOT / "profiled.hms"):
            raise ValueError("graph/index changed under profiling")
        if [(r["ef"], r["ids"]) for r in a["settings"]] != [(r["ef"], r["ids"]) for r in b["settings"]]:
            raise ValueError("ordered query IDs changed under profiling")
        diagnosis.save(ROOT / "complete.json", {"status": "completed-baseline-diagnosis", "native_source": diagnosis.SOURCE, "graph_index_and_ids_equal": True, "binary_sha256": diagnosis.sha(binary), "final_load": os.getloadavg(), "competitive_claim": False, "new_paid_compute_usd": 0})
        print(json.dumps({"status": "completed-baseline-diagnosis", "root": str(ROOT)}))
    finally:
        diagnosis.fcntl.flock(lock, diagnosis.fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    if args.attempt < 2:
        parser.error("prepared attempts start at 2")
    ROOT = PREVIOUS.with_name(f"vector-glove-local-diagnosis-attempt{args.attempt}")
    if ROOT.exists():
        raise SystemExit("Run directory already exists; preserve it and declare a new attempt path")
    try:
        main()
    except BaseException as error:
        if ROOT.exists():
            diagnosis.save(ROOT / "failure.json", {"error": f"{type(error).__name__}: {error}", "automatic_retry": False})
        raise
