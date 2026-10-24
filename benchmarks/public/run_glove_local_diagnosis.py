import fcntl
import hashlib
import json
import mmap
import os
from pathlib import Path
import random
import signal
import struct
import subprocess
import sys
import time


SOURCE = "12c80decb094d57ee13ba187a01bf1de8788ca2f"
TREE = Path(__file__).resolve().parents[2]
ROOT = Path("/Volumes/A/hms-coordination/benchmark-improvement-20261024/vector-glove-local-diagnosis")
CACHE = Path("/Volumes/A/.hms-bench-cache/glove-100-angular")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def rss_tree(pid):
    output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="], text=True)
    rows = [list(map(int, line.split())) for line in output.splitlines()]
    owned = {pid}
    while True:
        expanded = owned | {p for p, parent, _ in rows if parent in owned}
        if expanded == owned:
            break
        owned = expanded
    return sum(rss for p, _, rss in rows if p in owned)


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run(name, command, env, seconds, deadline, sampling=False):
    start = time.monotonic()
    sample = None
    sampled_peak = 0
    error = None
    with (ROOT / f"{name}.stdout").open("x") as stdout, (ROOT / f"{name}.stderr").open("x") as stderr:
        process = subprocess.Popen(command, cwd=TREE, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            while process.poll() is None:
                rss = rss_tree(process.pid)
                sampled_peak = max(sampled_peak, rss)
                if rss > 2 * 1024 * 1024:
                    raise RuntimeError("2 GiB sampled process-tree RSS guard exceeded")
                if time.monotonic() > min(deadline, start + seconds):
                    raise TimeoutError("declared diagnostic wall-time limit exceeded")
                if sampling and sample is None and "BUILD_STARTED" in (ROOT / f"{name}.stderr").read_text():
                    sample_log = (ROOT / "sample-launch.log").open("x")
                    try:
                        sample = subprocess.Popen(["/usr/bin/sample", str(process.pid), "5", "1", "-file", str(ROOT / "build.sample.txt")], stdout=sample_log, stderr=subprocess.STDOUT, start_new_session=True)
                    finally:
                        sample_log.close()
                time.sleep(0.2)
            if process.returncode:
                raise RuntimeError(f"{name} exited {process.returncode}")
        except BaseException as caught:
            error = f"{type(caught).__name__}: {caught}"
            raise
        finally:
            stop(process)
            if sample is not None:
                try:
                    sample.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    stop(sample)
            save(ROOT / f"{name}.receipt.json", {
                "command": command, "pid": process.pid, "returncode": process.returncode,
                "elapsed_s": time.monotonic() - start,
                "sampled_summed_process_tree_rss_kib": sampled_peak,
                "rss_is_not_peak_or_unique_memory": True,
                "error": error, "sampler_returncode": None if sample is None else sample.returncode,
            })


def main():
    os.umask(0o077)
    ROOT.mkdir()
    plan = {
        "source": SOURCE, "scope": "Local train-only baseline diagnosis, no candidate or competitive result",
        "authority": "Current user request permits independent diagnosis; no paid or remote admission inferred",
        "construction_count": 65536, "development_count": 256, "seed": 539365412,
        "sampling": "CPython random.Random(seed).sample(range(1183514),65792); retain exact IDs",
        "degree": 64, "build_ef": 500, "code_bits": 8, "residual_bits": 8,
        "query_efs": [80, 240, 640], "rerank": 64, "k": 10,
        "rayon_threads": 1, "build_jobs": 2,
        "order": ["plain", "profiled"], "cpu_sample": "one 5-second 1ms sample during profiled construction",
        "limits": {"compile_s": 600, "native_per_process_s": 180, "overall_s": 900, "sampled_tree_rss_gib": 2},
        "lock": str(Path.home() / "hms-timing.lock"),
        "official_test_arrays_read": False,
        "provenance_limit": "Cached train export is newly SHA-bound; meta names official HDF5 but raw export equality to its train array has not been independently verified here",
        "hypothesis": "Construction greedy search remains dominant; profiler must distinguish pool maintenance, distance kernels and adjacency/visited work before a native change",
        "rejection_rule": "No optimization from a broad phase timer alone; require sampled call-site evidence and exact identity of graph/index/ordered IDs across plain/profiled runs",
        "final_evaluation_authorized": False,
    }
    save(ROOT / "plan.json", plan)
    lock = (Path.home() / "hms-timing.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with Path("/Volumes/A/hms-coordination/RUN-ALL.md").open("a") as coordination:
            coordination.write(f"\n{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} Vector-only owner: local native timing lock acquired for train-only GloVe baseline diagnosis in {TREE}; exact S production sources, 65536 construction/256 disjoint development rows, 2 compile jobs then 1 Rayon thread serially, 900s overall/2GiB sampled process-tree RSS guard. Plan {ROOT / 'plan.json'}. No official test access, candidate, remote host action or suspended-service launch. Lexical/text ownership remains external.\n")
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TREE, text=True).strip() != SOURCE:
            raise RuntimeError("wrong baseline")
        if subprocess.check_output(["git", "diff", "--", "src", "Cargo.lock"], cwd=TREE):
            raise RuntimeError("production source changed")
        deadline = time.monotonic() + 900
        data = ROOT / "input"
        data.mkdir()
        meta = json.loads((CACHE / "meta.json").read_text())
        if (meta["n_train"], meta["dim"], meta["metric"]) != (1183514, 100, "angular"):
            raise ValueError("source metadata mismatch")
        ids = random.Random(plan["seed"]).sample(range(1183514), 65536 + 256)
        if len(set(ids)) != 65536 + 256:
            raise ValueError("partition overlap")
        train = CACHE / "train.f32"
        if train.stat().st_size != 1183514 * 100 * 4:
            raise ValueError("wrong source byte size")
        with train.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            for name, selected in [("construction", ids[:65536]), ("development", ids[65536:])]:
                (data / f"{name}.ids.u32").write_bytes(struct.pack(f"<{len(selected)}I", *selected))
                with (data / f"{name}.f32").open("xb") as output:
                    for index in selected:
                        output.write(mapped[index * 400:(index + 1) * 400])
        save(data / "declaration.json", {
            "construction_count": 65536, "development_count": 256,
            "official_test_arrays_read": False, "construction_development_overlap": 0,
            "source_sha256": sha(train), "source_meta_sha256": sha(CACHE / "meta.json"),
            "plan_sha256": sha(ROOT / "plan.json"),
            "files": {p.name: sha(p) for p in data.iterdir()},
        })
        source_files = subprocess.check_output(["git", "ls-files", "src"], cwd=TREE, text=True).splitlines()
        save(ROOT / "source-manifest.json", {
            "native": {name: sha(TREE / name) for name in source_files},
            "harness": {name: sha(TREE / name) for name in ["Cargo.toml", "Cargo.lock", "examples/vector_glove_diagnosis.rs", "benchmarks/public/run_glove_local_diagnosis.py"]},
            "rustc": subprocess.check_output(["rustc", "--version", "--verbose"], text=True),
            "hardware": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize", "hw.ncpu"], text=True),
            "load_before": os.getloadavg(), "python": sys.version,
        })
        env = {key: os.environ[key] for key in ["PATH", "HOME", "TMPDIR", "RUSTUP_HOME", "CARGO_HOME", "DEVELOPER_DIR", "SDKROOT"] if key in os.environ}
        env.update({"CARGO_TARGET_DIR": str(TREE / "target"), "CARGO_BUILD_JOBS": "2", "RAYON_NUM_THREADS": "1", "CARGO_PROFILE_RELEASE_DEBUG": "1", "CARGO_PROFILE_RELEASE_STRIP": "none"})
        run("compile", ["cargo", "build", "--offline", "--locked", "--release", "--features", "experimental", "--example", "vector_glove_diagnosis"], env, 600, deadline)
        binary = TREE / "target/release/examples/vector_glove_diagnosis"
        for mode in ["plain", "profiled"]:
            child_env = dict(env)
            if mode == "profiled":
                child_env["HMS_QGRAPH_BUILD_PROFILE"] = "1"
            run(mode, [str(binary), str(data), str(ROOT / f"{mode}.json"), mode], child_env, 180, deadline, sampling=mode == "profiled")
        a, b = [json.loads((ROOT / f"{mode}.json").read_text()) for mode in ["plain", "profiled"]]
        if a["graph_hash"] != b["graph_hash"] or sha(ROOT / "plain.hms") != sha(ROOT / "profiled.hms"):
            raise ValueError("graph/index changed under profiling")
        if [(r["ef"], r["ids"]) for r in a["settings"]] != [(r["ef"], r["ids"]) for r in b["settings"]]:
            raise ValueError("ordered query IDs changed under profiling")
        save(ROOT / "complete.json", {"status": "completed-baseline-diagnosis", "native_source": SOURCE, "graph_index_and_ids_equal": True, "binary_sha256": sha(binary), "final_load": os.getloadavg(), "competitive_claim": False, "new_paid_compute_usd": 0})
        print(json.dumps({"status": "completed-baseline-diagnosis", "root": str(ROOT)}))
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    if ROOT.exists():
        raise SystemExit("Run directory already exists; preserve it and declare a new attempt path")
    try:
        main()
    except BaseException as error:
        if ROOT.exists():
            save(ROOT / "failure.json", {"error": f"{type(error).__name__}: {error}", "automatic_retry": False})
        raise
