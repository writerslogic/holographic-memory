"""Run the ann-benchmarks HMS grid and one matched competitor family per dataset."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import modal

SOURCE_REV = "12c80decb094d57ee13ba187a01bf1de8788ca2f"
HARNESS_REV = "2e081ad32c1eccab72dcb739ad886c310b90f715"
GLASS_REV = "4bbb0d81f79341cea615ee74f0863f3755ff16c4"
DATASETS = {
    "glove-100-angular": ("544af1d5e84e112cd4749571dcfd8ca109818a572f850af75a3a09e093a953c4", "scann"),
    "sift-128-euclidean": ("dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984", "faiss-ivfpqfs"),
    "fashion-mnist-784-euclidean": ("7f24e5122b71346346686a6ea80e08d9145b9c020f9f2db4c4c799dfcfc96a07", "glass"),
}
RUN_ROOT = Path("/vol/ann-live-20261024")
CPU = 8
MEM_GIB = 32
USD_PER_HOUR = CPU * 0.0473 + MEM_GIB * 0.008
VOL = modal.Volume.from_name("hms-ann-live-20261024", create_if_missing=True)
app = modal.App("hms-ann-live-20261024")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("build-essential", "git", "curl", "pkg-config", "libgomp1", "util-linux", "time")
    .pip_install("uv==0.12.13")
    .run_commands(
        f"git clone -q --filter=blob:none https://github.com/erikbern/ann-benchmarks.git /opt/ann && git -C /opt/ann checkout -q {HARNESS_REV}",
        f"git clone -q --filter=blob:none https://github.com/writerslogic/holographic-memory.git /opt/hms && git -C /opt/hms checkout -q {SOURCE_REV}",
        "uv pip install --system -r /opt/ann/requirements.txt maturin==1.15.0 scann==1.4.2 faiss-cpu==1.15.1 pybind11==2.10.4 setuptools",
        "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.96.0",
        "cd /opt/hms && . /root/.cargo/env && CARGO_BUILD_JOBS=8 maturin build --release --locked --features python --out /opt/wheels",
        "uv pip install --system --reinstall /opt/wheels/*.whl",
        "cp -R /opt/hms/benchmarks/ann-benchmarks/hms /opt/ann/ann_benchmarks/algorithms/hms",
    )
)


def run_command(args: list[str], *, cwd: Path, log: Path, deadline: float, env: dict[str, str]) -> int:
    remaining = min(7200, max(1, int(deadline - time.monotonic())))
    with log.open("a") as output:
        output.write(json.dumps({"command": args, "started": time.time(), "timeout_s": remaining}) + "\n")
        output.flush()
        process = subprocess.Popen(args, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            output.write(json.dumps({"timeout": True, "ended": time.time()}) + "\n")
            raise
        output.write(json.dumps({"returncode": returncode, "ended": time.time()}) + "\n")
        return returncode


def result_count(root: Path, algorithm: str, degree: int | None = None) -> int:
    import h5py

    count = 0
    for path in (root / algorithm).glob("*.hdf5"):
        with h5py.File(path) as result:
            if result["neighbors"].shape != (10000, 10):
                continue
            if degree is None or f"degree={degree}," in str(result.attrs.get("name", "")):
                count += 1
    return count


def expected_count(dataset_file: Path, definitions: Path) -> int:
    import h5py
    from ann_benchmarks.definitions import get_definitions

    with h5py.File(dataset_file) as data:
        definitions = get_definitions(
            data["train"].shape[1],
            data.attrs.get("point_type", "float"),
            data.attrs["distance"],
            10,
            str(definitions),
        )
    return sum(len(item.query_argument_groups or [[]]) for item in definitions)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def define_jobs(bench: Path, dataset: str, comparator: str, root: Path) -> tuple[list[tuple[str, Path, int]], tuple[str, Path, int]]:
    import yaml

    hms_config = yaml.safe_load((bench / "ann_benchmarks/algorithms/hms/config.yml").read_text())
    hms_jobs = []
    for degree in (32, 48, 64):
        spec = json.loads(json.dumps(hms_config))
        group = f"degree-{degree}"
        spec["float"]["any"][0]["run_groups"] = {group: spec["float"]["any"][0]["run_groups"][group]}
        definition = root / "defs" / f"hms-d{degree}"
        (definition / "hms").mkdir(parents=True, exist_ok=True)
        (definition / "hms/config.yml").write_text(yaml.safe_dump(spec))
        hms_jobs.append((f"hms-d{degree}", definition, 36))

    package = "faiss" if comparator == "faiss-ivfpqfs" else comparator
    definition = root / "defs" / package
    (definition / package).mkdir(parents=True, exist_ok=True)
    source = yaml.safe_load((bench / f"ann_benchmarks/algorithms/{package}/config.yml").read_text())
    if package == "faiss":
        source["float"]["any"] = [entry for entry in source["float"]["any"] if entry["name"] == "faiss-ivfpqfs"]
    (definition / package / "config.yml").write_text(yaml.safe_dump(source))
    count = expected_count(bench / "data" / f"{dataset}.hdf5", definition)
    return hms_jobs, (comparator, definition, count)


@app.function(image=image, volumes={"/vol": VOL}, cpu=(CPU, CPU),
              memory=(MEM_GIB * 1024, MEM_GIB * 1024), timeout=23 * 3600)
def run_dataset(dataset: str, cap_usd: float, run_tag: str = "", glass_package: str = "source") -> dict:
    if dataset not in DATASETS or cap_usd <= 0 or not re.fullmatch(r"[a-z0-9-]{0,48}", run_tag) or glass_package not in ("source", "official"):
        raise ValueError("unsupported dataset or cap")
    started = time.monotonic()
    deadline = started + min(22 * 3600, cap_usd / USD_PER_HOUR * 3600)
    digest, comparator = DATASETS[dataset]
    root = RUN_ROOT.with_name(RUN_ROOT.name + ("-" + run_tag if run_tag else "")) / dataset
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    bench = Path("/work/ann")
    if not bench.exists():
        shutil.copytree("/opt/ann", bench, ignore=shutil.ignore_patterns(".git", "results"))
    if str(bench) not in sys.path:
        sys.path.insert(0, str(bench))
    (bench / "data").mkdir(exist_ok=True)
    (bench / "results").symlink_to(root / "results", target_is_directory=True)
    dataset_file = root / "data" / f"{dataset}.hdf5"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    if not dataset_file.exists() or sha256(dataset_file) != digest:
        request = urllib.request.Request(f"https://ann-benchmarks.com/{dataset}.hdf5", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=600) as response, dataset_file.open("wb") as output:
            shutil.copyfileobj(response, output)
    if sha256(dataset_file) != digest:
        raise ValueError("dataset SHA-256 mismatch")
    (bench / "data" / f"{dataset}.hdf5").symlink_to(dataset_file)

    cpu = min(os.sched_getaffinity(0))
    env = {**os.environ, **{key: "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TBB_NUM_THREADS")}}
    cpu_info = Path("/proc/cpuinfo").read_text()
    cpu_model = next((line.split(":", 1)[1].strip() for line in cpu_info.splitlines() if line.startswith("model name")), "unknown")
    comparator_package = "faiss" if comparator == "faiss-ivfpqfs" else comparator
    wheel = next(Path("/opt/wheels").glob("*.whl"))
    package_list = subprocess.run(["uv", "pip", "freeze", "--system"], check=True, capture_output=True, text=True).stdout
    (root / "requirements-resolved.txt").write_text(package_list)

    manifest = {
        "source_rev": SOURCE_REV,
        "harness_rev": HARNESS_REV,
        "dataset": dataset,
        "dataset_sha256": digest,
        "wheel_sha256": sha256(wheel),
        "hms_adapter_sha256": sha256(bench / "ann_benchmarks/algorithms/hms/module.py"),
        "hms_config_sha256": sha256(bench / "ann_benchmarks/algorithms/hms/config.yml"),
        "competitor_config_sha256": sha256(bench / f"ann_benchmarks/algorithms/{comparator_package}/config.yml"),
        "cpu": cpu_model,
        "cpu_affinity": cpu,
        "cpu_request": CPU,
        "memory_mib": MEM_GIB * 1024,
        "runs": 5,
        "k": 10,
        "mode": "upstream --local, one pinned CPU; official Docker/AWS hardware differs",
        "started": time.time(),
        "cap_usd": cap_usd,
        "run_tag": run_tag,
        "glass_package": glass_package if comparator == "glass" else None,
        "container_id": os.environ.get("MODAL_TASK_ID"),
        "cpuinfo": cpu_info,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    VOL.commit()

    if comparator == "glass":
        if glass_package == "official":
            wheel = root / "glassppy-1.0.5-cp310-cp310-manylinux_2_34_x86_64.whl"
            url = "https://files.pythonhosted.org/packages/70/07/c10e7e9056f6513cf13f64158adad06e63011cf96aa0390a9575e02bef36/glassppy-1.0.5-cp310-cp310-manylinux_2_34_x86_64.whl"
            with urllib.request.urlopen(url, timeout=60) as response:
                wheel.write_bytes(response.read())
            if sha256(wheel) != "3f4ab3a31a8ee4ba1ced664de035eb96a6a9e98009a1251d3c22cf6150a7aec6":
                raise ValueError("official Glass wheel SHA-256 mismatch")
            subprocess.run(["uv", "pip", "install", "--system", "--reinstall", str(wheel)], check=True)
            synthetic = "import glassppy as g,numpy as n; x=n.random.default_rng(7).normal(size=(1000,784)).astype('float32'); b=g.Index('HNSW',dim=784,metric='L2',R=8,L=30); graph=b.build(x); s=g.Searcher(graph,x,'L2',3); s.set_ef(20); r=s.search(x[0],10); assert len(r)==10; print('official Glass level3 synthetic smoke passed')"
            with (root / "logs/glass-level3-preflight.log").open("w") as output:
                subprocess.run(["taskset", "-c", str(cpu), "python", "-c", synthetic], env=env,
                               stdout=output, stderr=subprocess.STDOUT, check=True, timeout=120)
            manifest["glass_wheel_sha256"] = sha256(wheel)
            (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            VOL.commit()
        else:
            glass = Path("/work/pyglass")
            subprocess.run(["git", "clone", "-q", "https://github.com/hhy3/pyglass.git", str(glass)], check=True)
            subprocess.run(["git", "-C", str(glass), "checkout", "-q", GLASS_REV], check=True)
            subprocess.run(["uv", "pip", "install", "--system", "--no-build-isolation", str(glass / "python")], check=True)
    jobs, competitor_job = define_jobs(bench, dataset, comparator, root)
    jobs.append(competitor_job)
    outcomes = {}
    for name, definition, expected in jobs:
        algorithm = "hms" if name.startswith("hms-") else name
        degree = int(name[-2:]) if name.startswith("hms-d") else None
        attempts = 0
        no_progress = 0
        while result_count(root / "results" / dataset / "10", algorithm, degree) < expected:
            if time.monotonic() >= deadline or attempts >= max(3, 3 * expected):
                outcomes[name] = "incomplete"
                break
            attempts += 1
            before = result_count(root / "results" / dataset / "10", algorithm, degree)
            if algorithm == "glass" and (bench / "glass_indices").exists():
                archive = root / "glass-graphs-archive"
                archive.mkdir(exist_ok=True)
                shutil.move(str(bench / "glass_indices"), str(archive / f"{attempts}-{time.time_ns()}"))
            args = ["taskset", "-c", str(cpu), "python", "-u", "run.py", "--local", "--dataset", dataset,
                    "--definitions", str(definition), "--runs", "5", "--algorithm", algorithm]
            try:
                run_command(args, cwd=bench, log=root / "logs" / f"{name}.log", deadline=deadline, env=env)
            except subprocess.TimeoutExpired:
                outcomes[name] = "timeout"
                break
            VOL.commit()
            after = result_count(root / "results" / dataset / "10", algorithm, degree)
            no_progress = no_progress + 1 if after <= before else 0
            if no_progress >= 2:
                outcomes[name] = "failed_no_progress_after_two_attempts"
                break
        else:
            outcomes[name] = "done"
            (root / "checkpoints" / f"{name}.done").touch()
            VOL.commit()
    elapsed = time.monotonic() - started
    receipt = {"dataset": dataset, "outcomes": outcomes, "elapsed_s": elapsed,
               "estimated_usd": elapsed / 3600 * USD_PER_HOUR}
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    VOL.commit()
    return receipt


@app.function(image=image, volumes={"/vol": VOL}, cpu=2, memory=8192, timeout=3600)
def export_dataset(dataset: str, run_tag: str = "") -> dict:
    if dataset not in DATASETS or not re.fullmatch(r"[a-z0-9-]{0,48}", run_tag):
        raise ValueError("unsupported dataset")
    root = RUN_ROOT.with_name(RUN_ROOT.name + ("-" + run_tag if run_tag else "")) / dataset
    for name in ("hms-d32", "hms-d48", "hms-d64", DATASETS[dataset][1]):
        if not (root / "checkpoints" / f"{name}.done").exists():
            raise RuntimeError(f"{dataset}: {name} is incomplete")
    bench = Path("/work/export-ann")
    shutil.copytree("/opt/ann", bench, ignore=shutil.ignore_patterns(".git", "results"))
    (bench / "results").symlink_to(root / "results", target_is_directory=True)
    (bench / "data" / f"{dataset}.hdf5").symlink_to(root / "data" / f"{dataset}.hdf5")
    log = root / "logs" / "data-export.log"
    with log.open("w") as output:
        subprocess.run(["python", "data_export.py", "--output", str(root / "results.csv")],
                       cwd=bench, stdout=output, stderr=subprocess.STDOUT, check=True)
    files = sorted((root / "results" / dataset / "10").rglob("*.hdf5"))
    hashes = {str(path.relative_to(root)): sha256(path) for path in files}
    (root / "raw-results.sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    VOL.commit()
    return {"dataset": dataset, "raw_files": len(files), "csv": str(root / "results.csv")}


@app.function(image=modal.Image.debian_slim(python_version="3.10"), timeout=23 * 3600)
def coordinate(selected: list[str], cap_usd: float) -> list[dict]:
    calls = [(dataset, run_dataset.spawn(dataset, cap_usd / len(selected))) for dataset in selected]
    for dataset, call in calls:
        print(json.dumps({"dataset": dataset, "call_id": call.object_id, "cap_usd": cap_usd / len(selected)}), flush=True)
    results = []
    for dataset, call in calls:
        try:
            results.append(call.get())
        except Exception as error:
            results.append({"dataset": dataset, "error": str(error)[:1000]})
    return results


@app.local_entrypoint()
def main(cap_usd: float = 24.0, datasets: str = ""):
    selected = datasets.split(",") if datasets else list(DATASETS)
    if not selected or any(dataset not in DATASETS for dataset in selected) or cap_usd <= 0:
        raise ValueError("invalid datasets or cap")
    print(json.dumps(coordinate.remote(selected, cap_usd), indent=2))
