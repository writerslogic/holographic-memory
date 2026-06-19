"""HMS scaling benchmarks on Modal — parallel runs across dimensions and densities."""

import json
import logging
import modal
import datetime

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("hms-bench")

app = modal.App("hms-scaling-benchmarks")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "pkg-config")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
    .add_local_dir("/Volumes/A/HMS/src", "/app/src", copy=True)
    .add_local_file("/Volumes/A/HMS/Cargo.toml", "/app/Cargo.toml", copy=True)
    .add_local_file("/Volumes/A/HMS/Cargo.lock", "/app/Cargo.lock", copy=True)
    .add_local_file("/Volumes/A/HMS/build.rs", "/app/build.rs", copy=True)
    .add_local_dir("/Volumes/A/HMS/benches", "/app/benches", copy=True)
    .run_commands(
        "cd /app && cargo build --release --bin hms-scaling",
    )
)

CONFIGS = [
    (16384, 256),
    (65536, 256),
    (65536, 1024),
    (131072, 256),
    (131072, 1024),
    (262144, 1024),
    (262144, 4096),
    (524288, 1024),
    (524288, 4096),
]


@app.function(image=image, cpu=4, memory=16384, timeout=3600)
def run_benchmark(dim: int, density: int) -> dict:
    import subprocess
    try:
        result = subprocess.run(
            ["/app/target/release/hms-scaling", "--dim", str(dim), "--density", str(density), "--json"],
            capture_output=True, text=True, timeout=3000,
        )
        if result.returncode != 0:
            return {"dim": dim, "density": density, "error": result.stderr[-2000:]}
        data = json.loads(result.stdout)
        return data["scaling_benchmark"][0]
    except Exception as e:
        return {"dim": dim, "density": density, "error": str(e)[-2000:]}


@app.local_entrypoint()
def main():
    log.info("Launching %d benchmark configurations on Modal...", len(CONFIGS))
    futures = []
    for dim, density in CONFIGS:
        futures.append(run_benchmark.spawn(dim, density))

    results = []
    for i, future in enumerate(futures):
        dim, density = CONFIGS[i]
        log.info("  Waiting for D=%d, 1/%d...", dim, density)
        try:
            result = future.get()
        except Exception as e:
            result = {"dim": dim, "density": density, "error": str(e)[-2000:]}
        results.append(result)
        if "error" in result:
            log.error("    ERROR: %s", result["error"][:200])
        else:
            cw = result.get("capacity_wall", {})
            wall = cw.get("wall_at_95_recall", "?")
            log.info("    Capacity wall: %s items", wall)

    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scaling_benchmark": results,
    }

    outfile = "benchmark_scaling_results.json"
    with open(outfile, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Results saved to %s", outfile)

    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    for r in results:
        if "error" in r:
            log.info("  D=%6d 1/%4d: ERROR", r["dim"], r["density"])
            continue
        dim = r["dim"]
        dd = r["density_denom"]
        active = r["active_indices"]
        cw = r["capacity_wall"]["wall_at_95_recall"]
        ss = r.get("structured_retrieval_stress", [])
        max_roles = 0
        for s in ss:
            if s["accuracy"] >= 0.99:
                max_roles = s["n_roles"]
        tp = r.get("throughput", {})
        enc = tp.get("encode_ops_per_sec", 0)
        mem = r.get("memory", {})
        comp = mem.get("compression_ratio", 0)
        log.info(
            "  D=%6d 1/%4d (active=%4d): cap=%6d items, roles=%3d@100%%, encode=%.0f/s, %.0fx compression",
            dim, dd, active, cw, max_roles, enc, comp,
        )
