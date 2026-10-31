# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Collects benchmarks/results/mobile_v1.json: binary sizes of every built mobile artifact and
the repeated runs of the `mobile_bench` example.

    uv run python bindings/uniffi/scripts/collect_mobile_bench.py \
        --target-dir ./.warm-target-mobile --bench-dir /Volumes/hmsram/bench

The bench directory should be on a RAM disk or fast flash: on spinning disks the numbers
measure the disk. Run the Rust example binaries are built first with
`cargo build --release --example mobile_bench` in bindings/uniffi.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

ANDROID = {
    "arm64-v8a": "aarch64-linux-android",
    "armeabi-v7a": "armv7-linux-androideabi",
    "x86_64": "x86_64-linux-android",
    "x86": "i686-linux-android",
}
APPLE = [
    "aarch64-apple-ios",
    "aarch64-apple-ios-sim",
    "x86_64-apple-ios",
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "aarch64-apple-ios-macabi",
    "x86_64-apple-ios-macabi",
]


SIZE_NOTES = (
    'strippedSoBytes and strippedDylibBytes are the cdylib after rustc strip (what a linked app embeds is smaller: the linker drops unused code from the static library, whose size is not the app delta). The macOS dylib shown is not loadable on macOS 27 because of rustc strip; the JVM host library is built separately with strip=debuginfo. xcframeworkBytes is the sum of all static slices, an upper bound on download size, not on app size. releaseAarBytes holds the four ABIs. Bench disk bytes are dominated by the preallocated 64 MiB log segment (sparse where the file system supports it) and did not change between 1024 and 4096 dimensions.'
)


def size(path: Path):
    return path.stat().st_size if path.exists() else None


def sizes(target: Path):
    out = {"android": {}, "apple": {}}
    for abi, triple in ANDROID.items():
        so = ROOT / "bindings/kotlin/android/src/main/jniLibs" / abi / "libwarm_mobile.so"
        out["android"][abi] = {
            "triple": triple,
            "strippedSoBytes": size(so),
            "staticLibBytes": size(target / triple / "release/libwarm_mobile.a"),
        }
    for triple in APPLE:
        rel = target / triple / "release"
        out["apple"][triple] = {
            "strippedDylibBytes": size(rel / "libwarm_mobile.dylib"),
            "staticLibBytes": size(rel / "libwarm_mobile.a"),
        }
    xc = ROOT / "bindings/swift/Frameworks/warm_mobileFFI.xcframework"
    out["xcframeworkBytes"] = sum(f.stat().st_size for f in xc.rglob("*") if f.is_file()) if xc.exists() else None
    aar = list((ROOT / "bindings/kotlin/android/build/outputs/aar").glob("*release*.aar"))
    out["releaseAarBytes"] = size(aar[0]) if aar else None
    return out


def run_bench(binary: Path, bench_dir: Path, docs: int, words: int, dims: int, queries: int):
    env = dict(os.environ, BENCH_DIR=str(bench_dir))
    load_before = os.getloadavg()[0]
    result = subprocess.run(
        [str(binary), "--docs", str(docs), "--words", str(words), "--dims", str(dims), "--queries", str(queries)],
        env=env, check=True, capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    data["loadAverage1mBefore"] = load_before
    data["loadAverage1mAfter"] = os.getloadavg()[0]
    return data


def median_of(runs, path):
    values = []
    for run in runs:
        node = run
        for key in path:
            node = node[key]
        values.append(node)
    return statistics.median(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--bench-dir", required=True, type=Path)
    parser.add_argument("--docs", type=int, default=5000)
    parser.add_argument("--words", type=int, default=150)
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dims", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/mobile_v1.json")
    args = parser.parse_args()

    binary = args.target_dir / "release/examples/mobile_bench"
    runs = {}
    for dims in args.dims:
        runs[str(dims)] = [
            run_bench(binary, args.bench_dir, args.docs, args.words, dims, args.queries)
            for _ in range(args.repeats)
        ]
    summary = {
        dims: {
            "searchP50Ms": median_of(r, ["search", "p50Ms"]),
            "searchP95Ms": median_of(r, ["search", "p95Ms"]),
            "ingestSingleP50Ms": median_of(r, ["ingestSingle", "p50Ms"]),
            "ingestBatch100DocsPerSecond": median_of(r, ["ingestBatch100", "docsPerSecond"]),
            "reopenMs": median_of(r, ["reopenMs"]),
            "peakRssBytes": median_of(r, ["rssBytes", "peak"]),
            "diskBytesAfterCompact": median_of(r, ["diskBytesAfterCompact"]),
            "diskAllocatedBytesAfterCompact": median_of(r, ["diskAllocatedBytesAfterCompact"]),
        }
        for dims, r in runs.items()
    }
    out = {
        "schema": "mobile_v1",
        "generatedUnix": int(time.time()),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "host": {
            "machine": platform.machine(),
            "system": platform.platform(),
            "cpuBrand": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
            "memBytes": int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip() or 0),
        },
        "method": {
            "corpus": "synthetic: Zipf(1.0) over 30000 pronounceable words, fixed seed; measures cost, not ranking quality",
            "storage": str(args.bench_dir),
            "notes": "Rust-level API calls (the same functions Swift and Kotlin call through UniFFI); FFI marshalling not included. "
                     "The bench directory was a RAM disk so file system latency is excluded. The host was shared with other "
                     "builds, see loadAverage fields; medians over repeats, raw runs kept.",
            "docs": args.docs, "wordsPerDoc": args.words, "queries": args.queries, "repeats": args.repeats,
        },
        "binarySizes": sizes(args.target_dir),
        "sizeNotes": SIZE_NOTES,
        "summaryMedianByDimensions": summary,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
