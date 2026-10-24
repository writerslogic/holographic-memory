#!/usr/bin/env python3
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Isolate cached phasor width/layout without changing validation or iteration."""

import argparse
import datetime
import hashlib
import json
import platform
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "target/holography-phase-substrate"


def replace_once(source, old, new):
    assert source.count(old) == 1, f"source contract changed: {old!r}"
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, choices=["exploratory-concurrent", "serial-validation"])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--sample-ms", type=float, default=20)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/holography_phase_substrate_v1/layouts.json")
    args = parser.parse_args()
    source_path = ROOT / "benchmarks/results/holography_phase_substrate_v1/layout_phase_hvec.rs"
    source = source_path.read_text()
    wide_aos = source.replace("i16", "i64")
    wide_soa = replace_once(source, "OnceLock<Box<[[i16; 2]]>>", "OnceLock<(Vec<i64>, Vec<i64>)>")
    start = wide_soa.index("fn luts(")
    end = wide_soa.index("\nfn cached_resolution", start)
    wide_soa = wide_soa[:start] + """fn luts(n: u32) -> (Vec<i64>, Vec<i64>) {
    let cos = (0..n)
        .map(|k| (SCALE as f64 * (TAU * k as f64 / n as f64).cos()).round() as i64)
        .collect();
    let sin = (0..n)
        .map(|k| (SCALE as f64 * (TAU * k as f64 / n as f64).sin()).round() as i64)
        .collect();
    (cos, sin)
}
""" + wide_soa[end:]
    wide_soa = replace_once(wide_soa, "let lut = if let Some(index)", "let (cos, sin) = if let Some(index)")
    wide_soa = replace_once(wide_soa, """                let [cos, sin] = lut[phase as usize];
                *re += cos as i64;
                *im += sin as i64;""", """                *re += cos[phase as usize];
                *im += sin[phase as usize];""")
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "wide_aos.rs").write_text(wide_aos)
    (TARGET / "wide_soa.rs").write_text(wide_soa)
    binary = TARGET / "layout_probe"
    compile_command = ["rustc", "--edition=2021", "-C", "opt-level=3", "-C", "lto=true", "-C", "codegen-units=1", "-C", "strip=debuginfo", "benchmarks/research/phase_substrate/layout_probe.rs", "-o", str(binary)]
    subprocess.run(compile_command, cwd=ROOT, check=True)
    if args.prepare_only:
        print(f"Prepared {binary}; no timings collected")
        return
    result = {
        "label": args.label,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "serialization_scope": "this task only; unrelated external Rust jobs were observed on the host",
        "source_path": str(source_path.relative_to(ROOT)),
        "compile_command": compile_command,
        "source_sha256": {"production": hashlib.sha256(source.encode()).hexdigest(), "i64_aos": hashlib.sha256(wide_aos.encode()).hexdigest(), "i64_soa": hashlib.sha256(wide_soa.encode()).hexdigest(), "probe": hashlib.sha256((ROOT / "benchmarks/research/phase_substrate/layout_probe.rs").read_bytes()).hexdigest(), "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "runs": [],
    }
    for n in [16, 256]:
        for dim in [4096, 16384, 65536]:
            lines = subprocess.check_output([str(binary), str(n), str(dim), str(args.sample_ms), str(args.repeats)], cwd=ROOT, text=True).splitlines()
            result["runs"].extend(json.loads(line) for line in lines)
    result["summary"] = [
        {"n": row["n"], "dim": row["dim"], "bundle_count": row["bundle_count"], **{variant: statistics.median(row[f"{variant}_ns"]) for variant in ["baseline", "i16_aos", "i64_aos", "i64_soa"]}}
        for row in result["runs"]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
