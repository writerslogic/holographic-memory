# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pins a registry model: downloads the upstream files at one commit, hashes them, builds the
int8 pack with `warm-embed pack` and prints the Rust `ModelSpec` entry for
src/core/embed/registry.rs.

  uv run --no-project python benchmarks/embedding/pin_model.py --repo BAAI/bge-small-en-v1.5 \
      --id bge-small-en-v1.5 --pooling cls --binary $TARGET/release/warm-embed --workdir DIR
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

FILES = ["config.json", "tokenizer.json", "model.safetensors"]


def hf_cache_file(repo, rev, name):
    """A file of this exact revision already in a Hugging Face hub cache, if any."""
    import os
    roots = [Path(os.environ["HF_HOME"]) / "hub"] if os.environ.get("HF_HOME") else []
    roots.append(Path.home() / ".cache/huggingface/hub")
    for r in roots:
        f = r / ("models--" + repo.replace("/", "--")) / "snapshots" / rev / name
        if f.exists():
            return f.resolve()
    return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--pooling", required=True, choices=["mean", "cls"])
    ap.add_argument("--query-prefix", default="")
    ap.add_argument("--passage-prefix", default="")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--quantization", default="int8")
    ap.add_argument("--int8-group", type=int, default=64)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--revision")
    a = ap.parse_args()
    info = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{a.repo}", timeout=60))
    rev = a.revision or info["sha"]
    lic = (info.get("cardData") or {}).get("license")
    assert lic in ("mit", "apache-2.0"), f"licence {lic!r} is not redistributable under the project rules"
    pooling_cfg = json.load(urllib.request.urlopen(
        f"https://huggingface.co/{a.repo}/resolve/{rev}/1_Pooling/config.json", timeout=60))
    declared = "cls" if pooling_cfg.get("pooling_mode_cls_token") else (
        "mean" if pooling_cfg.get("pooling_mode_mean_tokens") else "other")
    assert declared == a.pooling, f"{a.repo} declares {declared!r} pooling, not {a.pooling!r}"
    d = a.workdir / a.id
    d.mkdir(parents=True, exist_ok=True)
    pins = []
    for name in FILES:
        dest = d / name
        if not dest.exists():
            cached = hf_cache_file(a.repo, rev, name)
            if cached:
                shutil.copyfile(cached, dest)
            else:
                urllib.request.urlretrieve(f"https://huggingface.co/{a.repo}/resolve/{rev}/{name}", dest)
        pins.append((name, sha256(dest), dest.stat().st_size))
    pack = d / f"{a.id}-{a.quantization}.hmsemb"
    out = subprocess.run(
        [a.binary, "pack", "--from-dir", str(d), "--model", a.repo, "--revision", rev,
         "--license", {"mit": "MIT", "apache-2.0": "Apache-2.0"}[lic], "--pooling", a.pooling,
         "--quantization", a.quantization, "--int8-group", str(a.int8_group), "--query-prefix", a.query_prefix,
         "--passage-prefix", a.passage_prefix, "--max-length", str(a.max_length), "--out", str(pack)],
        check=True, capture_output=True, text=True).stdout.split()
    print(f'''    ModelSpec {{
        id: "{a.id}",
        repo: "{a.repo}",
        revision: "{rev}",
        license: "{ {"mit": "MIT", "apache-2.0": "Apache-2.0"}[lic] }",
        pooling: Pooling::{a.pooling.capitalize()},
        query_prefix: {json.dumps(a.query_prefix)},
        passage_prefix: {json.dumps(a.passage_prefix)},
        max_length: {a.max_length},
        quantization: Quantization::{ {"int8": "Int8", "f16": "F16", "f32": "F32"}[a.quantization] },
        int8_group: {a.int8_group},
        files: &[''')
    for n, h, s in pins:
        print(f'            PinnedFile {{ name: "{n}", sha256: "{h}", size: {s} }},')
    print(f'''        ],
        pack_sha256: "{out[0]}",
        pack_bytes: {pack.stat().st_size},
    }},''')


if __name__ == "__main__":
    main()
