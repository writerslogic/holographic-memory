# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Numerical agreement of the Rust engine with the sentence-transformers fp32 reference.

  sample     write the fixed agreement sample (benchmarks/embedding/agreement_sample.json)
  reference  embed the sample with the PyTorch reference model -> <out>.json (base64 float32)
  compare    run `hms-embed embed` on the sample and report min/mean cosine per kind

The sample is chosen by sha256 order from SciFact documents, NFCorpus queries and LongMemEval
dev turns, plus fixed edge cases (empty-ish text, CJK, emoji, accents, code, a text past every
model's window). The acceptance bar is min cosine >= 0.999, not the mean.

  uv run --no-project --with sentence-transformers --with numpy python benchmarks/embedding/agreement.py reference --model BAAI/bge-small-en-v1.5 --out /tmp/ref.json
"""
import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DL = Path.home() / ".cache/hms-bench/downloads"
SAMPLE = HERE / "agreement_sample.json"

EDGE = [
    "?",
    "a",
    "Hello, world!",
    "Café déjà vu: naïve façade, Zürich, São Paulo.",
    "東京は日本の首都であり、世界最大の都市圏の一つです。",
    "Москва является столицей России.",
    "Emoji test 🚀🔥 and symbols ∑ ∫ ≈ ± µ.",
    "def f(x):\n    return [i**2 for i in range(x) if i % 2 == 0]  # squares of evens",
    "line one\nline two\n\n\tindented line three",
    "UPPERCASE AND lowercase MiXeD Case with numbers 12345 and 3.14159.",
    "word " * 700,
]


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def pick(items, n, salt):
    return sorted(items, key=lambda t: hashlib.sha256((salt + t).encode()).hexdigest())[:n]


def make_sample():
    sci = [(d.get("title") or "") + " " + d["text"] for d in jsonl(DL / "scifact/corpus.jsonl")]
    nfq = [q["text"] for q in jsonl(DL / "nfcorpus/queries.jsonl")]
    split = json.loads(Path(ROOT / "benchmarks/public/longmemeval_split.json").read_text())
    dev = set(split["dev"])
    turns = []
    for item in json.loads(Path(DL / "longmemeval_s_cleaned.json").read_text()):
        if item["question_id"] in dev:
            for s in item["haystack_sessions"][:3]:
                turns += [t["content"] for t in s if len(t["content"]) > 20]
    sample = {
        "passage": pick(sci, 16, "sci") + pick(turns, 12, "lme") + EDGE,
        "query": pick(nfq, 16, "nfq") + pick([t for t in turns if len(t) < 200], 6, "lmeq") + EDGE[:10],
    }
    SAMPLE.write_text(json.dumps(sample, ensure_ascii=False, indent=0))
    print({k: len(v) for k, v in sample.items()})


def reference(model_id, revision, out):
    from sentence_transformers import SentenceTransformer

    sample = json.loads(Path(SAMPLE).read_text())
    from select_model import CANDIDATES  # prefixes live with the candidate table

    qp, pp, _ = CANDIDATES[model_id]
    m = SentenceTransformer(model_id, revision=revision, device="cpu")
    m.max_seq_length = min(m.max_seq_length or 512, 512)
    res = {"model": model_id, "revision": revision, "max_seq_length": m.max_seq_length}
    for kind, prefix in (("passage", pp), ("query", qp)):
        v = m.encode([prefix + t for t in sample[kind]], normalize_embeddings=True, convert_to_numpy=True,
                     batch_size=8)
        res[kind] = base64.b64encode(v.astype("<f4").tobytes()).decode()
        res["dim"] = int(v.shape[1])
    Path(out).write_text(json.dumps(res))


def compare(ref_path, binary, model, extra):
    ref = json.loads(Path(ref_path).read_text())
    sample = json.loads(Path(SAMPLE).read_text())
    out = {}
    for kind in ("passage", "query"):
        texts = sample[kind]
        lines = [json.dumps(t, ensure_ascii=False) for t in texts]
        raw = HERE / f".agreement-{kind}.f32"
        subprocess.run([binary, "embed", "--model", model, "--raw-out", str(raw), "--input", "-", "--json-lines"] +
                       (["--query"] if kind == "query" else []) + extra,
                       input="\n".join(lines).encode(), check=True)
        got = np.fromfile(raw, dtype="<f4").reshape(len(texts), -1)
        raw.unlink()
        want = np.frombuffer(base64.b64decode(ref[kind]), dtype="<f4").reshape(len(texts), -1)
        cos = (got * want).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(want, axis=1))
        out[kind] = {"n": len(texts), "min_cosine": float(cos.min()), "mean_cosine": float(cos.mean()),
                     "worst_index": int(cos.argmin())}
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample")
    r = sub.add_parser("reference")
    r.add_argument("--model", required=True)
    r.add_argument("--revision", required=True)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--reference", required=True)
    c.add_argument("--binary", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("extra", nargs="*")
    a = ap.parse_args()
    if a.cmd == "sample":
        make_sample()
    elif a.cmd == "reference":
        reference(a.model, a.revision, a.out)
    else:
        compare(a.reference, a.binary, a.model, a.extra)
