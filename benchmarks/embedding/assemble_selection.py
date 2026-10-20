# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Assembles benchmarks/results/embedding_selection_v1.json from the stage outputs.

Inputs (all produced by the scripts next to this one): select_model.py (stage 1, fp32 PyTorch
quality, licence, size), run_engine_eval.py (stage 2, the Rust engine on int8 packs: quality,
latency, RSS, cold start), bench_torch_cpu.py (reference latency), agreement.py (cosine against
the fp32 reference). Adds paired-bootstrap confidence intervals of the per-query differences
between each model and the chosen default.

  uv run --no-project --with numpy python benchmarks/embedding/assemble_selection.py \
      --stage1 s1.json --stage2 DIR --torch torch.json --agreement DIR --default MODEL --out OUT
"""
import argparse
import json
from pathlib import Path

import numpy as np

METRICS = (("scifact_train", "per_query_ndcg"), ("nfcorpus_dev", "per_query_ndcg"), ("lme_dev", "per_question_ndcg"))


def bootstrap_diff(a, b, n=10000, seed=20261020):
    """Mean of a - b over paired items with a 95% percentile interval (resampling items)."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "items": int(len(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True, type=Path)
    ap.add_argument("--torch")
    ap.add_argument("--agreement", type=Path)
    ap.add_argument("--default", required=True, help="registry id of the chosen default")
    ap.add_argument("--rationale", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s1 = json.load(open(a.stage1))
    torch = json.load(open(a.torch)) if a.torch else {}
    engine = {}
    for f in sorted(a.stage2.glob("*-int8.json")):
        d = json.load(open(f))
        engine[d["model"]] = d
    default_repo = next(m for m in engine if m.split("/")[-1].lower() == a.default.lower()
                        or Path(engine[m]["pack"]).name.startswith(a.default))
    out = {
        "protocol": {
            "datasets": {"scifact": "BEIR SciFact, queries of qrels/train.tsv (809), nDCG@10 over the full corpus (5183)",
                         "nfcorpus": "BEIR NFCorpus, queries of qrels/dev.tsv (324), nDCG@10 over the full corpus (3633)",
                         "longmemeval_dev": "LongMemEval_S frozen dev ids (benchmarks/public/longmemeval_split.json), "
                                            "non-abstention (94), turn-level embeddings of both roles, session score "
                                            "= best turn, ranked over the question's own haystack"},
            "held_out": "no test split and none of the 400 held-out LongMemEval ids was read",
            "licence_rule": "MIT or Apache-2.0 only (read from the Hugging Face model card at the pinned commit)",
            "engine_quantization": "int8, one f32 scale per 64 weights per row; embeddings and linear layers; "
                                   "layer norms, biases and position tables stay f32",
            "agreement_bar": "min cosine >= 0.999 against the sentence-transformers fp32 reference on the fixed "
                             "sample of benchmarks/embedding/agreement_sample.json (71 texts)",
            "timing_note": "shared machine; load average recorded per run; compare models inside one run",
        },
        "default": default_repo,
        "rationale": a.rationale,
        "candidates": {},
    }
    for mid, v in s1.items():
        c = {"license": v["info"]["license"], "revision": v["info"]["revision"], "parameters": v["info"]["parameters"],
             "engine_architecture_supported": v.get("engine_architecture")}
        if "skipped" in v or "error" in v:
            c["not_evaluated"] = v.get("skipped") or v.get("error")
        else:
            c["fp32_pytorch"] = {k: v[k] for k in ("scifact_train", "nfcorpus_dev", "lme_dev")}
            c["max_seq_length"] = v["max_seq_length"]
            c["dimensions"] = v["dimensions"]
        if mid in torch.get("results", {}):
            c["pytorch_cpu"] = torch["results"][mid]
        if mid in engine:
            e = json.loads(json.dumps(engine[mid]))
            q = e.get("quality", {})
            for key, field in [m for m in METRICS if m[0] in q]:
                for k in ("per_query_ndcg", "per_question_ndcg", "per_question_recall5"):
                    q[key].pop(k, None)
            c["rust_engine_int8"] = e
            ref = engine[default_repo]["quality"]
            mine = engine[mid]["quality"]
            if mid != default_repo and "quality" in engine[mid]:
                c["vs_default_paired_bootstrap"] = {
                    key: bootstrap_diff(mine[key][field], ref[key][field]) for key, field in METRICS if key in mine}
            if a.agreement:
                short = Path(engine[mid]["pack"]).name.removesuffix("-int8.hmsemb")
                f = a.agreement / f"agree-{short}-int8.json"
                if f.exists():
                    c["agreement_with_fp32_reference"] = json.load(open(f))
        out["candidates"][mid] = c
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
