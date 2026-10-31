# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Embed the tuning-split queries of a cached BEIR dataset with the pinned MiniLM model (the
model and revision recorded in <cache>/<dataset>/meta.json), L2-normalized, as f32 rows next to
the learned-sparse encodings: <cache>/sparse/<dataset>/<model>/queries_tune_dense.f32.

Usage: learned_sparse_dense_tune.py --dataset nfcorpus --model os-doc-v2-distill [--cache DIR]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--cache", default=os.path.expanduser("~/.cache/warm-bench"))
    a = p.parse_args()
    from sentence_transformers import SentenceTransformer
    cache = Path(a.cache)
    meta = json.loads((cache / a.dataset / "meta.json").read_text())
    out = cache / "sparse" / a.dataset / a.model
    ids = json.loads((out / "queries_tune_ids.json").read_text())
    import zipfile, io
    texts = {}
    with zipfile.ZipFile(cache / "downloads" / f"{a.dataset}.zip") as z:
        with z.open(f"{a.dataset}/queries.jsonl") as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                if line.strip():
                    q = json.loads(line)
                    texts[q["_id"]] = q["text"]
    model = SentenceTransformer(meta["model"], revision=meta["model_revision"])
    emb = model.encode([texts[i] for i in ids], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    emb.astype(np.float32).tofile(out / "queries_tune_dense.f32")
    print(json.dumps({"rows": len(ids), "dim": int(emb.shape[1]), "out": str(out / "queries_tune_dense.f32")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
