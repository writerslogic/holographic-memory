# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Encode a cached BEIR dataset into learned sparse vectors for the WARM sparse engine.

Usage: learned_sparse_encode.py --dataset {nfcorpus,scifact} --model {splade-v3,splade-pp-ed,os-doc-v2-distill}
                                [--cache ~/.cache/warm-bench] [--device mps|cpu] [--batch 32]

Reads the cached corpus and test queries, plus the tuning split from the BEIR zip in
<cache>/downloads (NFCorpus dev, SciFact train), and writes under
<cache>/sparse/<dataset>/<model>/: corpus.sparse, queries_test.sparse, queries_tune.sparse (the
binary format `sparse-bench` reads: magic WARMSPRS1, u32 vocab, u32 n, then per record u32 len,
u32 ids ascending, f32 weights), the matching *_ids.json, qrels_test.tsv, qrels_tune.tsv, and the
Anserini JsonVectorCollection and JsonIntVector topics with weights quantized by 100.

Models: naver/splade-v3 (CC BY-NC-SA 4.0, comparator only; one encoder for queries and documents,
log1p(relu(logits)) max-pooled over tokens) and
opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill (Apache-2.0; documents by the
same pooling with special tokens zeroed, queries by tokenizer and the published IDF table, no
query-side inference).
"""
import argparse
import io
import json
import os
import struct
import sys
import time
import zipfile
from pathlib import Path

MODELS = {
    "splade-v3": ("naver/splade-v3", "cc-by-nc-sa-4.0", "https://huggingface.co/naver/splade-v3"),
    "splade-pp-ed": ("naver/splade-cocondenser-ensembledistil", "cc-by-nc-sa-4.0",
                     "https://huggingface.co/naver/splade-cocondenser-ensembledistil"),
    "os-doc-v2-distill": ("opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill", "apache-2.0",
                          "https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"),
}
TUNE_SPLIT = {"nfcorpus": "dev", "scifact": "train"}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_sparse(path, vocab, records):
    with open(path, "wb") as f:
        f.write(b"HMSSPRS1")
        f.write(struct.pack("<II", vocab, len(records)))
        for ids, weights in records:
            f.write(struct.pack("<I", len(ids)))
            f.write(struct.pack(f"<{len(ids)}I", *ids))
            f.write(struct.pack(f"<{len(ids)}f", *weights))


def nonzeros(row):
    ids = [i for i, w in enumerate(row) if w > 0.0]
    return ids, [float(row[i]) for i in ids]


def encode_model(model, tokenizer, texts, device, batch, max_len, zero_special):
    import torch
    out = []
    special = set(tokenizer.all_special_ids)
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            weights = torch.log1p(torch.relu(logits)) * enc["attention_mask"].unsqueeze(-1)
            vec = weights.max(dim=1).values
            if zero_special:
                vec[:, list(special)] = 0.0
        for row in vec.cpu().float().tolist():
            out.append(nonzeros(row))
        if (start // batch) % 20 == 0:
            print(f"  {start + len(chunk)}/{len(texts)}", flush=True)
    return out


def idf_queries(tokenizer, idf, texts):
    out = []
    for text in texts:
        ids = sorted(set(tokenizer(text, add_special_tokens=False)["input_ids"]))
        pairs = [(i, idf[i]) for i in ids if idf[i] > 0.0]
        out.append(([i for i, _ in pairs], [w for _, w in pairs]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=sorted(TUNE_SPLIT), required=True)
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--cache", default=os.path.expanduser("~/.cache/warm-bench"))
    p.add_argument("--device", default="mps")
    p.add_argument("--batch", type=int, default=32)
    a = p.parse_args()
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    device = a.device if (a.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    repo, license_, url = MODELS[a.model]
    cache = Path(a.cache)
    data = cache / a.dataset
    out = cache / "sparse" / a.dataset / a.model
    out.mkdir(parents=True, exist_ok=True)
    corpus = read_jsonl(data / "corpus.jsonl")
    test_queries = read_jsonl(data / "queries.jsonl")
    split = TUNE_SPLIT[a.dataset]
    with zipfile.ZipFile(cache / "downloads" / f"{a.dataset}.zip") as z:
        all_queries = {}
        with z.open(f"{a.dataset}/queries.jsonl") as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                if line.strip():
                    q = json.loads(line)
                    all_queries[q["_id"]] = q["text"]
        tune_qrels = z.read(f"{a.dataset}/qrels/{split}.tsv").decode("utf-8")
    tune_ids = []
    seen = set()
    for line in tune_qrels.splitlines()[1:]:
        qid = line.split("\t")[0]
        if qid in all_queries and qid not in seen:
            seen.add(qid)
            tune_ids.append(qid)
    tune_queries = [{"id": qid, "text": all_queries[qid]} for qid in tune_ids]
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForMaskedLM.from_pretrained(repo).to(device).eval()
    vocab = len(tokenizer)
    t0 = time.time()
    print(f"encoding {len(corpus)} documents with {repo} on {device}", flush=True)
    docs = encode_model(model, tokenizer, [d["text"] for d in corpus], device, a.batch, 256, a.model.startswith("os-"))
    doc_seconds = time.time() - t0
    if a.model.startswith("splade"):
        qs_test = encode_model(model, tokenizer, [q["text"] for q in test_queries], device, a.batch, 64, False)
        qs_tune = encode_model(model, tokenizer, [q["text"] for q in tune_queries], device, a.batch, 64, False)
    else:
        from huggingface_hub import hf_hub_download
        table = json.loads(Path(hf_hub_download(repo, "idf.json")).read_text(encoding="utf-8"))
        idf = [0.0] * vocab
        for token, weight in table.items():
            idf[tokenizer.convert_tokens_to_ids(token)] = float(weight)
        qs_test = idf_queries(tokenizer, idf, [q["text"] for q in test_queries])
        qs_tune = idf_queries(tokenizer, idf, [q["text"] for q in tune_queries])
    write_sparse(out / "corpus.sparse", vocab, docs)
    write_sparse(out / "queries_test.sparse", vocab, qs_test)
    write_sparse(out / "queries_tune.sparse", vocab, qs_tune)
    (out / "corpus_ids.json").write_text(json.dumps([d["id"] for d in corpus]))
    (out / "queries_test_ids.json").write_text(json.dumps([q["id"] for q in test_queries]))
    (out / "queries_tune_ids.json").write_text(json.dumps([q["id"] for q in tune_queries]))
    (out / "qrels_test.tsv").write_bytes((data / "qrels.tsv").read_bytes())
    (out / "qrels_tune.tsv").write_text(tune_qrels)
    # Anserini: JsonVectorCollection documents and JsonIntVector topics, weights quantized by 100.
    anserini = out / "anserini"
    (anserini / "collection").mkdir(parents=True, exist_ok=True)
    with open(anserini / "collection" / "docs.jsonl", "w") as f:
        for d, (ids, ws) in zip(corpus, docs):
            vec = {tokenizer.convert_ids_to_tokens(i): int(round(w * 100)) for i, w in zip(ids, ws)}
            vec = {t: w for t, w in vec.items() if w > 0}
            f.write(json.dumps({"id": d["id"], "contents": "", "vector": vec}) + "\n")
    for name, qs, recs in (("test", test_queries, qs_test), ("tune", tune_queries, qs_tune)):
        with open(anserini / f"topics_{name}.jsonl", "w") as f:
            for q, (ids, ws) in zip(qs, recs):
                vec = {tokenizer.convert_ids_to_tokens(i): int(round(w * 100)) for i, w in zip(ids, ws)}
                vec = {t: w for t, w in vec.items() if w > 0}
                f.write(json.dumps({"id": q["id"], "vector": vec}) + "\n")
    nnz = sum(len(ids) for ids, _ in docs) / max(1, len(docs))
    (out / "meta.json").write_text(json.dumps({"model": repo, "license": license_, "url": url, "dataset": a.dataset, "device": device,
               "documents": len(corpus), "test_queries": len(test_queries), "tune_queries": len(tune_queries),
               "tune_split": split, "vocab": vocab, "mean_doc_nonzeros": nnz, "document_encode_seconds": doc_seconds,
               "max_length": {"document": 256, "query": 64}, "anserini_quantization": 100}, indent=1))
    print(json.dumps({"out": str(out), "mean_doc_nonzeros": nnz, "doc_seconds": doc_seconds}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
