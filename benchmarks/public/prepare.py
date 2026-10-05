# /// script
# requires-python = ">=3.10"
# dependencies = ["h5py>=3.10", "numpy>=1.26", "sentence-transformers>=3.0"]
# ///
"""Download public benchmark data and write it in the flat format public-bench reads.

ANN sets come from ann-benchmarks (https://ann-benchmarks.com/<name>.hdf5) and ship their own
ground-truth neighbours. BEIR sets come from the BEIR distribution and are embedded here with a
pinned sentence-transformers model. Output goes to $HMS_BENCH_DATA (default ~/.cache/hms-bench),
never into the repository.

Layout per dataset directory:
  meta.json                         shape and provenance
  train.f32 / test.f32              row-major little-endian float32 (ANN)
  neighbors.i32                     row-major int32, ground-truth ids per test row (ANN)
  corpus.f32 / queries.f32          embeddings (BEIR)
  corpus.jsonl / queries.jsonl      {"id", "text"} in embedding row order (BEIR)
  qrels.tsv                         query-id, corpus-id, relevance (BEIR test split)

LongMemEval_S writes, per granularity g in {turn, session}, g.jsonl (one line per retrievable item
of each question's own haystack: {"q": question index, "id": corpus id, "text", "e": embedding
row}) and g.f32 (unique embeddings), plus questions.jsonl, queries.f32 and meta.json. Items and
ids follow the official src/retrieval/run_retrieval.py: user turns only; a session is the
concatenation of its user turns.
"""

import argparse
import hashlib
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ANN_URL = "https://ann-benchmarks.com/{name}.hdf5"
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
LME_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
LME_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/"
    + LME_REVISION
    + "/longmemeval_{variant}_cleaned.json"
)
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def root() -> Path:
    return Path(os.environ.get("HMS_BENCH_DATA", Path.home() / ".cache" / "hms-bench"))


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"downloading {url}")
        # The ann-benchmarks host rejects urllib's default User-Agent.
        req = urllib.request.Request(url, headers={"User-Agent": "hms-public-bench/1.0"})
        with urllib.request.urlopen(req) as resp, tmp.open("wb") as out:
            while block := resp.read(1 << 20):
                out.write(block)
        tmp.rename(dest)
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_f32(path: Path, a: np.ndarray) -> None:
    np.ascontiguousarray(a, dtype="<f4").tofile(path)


def prepare_ann(name: str, limit: int | None) -> None:
    import h5py

    src = fetch(ANN_URL.format(name=name), root() / "downloads" / f"{name}.hdf5")
    out = root() / name
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"], dtype=np.int32)
        metric = f.attrs.get("distance", "unknown")
    if limit is not None:
        # Ground truth refers to the full train set; a prefix is only valid with recomputed truth.
        raise SystemExit("--limit is not supported for ANN sets: shipped ground truth covers the full set")
    write_f32(out / "train.f32", train)
    write_f32(out / "test.f32", test)
    np.ascontiguousarray(neighbors, dtype="<i4").tofile(out / "neighbors.i32")
    meta = {
        "kind": "ann",
        "name": name,
        "source": ANN_URL.format(name=name),
        "sha256": sha256(src),
        "metric": str(metric),
        "n_train": int(train.shape[0]),
        "n_test": int(test.shape[0]),
        "dim": int(train.shape[1]),
        "n_neighbors": int(neighbors.shape[1]),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


def prepare_beir(name: str) -> None:
    from sentence_transformers import SentenceTransformer

    src = fetch(BEIR_URL.format(name=name), root() / "downloads" / f"{name}.zip")
    extracted = root() / "downloads" / name
    if not extracted.exists():
        with zipfile.ZipFile(src) as z:
            z.extractall(root() / "downloads")
    out = root() / name
    out.mkdir(parents=True, exist_ok=True)

    qrels = {}
    with (extracted / "qrels" / "test.tsv").open() as f:
        next(f)
        for line in f:
            qid, did, rel = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[did] = int(rel)
    corpus = []
    with (extracted / "corpus.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            text = f"{d.get('title', '')}\n{d['text']}".strip()
            corpus.append({"id": d["_id"], "text": text})
    queries = []
    with (extracted / "queries.jsonl").open() as f:
        for line in f:
            q = json.loads(line)
            if q["_id"] in qrels:
                queries.append({"id": q["_id"], "text": q["text"]})

    model = SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cpu")
    emb = lambda texts: model.encode(  # noqa: E731
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True
    )
    write_f32(out / "corpus.f32", emb([d["text"] for d in corpus]))
    write_f32(out / "queries.f32", emb([q["text"] for q in queries]))
    with (out / "corpus.jsonl").open("w") as f:
        for d in corpus:
            f.write(json.dumps(d) + "\n")
    with (out / "queries.jsonl").open("w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    with (out / "qrels.tsv").open("w") as f:
        for qid, docs in qrels.items():
            for did, rel in docs.items():
                f.write(f"{qid}\t{did}\t{rel}\n")
    meta = {
        "kind": "beir",
        "name": name,
        "source": BEIR_URL.format(name=name),
        "sha256": sha256(src),
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "normalization": "l2",
        "n_corpus": len(corpus),
        "n_queries": len(queries),
        "dim": int(model.get_sentence_embedding_dimension()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


def lme_items(entry: dict, granularity: str) -> list[tuple[str, str]]:
    """(corpus id, text) pairs, mirroring process_item_flat_index in LongMemEval's run_retrieval.py."""
    items = []
    for sid, sess in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
        users = [(i, t) for i, t in enumerate(sess) if t["role"] == "user"]
        if granularity == "session":
            cid = sid
            if "answer" in sid and not any(t["has_answer"] for _, t in users):
                cid = sid.replace("answer", "noans")
            items.append((cid, " ".join(t["content"] for _, t in users)))
        else:
            for i, t in users:
                cid = f"{sid}_{i + 1}"
                if "answer" in sid and not t["has_answer"]:
                    cid = cid.replace("answer", "noans")
                items.append((cid, t["content"]))
    return items


def prepare_longmemeval(variant: str, limit: int | None) -> None:
    from sentence_transformers import SentenceTransformer

    url = LME_URL.format(variant=variant)
    src = fetch(url, root() / "downloads" / f"longmemeval_{variant}_cleaned.json")
    data = json.loads(src.read_text())[:limit]
    out = root() / f"longmemeval_{variant}"
    out.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cpu")
    emb = lambda texts: model.encode(  # noqa: E731
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True
    )
    counts = {}
    for gran in ("turn", "session"):
        rows, unique = [], {}
        for qi, entry in enumerate(data):
            for cid, text in lme_items(entry, gran):
                rows.append((qi, cid, text, unique.setdefault(text, len(unique))))
        write_f32(out / f"{gran}.f32", emb(list(unique)))
        with (out / f"{gran}.jsonl").open("w") as f:
            for qi, cid, text, e in rows:
                f.write(json.dumps({"q": qi, "id": cid, "text": text, "e": e}) + "\n")
        counts[gran] = {"items": len(rows), "unique_embeddings": len(unique)}
    write_f32(out / "queries.f32", emb([e["question"] for e in data]))
    with (out / "questions.jsonl").open("w") as f:
        for e in data:
            f.write(json.dumps({"id": e["question_id"], "type": e["question_type"], "text": e["question"]}) + "\n")
    meta = {
        "kind": "longmemeval",
        "name": f"longmemeval_{variant}",
        "release": "xiaowu0162/longmemeval-cleaned",
        "release_revision": LME_REVISION,
        "source": url,
        "sha256": sha256(src),
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "normalization": "l2",
        "max_seq_length": int(model.max_seq_length),
        "n_questions": len(data),
        "counts": counts,
        "dim": int(model.get_sentence_embedding_dimension()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kind", choices=["ann", "beir", "longmemeval"])
    ap.add_argument("name")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.kind == "ann":
        prepare_ann(args.name, args.limit)
    elif args.kind == "longmemeval":
        prepare_longmemeval(args.name, args.limit)
    else:
        prepare_beir(args.name)


if __name__ == "__main__":
    main()
