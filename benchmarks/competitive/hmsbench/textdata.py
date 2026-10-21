# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Text datasets as the `{documents, queries}` input JSON plus qrels, identical for every system.

beir:<name>   BEIR train/dev qrels (the existing document-competition protocol: SciFact train,
              NFCorpus dev, FiQA dev; the test qrels stay unread). One chunk per document.
memory        Agent-memory-shaped: every session of the 100 LongMemEval_S dev questions as one
              searchable memory (answer sessions are the relevant documents); the 400 held-out
              questions are never read.
"""

import json
import urllib.request
import zipfile

from .common import CACHE, ROOT

DATA = CACHE / "competitive" / "data"
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
LME_URL = ("https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/"
           "98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json")
BEIR_SPLITS = {"scifact": "train", "nfcorpus": "dev", "fiqa": "dev"}
MAX_WORDS = 4000
K = 100


def _fetch(url, dest):
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "hms-bench/1.0"})
        with urllib.request.urlopen(request) as response, tmp.open("wb") as out:
            while block := response.read(1 << 20):
                out.write(block)
        tmp.rename(dest)
    return dest


def _options():
    return {"k": K, "candidateLimit": K, "lexicalWeight": 1, "semanticWeight": 0}


def _document(identifier, text, bucket):
    return {"id": identifier, "text": text, "metadata": {"bucket": bucket},
            "chunkWords": 4096, "overlapWords": 0, "storeText": False}


def beir(name):
    split = BEIR_SPLITS[name]
    folder = DATA / f"text-{name}-{split}"
    if (folder / "input.json").exists():
        return folder
    source = CACHE / "downloads" / name
    if not (source / "corpus.jsonl").exists():
        archive = _fetch(BEIR_URL.format(name=name), CACHE / "downloads" / f"{name}.zip")
        with zipfile.ZipFile(archive) as z:
            z.extractall(CACHE / "downloads")
    qrels = {}
    with (source / "qrels" / f"{split}.tsv").open() as stream:
        next(stream)
        for line in stream:
            qid, did, grade = line.rstrip().split("\t")
            qrels.setdefault(qid, {})[did] = int(grade)
    documents = []
    # FiQA has whitespace-only documents; no lexical system can match them, so every system
    # gets the corpus without them and meta.json records how many were left out.
    empty = 0
    for line in (source / "corpus.jsonl").open():
        row = json.loads(line)
        text = (row.get("title", "") + "\n" + row["text"]).strip()
        if not text:
            empty += 1
            continue
        if len(text.split()) > 4096:
            raise ValueError("one complete chunk per document is required")
        documents.append(_document(row["_id"], text, len(row["_id"]) % 17))
    queries = [{"id": q["_id"], "text": q["text"], "options": _options()}
               for q in map(json.loads, (source / "queries.jsonl").open()) if q["_id"] in qrels]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "input.json").write_text(json.dumps({"documents": documents, "queries": queries}) + "\n")
    (folder / "qrels.json").write_text(json.dumps(qrels) + "\n")
    (folder / "meta.json").write_text(json.dumps({"dataset": name, "split": split, "documents": len(documents),
                                                  "empty_documents_skipped": empty, "queries": len(queries),
                                                  "source": BEIR_URL.format(name=name)}) + "\n")
    return folder


def memory():
    folder = DATA / "text-memory-lme-dev"
    if (folder / "input.json").exists():
        return folder
    raw = _fetch(LME_URL, CACHE / "downloads" / "longmemeval_s_cleaned.json")
    dev = set(json.loads((ROOT / "benchmarks/public/longmemeval_split.json").read_text())["dev"])
    sessions, queries, qrels = {}, [], {}
    for entry in json.loads(raw.read_text()):
        if entry["question_id"] not in dev:
            continue
        answers = {s: 1 for s in entry["answer_session_ids"]}
        if not answers:
            continue
        for sid, turns in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
            if sid not in sessions:
                text = "\n".join(f'{t["role"]}: {t["content"]}' for t in turns)
                sessions[sid] = " ".join(text.split()[:MAX_WORDS])
        queries.append({"id": entry["question_id"], "text": entry["question"], "options": _options()})
        qrels[entry["question_id"]] = answers
    documents = [_document(sid, text, len(sid) % 17) for sid, text in sessions.items()]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "input.json").write_text(json.dumps({"documents": documents, "queries": queries}) + "\n")
    (folder / "qrels.json").write_text(json.dumps(qrels) + "\n")
    (folder / "meta.json").write_text(json.dumps({"dataset": "longmemeval-s-dev-sessions", "documents": len(documents),
                                                  "queries": len(queries), "truncated_words": MAX_WORDS, "source": LME_URL}) + "\n")
    return folder


def load(dataset):
    if dataset.startswith("beir:"):
        return beir(dataset.split(":", 1)[1])
    if dataset == "memory":
        return memory()
    raise ValueError(f"unknown text dataset {dataset}")
