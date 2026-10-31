# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Plain SQLite + FastAPI baseline for deploy/bench/server_bench.py.

The simplest service a team would write without a vector database: documents (id, text,
metadata JSON, float32 embedding blob) persist in SQLite; an in-memory numpy matrix mirrors the
embeddings; a query is an exact cosine scan (matrix @ q) with a partial sort. One uvicorn
worker, no auth. Endpoints mirror warm-server's shapes so the load generator is shared.

Run: uvicorn baseline_sqlite_fastapi:app --port 8099   (DB path from BASELINE_DB)
"""

import json
import os
import sqlite3
import threading

import numpy as np
from fastapi import FastAPI, Request

DIM = int(os.environ.get("BASELINE_DIM", "384"))
DB = os.environ.get("BASELINE_DB", "baseline.sqlite")

app = FastAPI()
lock = threading.Lock()
conn = sqlite3.connect(DB, check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")
conn.execute(
    "CREATE TABLE IF NOT EXISTS docs (row INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, text TEXT, metadata TEXT, vec BLOB)"
)
conn.commit()

ids: list[str] = []
index: dict[str, int] = {}
matrix = np.zeros((1024, DIM), dtype=np.float32)  # capacity-doubling mirror of the embeddings


def normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def put(doc_id: str, vec: np.ndarray) -> None:
    global matrix
    slot = index.get(doc_id)
    if slot is None:
        slot = len(ids)
        if slot == len(matrix):
            matrix = np.vstack([matrix, np.zeros_like(matrix)])
        ids.append(doc_id)
        index[doc_id] = slot
    matrix[slot] = vec


for _id, _blob in conn.execute("SELECT id, vec FROM docs ORDER BY row"):
    put(_id, np.frombuffer(_blob, dtype=np.float32))


@app.post("/documents/batch")
async def add_batch(request: Request):
    docs = await request.json()
    with lock:
        rows = []
        for d in docs:
            v = normalise(np.asarray(d["vector"], dtype=np.float32))
            rows.append((d["id"], d.get("text", ""), json.dumps(d.get("metadata", {})), v.tobytes()))
            put(d["id"], v)
        conn.executemany(
            "INSERT OR REPLACE INTO docs (id, text, metadata, vec) VALUES (?, ?, ?, ?)", rows
        )
        conn.commit()
    return {"added": len(docs)}


@app.post("/query")
async def query(request: Request):
    body = await request.json()
    q = normalise(np.asarray(body["query_vector"], dtype=np.float32))
    k = int(body.get("top_k", 3))
    scores = matrix[: len(ids)] @ q
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
    top = top[np.argsort(-scores[top])]
    out = []
    for i in top:
        row = conn.execute("SELECT text, metadata FROM docs WHERE id = ?", (ids[i],)).fetchone()
        out.append({"id": ids[i], "text": row[0], "metadata": json.loads(row[1]), "score": float(scores[i])})
    return {"matches": out}


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "docs": len(ids)}
