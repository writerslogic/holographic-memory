# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Five-minute quickstart: ingest a corpus, search it, compare with SQLite FTS5, mask vectors.

From the repository root:

    cargo build --release --features server --bin warm-server
    WARM_API_KEY=quickstart target/release/warm-server --input-dim 384 &
    WARM_API_KEY=quickstart uv run --with bindings/python/warm-sdk \
        python examples/quickstart/python/quickstart.py

The Python package is an HTTP client for warm-server, and the server stores vectors you supply.
`toy_embed` below is a deliberately tiny lexical stand-in (hashed bag of words), so this
script needs no model download. Swap in a real embedding model for semantic search.
"""

import hashlib
import json
import math
import os
import re
import sqlite3
import time
from pathlib import Path

from writerslogic_warm_sdk import WarmClient

HERE = Path(__file__).resolve().parent.parent
URL = os.environ.get("WARM_URL", "http://127.0.0.1:8080")
API_KEY = os.environ.get("WARM_API_KEY")
DIM = 384


def toy_embed(text: str) -> list[float]:
    """Signed hashed bag of words, L2-normalised. Strips a few English suffixes."""
    vec = [0.0] * DIM
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        stem = re.sub(r"(ing|ed|es|s)$", "", word) if len(word) > 4 else word
        digest = hashlib.sha256(stem.encode()).digest()
        index = int.from_bytes(digest[:4], "little") % DIM
        vec[index] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def fts5_query(query: str) -> str:
    """FTS5 treats a bare multi-word query as AND; OR of the words is the fair lexical baseline."""
    return " OR ".join(f'"{w}"' for w in re.findall(r"\w+", query))


corpus = [json.loads(line) for line in (HERE / "corpus.jsonl").read_text().splitlines()]
queries = json.loads((HERE / "queries.json").read_text())
tenant = f"quickstart-{int(time.time())}"  # a tenant is a store; a fresh one per run

# --8<-- [start:ingest]
# 1. Ingest. The server keeps one store per tenant; you send the text and a vector per document.
docs = [
    {"id": d["id"], "vector": toy_embed(d["text"]), "text": d["text"], "metadata": {"title": d["title"]}}
    for d in corpus
]
client = WarmClient(url=URL, api_key=API_KEY, tenant_id=tenant)
started = time.perf_counter()
client.add_documents(docs)
print(f"ingested {len(docs)} documents in {(time.perf_counter() - started) * 1000:.1f} ms")
# --8<-- [end:ingest]

# --8<-- [start:compare]
# The same corpus in SQLite FTS5 (bm25 ranking), for comparison.
db = sqlite3.connect(":memory:")
db.execute("CREATE VIRTUAL TABLE docs USING fts5(id UNINDEXED, text)")
db.executemany("INSERT INTO docs (id, text) VALUES (?, ?)", [(d["id"], d["text"]) for d in corpus])

# 2. Search both engines with the same queries.
print(f"\n{'query':<58} {'WARM':<9} {'FTS5':<9} expected")
warm_hits = fts_hits = 0
for q in queries:
    matches = client.query(toy_embed(q["query"]), top_k=3)["matches"]
    warm = matches[0]["id"] if matches else "-"
    row = db.execute(
        "SELECT id FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT 1", (fts5_query(q["query"]),)
    ).fetchone()
    fts = row[0] if row else "-"
    warm_hits += warm == q["expect"]
    fts_hits += fts == q["expect"]
    print(f"{q['query']:<58} {warm:<9} {fts:<9} {q['expect']}")
print(f"top-1 correct: WARM {warm_hits}/{len(queries)}, FTS5 {fts_hits}/{len(queries)}")
assert warm_hits >= 4, f"WARM found the expected article for only {warm_hits} queries"
# --8<-- [end:compare]

# --8<-- [start:hit]
# Hits carry the stored text, metadata and the exact cosine score.
top = client.query(toy_embed("asylum from persecution"), top_k=1)["matches"][0]
print(f"\n{top['id']} score {top['score']:.3f}: {top['text']}")
# --8<-- [end:hit]
client.close()

# --8<-- [start:encrypt]
# 3. Protect. warm-server has no encryption at rest, so the client masks vectors with a key the
# server never sees. This is obfuscation, not encryption: the server still learns every pairwise
# similarity and stores text and metadata in plaintext (docs/SERVER.md).
masked = WarmClient(
    url=URL, api_key=API_KEY, tenant_id=tenant + "-masked", zero_trust_key="correct horse battery staple"
)
masked.add_documents(docs)
plain_vec = toy_embed("paid holidays")
assert masked.encrypt_vector(plain_vec) != plain_vec
top = masked.query(plain_vec, top_k=1)["matches"][0]
print(f"masked store, top hit for 'paid holidays': {top['id']}")
assert top["id"] == "udhr-24"

wrong = WarmClient(url=URL, api_key=API_KEY, tenant_id=tenant + "-masked", zero_trust_key="wrong key")
top = wrong.query(toy_embed("paid holidays"), top_k=1)["matches"][0]
print(f"same store queried with the wrong key, top hit: {top['id']}")
assert top["id"] != "udhr-24"
# --8<-- [end:encrypt]
masked.close()
wrong.close()
