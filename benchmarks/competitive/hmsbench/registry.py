# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Which systems exist, how each is isolated, and the static operational facts.

Facts are recorded into every results file next to the measurements, each with a `basis`
("measured" with a results file, or "documentation" with the URL), so the generated table never
carries a claim that has no source in the results directory.
"""

import platform

HMS = "hms"
# The HMS text arm runs twice: `hms` ingests with `--encode lexical` (BM25 index only, the
# like-for-like row against the lexical comparators) and `hms-eager` with the default eager
# encoding (chunk hypervectors built at ingest as well). Each build repeat is its own cold
# process in both rows. `hms-eager` is never a comparator in a verdict (same family).
HMS_EAGER = "hms-eager"
HMS_TEXT_ENCODE = {HMS: "lexical", HMS_EAGER: "eager"}

# kind: ann | text.  env: pinned environment (envs.PINS key).  worker: key inside the worker module.
SYSTEMS = {
    HMS: {"display": "HMS", "kind": "both", "env": None, "package": "holographic-memory", "family": "hms"},
    HMS_EAGER: {"display": "HMS (eager hypervectors)", "kind": "text", "env": None, "package": "holographic-memory", "family": "hms"},
    "hnswlib": {"display": "hnswlib", "kind": "ann", "env": "hnswlib", "package": "hnswlib", "family": "hnsw"},
    "faiss-hnsw": {"display": "FAISS HNSW", "kind": "ann", "env": "faiss", "package": "faiss-cpu", "family": "hnsw"},
    "faiss-flat": {"display": "FAISS Flat (exact)", "kind": "ann", "env": "faiss", "package": "faiss-cpu", "family": "exact"},
    "faiss-ivfpq": {"display": "FAISS IVF-PQ (+refine)", "kind": "ann", "env": "faiss", "package": "faiss-cpu", "family": "ivf"},
    "usearch": {"display": "USearch", "kind": "ann", "env": "usearch", "package": "usearch", "family": "hnsw"},
    "annoy": {"display": "Annoy", "kind": "ann", "env": "annoy", "package": "annoy", "family": "trees"},
    "sqlite-vec": {"display": "sqlite-vec (exact)", "kind": "ann", "env": "sqlite-vec", "package": "sqlite-vec", "family": "exact"},
    "lancedb": {"display": "LanceDB", "kind": "ann", "env": "lancedb", "package": "lancedb", "family": "ivf"},
    "chroma": {"display": "Chroma", "kind": "ann", "env": "chroma", "package": "chromadb", "family": "hnsw"},
    "qdrant-local": {"display": "Qdrant (embedded local mode)", "kind": "ann", "env": "qdrant", "package": "qdrant-client", "family": "exact"},
    "milvus-lite": {"display": "Milvus Lite", "kind": "ann", "env": "milvus", "package": "pymilvus", "family": "hnsw"},
    "duckdb-vss": {"display": "DuckDB vss", "kind": "ann", "env": "duckdb", "package": "duckdb", "family": "hnsw"},
    "sqlite-fts5": {"display": "SQLite FTS5", "kind": "text", "env": "stdlib", "package": "sqlite3", "family": "lexical"},
    "lucene": {"display": "Apache Lucene 9.12.3 (JVM)", "kind": "text", "env": None, "package": "lucene-core", "family": "lexical"},
    "tantivy": {"display": "tantivy", "kind": "text", "env": "tantivy", "package": "tantivy", "family": "lexical"},
    "whoosh": {"display": "Whoosh", "kind": "text", "env": "whoosh", "package": "whoosh-reloaded", "family": "lexical"},
    "bm25s": {"display": "bm25s", "kind": "text", "env": "bm25s", "package": "bm25s", "family": "lexical"},
}

# Where each system's query is timed. Binding overhead is part of what a user of that package
# pays, but it differs by language, so it is printed next to every timing row.
TIMING_BOUNDARY = {
    HMS: "native Rust API call in the engine process (no language binding); text: BM25 index only (`--encode lexical`); ANN: public-bench loop",
    HMS_EAGER: "native Rust API call in the engine process (no language binding); text: BM25 index plus chunk hypervectors at ingest (`--encode eager`)",
    "hnswlib": "CPython binding call per query", "faiss-hnsw": "CPython (SWIG) call per query", "faiss-flat": "CPython (SWIG) call per query",
    "faiss-ivfpq": "CPython (SWIG) call per query", "usearch": "CPython binding call per query", "annoy": "CPython binding call per query",
    "sqlite-vec": "CPython sqlite3 call per query", "lancedb": "CPython + Arrow call per query (multi-threaded Rust runtime inside)",
    "chroma": "CPython client call per query (Rust core inside)", "qdrant-local": "CPython client call per query (embedded local mode)",
    "milvus-lite": "CPython client call per query (embedded server process)", "duckdb-vss": "CPython duckdb call per query",
    "sqlite-fts5": "CPython sqlite3 call per query (query string built outside the timer)", "tantivy": "CPython tantivy-py call per query (stored id fetched per hit)",
    "whoosh": "pure Python", "bm25s": "CPython + NumPy (single-threaded retrieve)", "lucene": "in-process JVM API call, after a full untimed JIT warm-up pass",
}

_DOC = "documentation, not exercised here"

FACTS = {
    HMS: {
        "service_needed": {"value": "no", "basis": "measured", "source": "benchmarks/results/security_matrix_v2.json",
                           "note": "in-process library; 0 egress sockets in the security matrix"},
        "encrypted_at_rest": {"value": "yes", "basis": "measured", "source": "benchmarks/results/security_matrix_v2.json",
                              "note": "AES-GCM with Argon2id (feature `security`); encrypted QPS ratio 0.929 of plain; not enabled in the rows above"},
    },
    "hnswlib": {"service_needed": {"value": "no", "basis": _DOC}, "encrypted_at_rest": {"value": "no", "basis": _DOC}},
    "faiss-hnsw": {"service_needed": {"value": "no", "basis": _DOC}, "encrypted_at_rest": {"value": "no", "basis": _DOC}},
    "faiss-flat": {"service_needed": {"value": "no", "basis": _DOC}, "encrypted_at_rest": {"value": "no", "basis": _DOC}},
    "faiss-ivfpq": {"service_needed": {"value": "no", "basis": _DOC}, "encrypted_at_rest": {"value": "no", "basis": _DOC}},
    "usearch": {"service_needed": {"value": "no", "basis": _DOC}, "encrypted_at_rest": {"value": "no", "basis": _DOC}},
    "annoy": {"service_needed": {"value": "no", "basis": _DOC}, "encrypted_at_rest": {"value": "no", "basis": _DOC}},
    "sqlite-vec": {"service_needed": {"value": "no", "basis": _DOC}, "encrypted_at_rest": {"value": "no (SQLCipher is a separate build)", "basis": _DOC}},
    "lancedb": {"service_needed": {"value": "no (embedded)", "basis": _DOC}, "encrypted_at_rest": {"value": "no built-in", "basis": _DOC}},
    "chroma": {"service_needed": {"value": "no (PersistentClient)", "basis": _DOC}, "encrypted_at_rest": {"value": "no built-in", "basis": _DOC}},
    "qdrant-local": {"service_needed": {"value": "no (local mode; server needed for indexed ANN)", "basis": _DOC},
                     "encrypted_at_rest": {"value": "no built-in", "basis": _DOC}},
    "milvus-lite": {"service_needed": {"value": "no (embedded)", "basis": _DOC}, "encrypted_at_rest": {"value": "no built-in", "basis": _DOC}},
    "duckdb-vss": {"service_needed": {"value": "no", "basis": _DOC},
                   "encrypted_at_rest": {"value": "database-file encryption (DuckDB 1.4+); not exercised with the vss index", "basis": _DOC}},
    "sqlite-fts5": {"service_needed": {"value": "no", "basis": _DOC}, "encrypted_at_rest": {"value": "no (SQLCipher is a separate build)", "basis": _DOC}},
    "tantivy": {"service_needed": {"value": "no", "basis": _DOC}, "encrypted_at_rest": {"value": "no", "basis": _DOC}},
    "whoosh": {"service_needed": {"value": "no", "basis": _DOC}, "encrypted_at_rest": {"value": "no", "basis": _DOC}},
    "bm25s": {"service_needed": {"value": "no", "basis": _DOC}, "encrypted_at_rest": {"value": "no", "basis": _DOC}},
    "lucene": {"service_needed": {"value": "no (library; needs a JVM)", "basis": _DOC}, "encrypted_at_rest": {"value": "no (filesystem-level only)", "basis": _DOC}},
}

# Systems named in the brief that were not run, with the reason recorded in every results file.
NOT_RUN = {
    "pgvector": "needs a PostgreSQL server with the extension; not installable from pip/cargo/npm (service required)",
    "mem0": "agent-memory systems belong to the next harness stage (plug in through the same adapter interface)",
    "zep": "agent-memory systems belong to the next harness stage",
    "letta": "agent-memory systems belong to the next harness stage",
    "langmem": "agent-memory systems belong to the next harness stage",
}

FACTS[HMS_EAGER] = FACTS[HMS]

SYSTEMS.update({
    "ngt": {"display": "NGT (ONNG)", "kind": "ann", "env": "ngt", "package": "ngt", "family": "graph"},
    "scann": {"display": "ScaNN", "kind": "ann", "env": "scann", "package": "scann", "family": "ivf"},
    "pynndescent": {"display": "PyNNDescent", "kind": "ann", "env": "pynndescent", "package": "pynndescent", "family": "graph"},
    "voyager": {"display": "Voyager", "kind": "ann", "env": "voyager", "package": "voyager", "family": "hnsw"},
    "diskann": {"display": "DiskANN (in-memory Vamana)", "kind": "ann", "env": "diskann", "package": "diskannpy", "family": "graph"},
})
TIMING_BOUNDARY.update({
    "ngt": "CPython (pybind11) call per query", "scann": "CPython (pybind11) call per query; the ScaNN searcher runs its own thread pool for training only",
    "pynndescent": "CPython + numba-compiled kernel per query (JIT compile inside cold start)",
    "voyager": "CPython 3.12 binding call per query", "diskann": "CPython 3.11 binding call per query (diskannpy StaticMemoryIndex)",
})
FACTS.update({k: {"service_needed": {"value": "no", "basis": _DOC}, "encrypted_at_rest": {"value": "no", "basis": _DOC}} for k in ("ngt", "scann", "pynndescent", "voyager", "diskann")})
# Reasons that apply to one OS only, recorded in results files produced on that OS.
_PLATFORM_NOT_RUN = {"scann": {"Darwin": "no macOS wheel for scann 1.4.2 (manylinux_2_27 x86_64 and aarch64 only); Linux x86_64 runs it"},
                     "diskann": {"Darwin": "no macOS wheel for diskannpy 0.7.0 (manylinux_2_28 x86_64 and win_amd64, CPython 3.9-3.11 only); Linux x86_64 runs it"}}
NOT_RUN.update({k: v[platform.system()] for k, v in _PLATFORM_NOT_RUN.items() if platform.system() in v})

ANN_SYSTEMS = [k for k, v in SYSTEMS.items() if v["kind"] == "ann"]
TEXT_SYSTEMS = [k for k, v in SYSTEMS.items() if v["kind"] == "text"]
