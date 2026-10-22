# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Server-mode systems: registry entries merged into registry.py.

Server rows are measured against `hms-server` (the HTTP service, feature `server`) and appear in
their own table sections, never next to embedded rows. Kind `server-ann` / `server-text` keeps
them out of the embedded ANN_SYSTEMS / TEXT_SYSTEMS lists.
"""

HMS_SERVER = "hms-server"
ENV = "server-clients"

SERVER_SYSTEMS = {
    HMS_SERVER: {"display": "hms-server (HTTP)", "kind": "server-ann", "env": ENV, "engine": "hms-server", "package": "holographic-memory", "family": "hms"},
    "qdrant-server": {"display": "Qdrant server", "kind": "server-ann", "env": ENV, "engine": "qdrant", "package": "qdrant", "family": "hnsw"},
    "weaviate": {"display": "Weaviate", "kind": "server-ann", "env": ENV, "engine": "weaviate", "package": "weaviate", "family": "hnsw"},
    "opensearch-knn": {"display": "OpenSearch k-NN (faiss HNSW)", "kind": "server-ann", "env": ENV, "engine": "opensearch", "package": "opensearch", "family": "hnsw"},
    "pgvector": {"display": "PostgreSQL + pgvector", "kind": "server-ann", "env": ENV, "engine": "postgres", "package": "pgvector", "family": "hnsw"},
    "redis-vss": {"display": "Redis 8 query engine", "kind": "server-ann", "env": ENV, "engine": "redis", "package": "redis", "family": "hnsw"},
    "meilisearch": {"display": "Meilisearch", "kind": "server-text", "env": ENV, "engine": "meilisearch", "package": "meilisearch", "family": "lexical"},
    "typesense": {"display": "Typesense", "kind": "server-text", "env": ENV, "engine": "typesense", "package": "typesense", "family": "lexical"},
    "opensearch-bm25": {"display": "OpenSearch BM25", "kind": "server-text", "env": ENV, "engine": "opensearch", "package": "opensearch", "family": "lexical"},
    "pg-tsvector": {"display": "PostgreSQL tsvector (ts_rank)", "kind": "server-text", "env": ENV, "engine": "postgres", "package": "postgresql", "family": "lexical"},
    "paradedb": {"display": "ParadeDB pg_search (BM25)", "kind": "server-text", "env": ENV, "engine": "paradedb", "package": "pg_search", "family": "lexical"},
}

_HTTP = "one persistent HTTP/1.1 keep-alive connection from a CPython client on loopback; request body serialized before the timer, response JSON decoded inside it"
SERVER_TIMING_BOUNDARY = {
    HMS_SERVER: f"{_HTTP}; ANN: POST /api/v1/query with index=graph and per-request ef; text: POST /api/v1/memory/recall, lexical weight only",
    "qdrant-server": f"{_HTTP}; POST /collections/c/points/query (REST, not gRPC); max_search_threads=1",
    "weaviate": f"{_HTTP}; GraphQL nearVector (REST, not gRPC); ef set on the class between sweep points",
    "opensearch-knn": f"{_HTTP}; _search knn query with method_parameters.ef_search; concurrent segment search off, one shard",
    "pgvector": "psycopg 3 over one TCP connection, server-side prepared statement; max_parallel_workers_per_gather=0; enable_seqscan=off",
    "redis-vss": "redis-py over one TCP connection, FT.SEARCH KNN with NOCONTENT; query engine WORKERS 0 (main thread)",
    "meilisearch": f"{_HTTP}; POST /indexes/docs/search (its own ranking rules, not BM25; at most 10 query words are used)",
    "typesense": f"{_HTTP}; GET search with drop_tokens_threshold=100 (its own text-match ranking, not BM25)",
    "opensearch-bm25": f"{_HTTP}; _search match query (OR), english analyzer, default BM25; one shard",
    "pg-tsvector": "psycopg 3 over one TCP connection, server-side prepared statement; OR of the query words via to_tsquery, ranked by ts_rank (not BM25)",
    "paradedb": "psycopg 3 over one TCP connection, server-side prepared statement; `|||` (match any word) over an english-stemmed field, BM25 via pdb.score",
}

_DOC = "documentation, not exercised here"
SERVER_FACTS = {
    HMS_SERVER: {"service_needed": {"value": "yes (this row; the library needs none)", "basis": _DOC},
                 "encrypted_at_rest": {"value": "yes with feature `security` (not enabled in these rows)", "basis": _DOC}},
    "qdrant-server": {"service_needed": {"value": "yes", "basis": _DOC}, "encrypted_at_rest": {"value": "no built-in", "basis": _DOC}},
    "weaviate": {"service_needed": {"value": "yes", "basis": _DOC}, "encrypted_at_rest": {"value": "no built-in", "basis": _DOC}},
    "opensearch-knn": {"service_needed": {"value": "yes (JVM)", "basis": _DOC}, "encrypted_at_rest": {"value": "no built-in (filesystem level)", "basis": _DOC}},
    "pgvector": {"service_needed": {"value": "yes", "basis": _DOC}, "encrypted_at_rest": {"value": "no built-in (filesystem level)", "basis": _DOC}},
    "redis-vss": {"service_needed": {"value": "yes", "basis": _DOC}, "encrypted_at_rest": {"value": "no", "basis": _DOC}},
    "meilisearch": {"service_needed": {"value": "yes", "basis": _DOC}, "encrypted_at_rest": {"value": "no built-in", "basis": _DOC}},
    "typesense": {"service_needed": {"value": "yes", "basis": _DOC}, "encrypted_at_rest": {"value": "no built-in", "basis": _DOC}},
    "opensearch-bm25": {"service_needed": {"value": "yes (JVM)", "basis": _DOC}, "encrypted_at_rest": {"value": "no built-in (filesystem level)", "basis": _DOC}},
    "pg-tsvector": {"service_needed": {"value": "yes", "basis": _DOC}, "encrypted_at_rest": {"value": "no built-in (filesystem level)", "basis": _DOC}},
    "paradedb": {"service_needed": {"value": "yes (PostgreSQL extension)", "basis": _DOC}, "encrypted_at_rest": {"value": "no built-in (filesystem level)", "basis": _DOC}},
}

# What a replace/delete acknowledgement means for each server, as configured here.
DURABILITY = {
    HMS_SERVER: "one journal frame per request, msync(MS_SYNC) of the mapped segment before the response, then fcntl(F_FULLFSYNC) on macOS",
    "qdrant-server": "wait=true: applied to the WAL and segment before the response; WAL fsync is periodic, not per request",
    "weaviate": "acknowledged after the write-ahead log append; per-request fsync not documented",
    "opensearch-knn": "translog fsync per request (index.translog.durability=request, the default); refresh=true makes it visible",
    "opensearch-bm25": "translog fsync per request (index.translog.durability=request, the default); refresh=true makes it visible",
    "pgvector": "autocommit; synchronous_commit=on, WAL flushed per commit (wal_sync_method=fsync_writethrough on macOS, the default elsewhere)",
    "pg-tsvector": "autocommit; synchronous_commit=on, WAL flushed per commit (wal_sync_method=fsync_writethrough on macOS, the default elsewhere)",
    "paradedb": "autocommit; synchronous_commit=on, WAL flushed per commit (wal_sync_method=fsync_writethrough on macOS, the default elsewhere)",
    "redis-vss": "appendonly yes, appendfsync always (plain fsync; no F_FULLFSYNC on macOS)",
    "meilisearch": "the enqueued task is polled until it succeeded (LMDB commit)",
    "typesense": "acknowledged after the write is applied; per-request fsync not documented",
}

SERVER_NOT_RUN = {
    "milvus-standalone": "no non-Docker artifact: releases 3.0.3 and 2.6.25 ship only Docker Compose files; the last native package "
                         "(milvus_2.6.4 .deb/.rpm, Linux only) is superseded",
}

SERVER_PINS = {ENV: ["psycopg[binary]==3.3.6", "redis==8.1.0"]}

SERVER_ANN_SYSTEMS = [k for k, v in SERVER_SYSTEMS.items() if v["kind"] == "server-ann"]
# hms-server serves both kinds; its text row goes through the memory API (server_worker.HmsServerText).
SERVER_TEXT_SYSTEMS = [HMS_SERVER] + [k for k, v in SERVER_SYSTEMS.items() if v["kind"] == "server-text"]
