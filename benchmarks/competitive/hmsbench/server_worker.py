# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Server-mode comparators: runs inside the `server-clients` environment, one fresh process per
phase, and owns the server process tree for that phase (server_proc.Server).

  build  start the server on an empty data directory, create the index, ingest, wait until it is
         queryable (optimizer idle, refreshed, merged, warmed up), stop it cleanly
  serve  start the server on the built data directory; cold start is process start to the first
         answer; then the recall/QPS/latency sweep (ANN) or the query passes (text)
  ops    start the server on a copy, time 100 replacements and 100 deletions, each acknowledged
         by the server under its durability setting (recorded per system in DURABILITY)

Protocol: loopback TCP, one client, one persistent connection, sequential queries, timed in this
client around the full request (send, server work, receive, response decode). Request bodies are
serialized before the timer. Server RSS is sampled outside timed loops (server_proc).
"""

import argparse
import http.client
import json
import os
import platform
import shutil
import socket
import sys
import time
import urllib.parse
import uuid
from pathlib import Path

T_IMPORT = time.time_ns()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import server_proc  # noqa: E402
import servers  # noqa: E402
from ann_worker import EF_SWEEP, K, NPROBE_SWEEP, Unsupported, load_rows, normalize, pctl, peak_rss  # noqa: E402
from text_worker import words  # noqa: E402

TEXT_K = 100
DARWIN = platform.system() == "Darwin"


class HttpError(RuntimeError):
    pass


class Http:
    """One persistent HTTP/1.1 connection. A dropped keep-alive connection is reopened once."""

    def __init__(self, port, headers=None, timeout=3600):
        self.port, self.headers, self.timeout = port, headers or {}, timeout
        self.conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)

    def raw(self, method, path, body=None, content_type="application/json"):
        headers = dict(self.headers)
        if body is not None:
            headers["Content-Type"] = content_type
        for attempt in (0, 1):
            try:
                self.conn.request(method, path, body=body, headers=headers)
                response = self.conn.getresponse()
                data = response.read()
                break
            except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError, http.client.CannotSendRequest):
                self.conn.close()
                self.conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
                if attempt:
                    raise
        if response.status >= 400:
            raise HttpError(f"{method} {path} -> {response.status}: {data[:600]!r}")
        return response.status, data

    def call(self, method, path, obj=None, body=None, content_type="application/json"):
        if obj is not None:
            body = json.dumps(obj).encode()
        _, data = self.raw(method, path, body, content_type)
        return json.loads(data) if data else None

    def probe(self, path, ok=lambda status, data: status < 400):
        """Readiness check on a fresh connection (the server may not be listening yet)."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request("GET", path, headers=self.headers)
            response = conn.getresponse()
            return ok(response.status, response.read())
        finally:
            conn.close()


def wait_until(predicate, timeout, interval=0.02, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise TimeoutError(f"{what} not reached within {timeout}s")


class ServerSystem:
    """One server-backed system. Subclasses set ENGINE and implement the server command line,
    readiness, index creation, ingest and queries."""

    ENGINE = None
    BUILDS = {"default": {}}
    SWEEPS = {"default": [{}]}
    READY_TIMEOUT = 300

    def __init__(self, workdir, threads, phase, dim=None, angular=True):
        self.workdir, self.threads, self.phase, self.dim, self.angular = Path(workdir), threads, phase, dim, angular
        self.data = self.workdir / "data"
        self.logs = self.workdir / "logs"
        self.server = None

    # --- lifecycle
    def command(self):
        """(argv, env, cwd) of the server for this phase; allocates its ports."""
        raise NotImplementedError

    def ready(self):
        raise NotImplementedError

    def ports(self, count):
        return server_proc.stable_ports(self.workdir / "ports.txt", count)

    def after_start(self, params):
        """Work a restarted server needs before its index answers (loading, warm-up)."""

    def start(self):
        self.data.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.pre_start()
        argv, env, cwd = self.command()
        # As root (Modal), the server runs as an unprivileged user and owns its whole workdir
        # (data, logs, OpenSearch's config copy).
        server_proc.chown_for_server(self.workdir)
        self.server = server_proc.Server(f"{self.ENGINE}-{self.phase}", argv, self.logs, env=env, cwd=cwd).start()
        self.server.wait_ready(self.ready, timeout=self.READY_TIMEOUT)
        self.connect()
        return self.server

    def pre_start(self):
        pass

    def connect(self):
        pass

    def stop(self):
        if self.server is not None:
            self.close()
            self.server.stop()

    def close(self):
        pass

    # --- operations
    def create(self, params):
        pass

    def wait_queryable(self, params):
        pass

    def configure(self, sp):
        pass

    def replace(self, ids, vectors):
        raise Unsupported("no replacement")

    def delete(self, ids):
        raise Unsupported("no deletion")


# ------------------------------------------------------------------ hms-server

def hms_server_binary():
    target = Path(os.environ.get("CARGO_TARGET_DIR", HERE.parents[2] / "target"))
    path = Path(os.environ.get("HMS_BENCH_HMS_SERVER", target / "release" / "hms-server"))
    if not path.exists():
        raise servers.NotAvailable(f"{path} missing; build with `cargo build --release --features server --bin hms-server`")
    return path


class HmsServer(ServerSystem):
    ENGINE = "hms-server"
    BUILDS = {"graph": {"index": "graph", "build": "server default (BuildParams::default, VGraph::from_graph)", "ingest_encode": "lexical"}}
    SWEEPS = {"graph": [{"ef": e} for e in EF_SWEEP]}

    def command(self):
        (self.port,) = self.ports(1)
        argv = [hms_server_binary(), "--bind", f"127.0.0.1:{self.port}", "--data-dir", self.data, "--no-auth",
                "--input-dim", self.dim, "--ingest-encode", "lexical", "--graph-min-rows", "1", "--tenant-idle-seconds", "0",
                "--max-top-k", "100", "--log-level", "warn", "--compaction-interval-seconds", "0"]
        return argv, {"RAYON_NUM_THREADS": str(self.threads)}, None

    def ready(self):
        return Http(self.port).probe("/readyz")

    def connect(self):
        self.http = Http(self.port)

    def ingest(self, train, params):
        for lo in range(0, len(train), 1000):
            batch = [{"id": str(i), "vector": train[i].tolist()} for i in range(lo, min(lo + 1000, len(train)))]
            self.http.call("POST", "/api/v1/documents/batch", batch)

    def wait_queryable(self, params):
        # Maintenance builds the persisted graph at build time; after a restart the server
        # opens it and the call is a no-op, still sent so cold start covers the check.
        self.http.call("POST", "/api/v1/index/maintain", {})

    after_start = wait_queryable

    def configure(self, sp):
        self.ef = sp["ef"]

    def prepare(self, q):
        return json.dumps({"query_vector": q.tolist(), "top_k": K, "index": "graph", "ef": self.ef}).encode()

    def query(self, body):
        return [int(m["id"]) for m in self.http.call("POST", "/api/v1/query", body=body)["matches"]]

    def replace(self, ids, vectors):
        for i, v in zip(ids, vectors):
            self.http.call("POST", "/api/v1/documents", {"id": str(int(i)), "vector": v.tolist()})

    def delete(self, ids):
        for i in ids:
            self.http.call("DELETE", f"/api/v1/documents/{int(i)}")


class HmsServerText(HmsServer):
    """Lexical text: `remember` (memory API) stores each document as a fact; queries go to
    `/api/v1/search`, document search's BM25 ranking returning ids and scores only, with the
    same options as the embedded `hms` row (equal to `recall` with lexicalWeight 1 /
    semanticWeight 0 on facts without validity or supersession metadata)."""

    BUILDS = {"default": {}}

    def command(self):
        argv, env, cwd = super().command()
        argv[argv.index("--input-dim") + 1] = "384"
        # Replacing a document means re-remembering its id with new text (the default rejects it).
        config = self.workdir / "memory.json"
        config.write_text(json.dumps({"onIdConflict": "replace"}))
        return argv + ["--memory-config", config], env, cwd

    def ingest(self, documents, params=None):
        for lo in range(0, len(documents), 1000):
            facts = [{"id": d["id"], "text": d["text"]} for d in documents[lo:lo + 1000]]
            self.http.call("POST", "/api/v1/memory/remember", {"facts": facts})

    def wait_queryable(self, params):
        pass

    def after_start(self, params):
        pass

    def prepare(self, text):
        return json.dumps({"query": text, "top_k": TEXT_K}).encode()

    def search(self, body):
        return [(h["id"], h["score"]) for h in self.http.call("POST", "/api/v1/search", body=body)["hits"]]

    def replace_doc(self, doc, n):
        self.http.call("POST", "/api/v1/memory/remember", {"id": doc["id"], "text": doc["text"] + " benchmarkrevision"})

    def delete_doc(self, doc, n):
        self.http.call("POST", "/api/v1/memory/forget", {"scope": {"kind": "ids", "ids": [doc["id"]]}})


# ------------------------------------------------------------------ Qdrant

class QdrantServer(ServerSystem):
    ENGINE = "qdrant"
    # m32e500 matches the HMS graph's build effort (degree 64 at level 0, construction ef 500).
    BUILDS = {"m16": {"m": 16, "ef_construct": 100}, "m32e500": {"m": 32, "ef_construct": 500}}
    SWEEPS = {k: [{"ef": e} for e in EF_SWEEP] for k in BUILDS}

    def command(self):
        self.port, grpc = self.ports(2)
        env = {"QDRANT__SERVICE__HOST": "127.0.0.1", "QDRANT__SERVICE__HTTP_PORT": str(self.port), "QDRANT__SERVICE__GRPC_PORT": str(grpc),
               "QDRANT__STORAGE__STORAGE_PATH": str(self.data / "storage"), "QDRANT__STORAGE__SNAPSHOTS_PATH": str(self.data / "snapshots"),
               "QDRANT__TELEMETRY_DISABLED": "true", "QDRANT__CLUSTER__ENABLED": "false", "QDRANT__LOG_LEVEL": "WARN",
               "QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS": "1",
               "QDRANT__STORAGE__PERFORMANCE__MAX_OPTIMIZATION_THREADS": str(self.threads)}
        return [servers.artifact_binary("qdrant")], env, self.workdir

    def ready(self):
        return Http(self.port).probe("/readyz")

    def connect(self):
        self.http = Http(self.port)

    def create(self, params):
        self.http.call("PUT", "/collections/c", {
            "vectors": {"size": self.dim, "distance": "Dot" if self.angular else "Euclid"},
            # full_scan_threshold and indexing_threshold (KB) are lowered so a small benchmark
            # collection is indexed and searched through the graph instead of a plain scan; one
            # target segment holds one graph, as in every other row.
            "hnsw_config": {"m": params["m"], "ef_construct": params["ef_construct"], "max_indexing_threads": self.threads,
                            "full_scan_threshold": 10},
            "optimizers_config": {"default_segment_number": 1, "indexing_threshold": 1}})

    def ingest(self, train, params):
        for lo in range(0, len(train), 1000):
            hi = min(lo + 1000, len(train))
            self.http.call("PUT", "/collections/c/points?wait=true", {"batch": {"ids": list(range(lo, hi)), "vectors": train[lo:hi].tolist()}})

    def _info(self):
        return self.http.call("GET", "/collections/c")["result"]

    def wait_queryable(self, params):
        # Green with the optimizer idle, then an explicit optimizer trigger (an empty optimizers
        # PATCH, Qdrant's documented way to run pending optimizations) until the segment count is
        # stable, so the stopped collection holds no pending merge.
        previous = None
        while True:
            wait_until(lambda: self._info()["status"] == "green", 7200, 0.1, "qdrant collection green")
            segments = self._info()["segments_count"]
            if segments == previous:
                break
            previous = segments
            self.http.call("PATCH", "/collections/c", {"optimizers_config": {}})
            time.sleep(0.5)
        self.segments = segments

    def after_start(self, params):
        # A restarted collection reports grey (pending work waits for an update) or green;
        # either serves every point. Optimizations are not triggered while serving.
        wait_until(lambda: self._info()["status"] in ("green", "grey"), 7200, 0.05, "qdrant collection loaded")

    def configure(self, sp):
        self.ef = sp["ef"]

    def prepare(self, q):
        return json.dumps({"query": q.tolist(), "limit": K, "params": {"hnsw_ef": self.ef, "exact": False}}).encode()

    def query(self, body):
        return [p["id"] for p in self.http.call("POST", "/collections/c/points/query", body=body)["result"]["points"]]

    def replace(self, ids, vectors):
        for i, v in zip(ids, vectors):
            self.http.call("PUT", "/collections/c/points?wait=true", {"points": [{"id": int(i), "vector": v.tolist()}]})

    def delete(self, ids):
        for i in ids:
            self.http.call("POST", "/collections/c/points/delete?wait=true", {"points": [int(i)]})


# ------------------------------------------------------------------ Weaviate

class WeaviateServer(ServerSystem):
    ENGINE = "weaviate"
    BUILDS = {"m32e128": {"maxConnections": 32, "efConstruction": 128}, "m32e500": {"maxConnections": 32, "efConstruction": 500}}
    SWEEPS = {k: [{"ef": e} for e in EF_SWEEP] for k in BUILDS}

    def command(self):
        self.port, grpc, gossip, data, raft, raft_rpc = self.ports(6)
        env = {"PERSISTENCE_DATA_PATH": str(self.data), "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED": "true",
               "DEFAULT_VECTORIZER_MODULE": "none", "ENABLE_MODULES": "", "DISABLE_TELEMETRY": "true", "LOG_LEVEL": "warning",
               "CLUSTER_HOSTNAME": "node1", "CLUSTER_GOSSIP_BIND_PORT": str(gossip), "CLUSTER_DATA_BIND_PORT": str(data),
               "RAFT_PORT": str(raft), "RAFT_INTERNAL_RPC_PORT": str(raft_rpc), "RAFT_BOOTSTRAP_EXPECT": "1", "RAFT_JOIN": f"node1:{raft}",
               "GRPC_PORT": str(grpc), "QUERY_DEFAULTS_LIMIT": "10", "ASYNC_INDEXING": "false",
               # The read-only guard trips on a nearly full host disk; it is a safety stop, not a
               # performance setting, so it is lifted for the benchmark.
               "DISK_USE_READONLY_PERCENTAGE": "100", "DISK_USE_WARNING_PERCENTAGE": "100"}
        return [servers.artifact_binary("weaviate"), "--host", "127.0.0.1", "--port", str(self.port), "--scheme", "http"], env, self.workdir

    def ready(self):
        return Http(self.port).probe("/v1/.well-known/ready")

    def connect(self):
        self.http = Http(self.port)

    def create(self, params):
        self.http.call("POST", "/v1/schema", {
            "class": "Vec", "vectorizer": "none", "vectorIndexType": "hnsw",
            "vectorIndexConfig": {"distance": "dot" if self.angular else "l2-squared", "ef": -1, **params},
            "properties": [{"name": "n", "dataType": ["int"]}]})

    def ingest(self, train, params):
        for lo in range(0, len(train), 1000):
            objects = [{"class": "Vec", "id": str(uuid.UUID(int=i)), "properties": {"n": i}, "vector": train[i].tolist()}
                       for i in range(lo, min(lo + 1000, len(train)))]
            out = self.http.call("POST", "/v1/batch/objects", {"objects": objects})
            errors = [o["result"]["errors"] for o in out if o.get("result", {}).get("errors")]
            if errors:
                raise RuntimeError(f"weaviate batch errors: {errors[:3]}")

    def _indexed(self):
        nodes = self.http.call("GET", "/v1/nodes?output=verbose")["nodes"]
        shards = [s for n in nodes for s in (n.get("shards") or []) if s.get("class") == "Vec"]
        return shards and all(s.get("vectorIndexingStatus") == "READY" and not s.get("vectorQueueLength") for s in shards)

    def wait_queryable(self, params):
        wait_until(self._indexed, 7200, 0.1, "weaviate shards READY with empty vector queues")

    after_start = wait_queryable

    def configure(self, sp):
        schema = self.http.call("GET", "/v1/schema/Vec")
        schema["vectorIndexConfig"]["ef"] = sp["ef"]
        self.http.call("PUT", "/v1/schema/Vec", schema)

    def prepare(self, q):
        vec = ",".join(repr(float(x)) for x in q)
        return json.dumps({"query": "{Get{Vec(nearVector:{vector:[%s]},limit:%d){n}}}" % (vec, K)}).encode()

    def query(self, body):
        out = self.http.call("POST", "/v1/graphql", body=body)
        if out.get("errors"):
            raise RuntimeError(f"weaviate query errors: {out['errors'][:2]}")
        return [o["n"] for o in out["data"]["Get"]["Vec"]]

    def replace(self, ids, vectors):
        for i, v in zip(ids, vectors):
            oid = str(uuid.UUID(int=int(i)))
            self.http.call("PUT", f"/v1/objects/Vec/{oid}", {"class": "Vec", "id": oid, "properties": {"n": int(i)}, "vector": v.tolist()})

    def delete(self, ids):
        for i in ids:
            self.http.call("DELETE", f"/v1/objects/Vec/{uuid.UUID(int=int(i))}")


# ------------------------------------------------------------------ OpenSearch

class OpenSearchBase(ServerSystem):
    ENGINE = "opensearch"
    READY_TIMEOUT = 600
    INDEX = "idx"

    def command(self):
        self.port, transport = self.ports(2)
        home = servers.artifact_binary("opensearch").parents[1]
        conf = self.workdir / "config"
        if not conf.exists():
            shutil.copytree(home / "config", conf)
        heap = os.environ.get("HMS_BENCH_OPENSEARCH_HEAP", "4g")
        settings = {"path.data": self.data, "path.logs": self.logs, "http.port": self.port, "transport.port": transport,
                    "network.host": "127.0.0.1", "discovery.type": "single-node", "cluster.name": "hmsbench", "node.name": "n1",
                    "plugins.security.disabled": "true", "bootstrap.memory_lock": "false"}
        argv = [home / "bin" / "opensearch", *(x for k, v in settings.items() for x in ("-E", f"{k}={v}"))]
        env = {"OPENSEARCH_PATH_CONF": str(conf), "OPENSEARCH_JAVA_OPTS": f"-Xms{heap} -Xmx{heap}"}
        return argv, env, self.workdir

    def pre_start(self):
        self.workdir.mkdir(parents=True, exist_ok=True)

    def ready(self):
        return Http(self.port).probe("/_cluster/health", lambda s, d: s == 200 and json.loads(d)["status"] in ("green", "yellow"))

    def connect(self):
        self.http = Http(self.port)
        # One query, one search thread: concurrent segment search off (a dynamic cluster setting).
        self.http.call("PUT", "/_cluster/settings", {"persistent": {"search.concurrent_segment_search.mode": "none"}})

    def finish_index(self):
        """Queryable and steady: refresh, merge to one segment, refresh, restore the refresh interval."""
        i = self.INDEX
        self.http.call("POST", f"/{i}/_refresh")
        self.http.call("POST", f"/{i}/_forcemerge?max_num_segments=1")
        self.http.call("PUT", f"/{i}/_settings", {"index": {"refresh_interval": "1s"}})
        self.http.call("POST", f"/{i}/_refresh")

    def index_ready(self):
        return self.http.call("GET", f"/_cluster/health/{self.INDEX}?wait_for_status=yellow&timeout=60s")["status"] in ("green", "yellow")

    def bulk(self, lines):
        out = self.http.call("POST", "/_bulk", body=("\n".join(lines) + "\n").encode(), content_type="application/x-ndjson")
        if out.get("errors"):
            raise RuntimeError("opensearch bulk errors: " + json.dumps([x for x in out["items"] if "error" in next(iter(x.values()))][:2]))


class OpenSearchKnn(OpenSearchBase):
    BUILDS = {"faiss_m16": {"engine": "faiss", "m": 16, "ef_construction": 100}, "faiss_m32e500": {"engine": "faiss", "m": 32, "ef_construction": 500}}
    SWEEPS = {k: [{"ef": e} for e in EF_SWEEP] for k in BUILDS}

    def create(self, params):
        self.http.call("PUT", f"/{self.INDEX}", {
            "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"}},
            "mappings": {"properties": {"vec": {"type": "knn_vector", "dimension": self.dim, "method": {
                "name": "hnsw", "engine": params["engine"], "space_type": "innerproduct" if self.angular else "l2",
                "parameters": {"m": params["m"], "ef_construction": params["ef_construction"]}}}}}})

    def ingest(self, train, params):
        for lo in range(0, len(train), 1000):
            lines = []
            for i in range(lo, min(lo + 1000, len(train))):
                lines += [json.dumps({"index": {"_index": self.INDEX, "_id": str(i)}}), json.dumps({"vec": train[i].tolist()})]
            self.bulk(lines)

    def warm(self):
        self.http.call("GET", f"/_plugins/_knn/warmup/{self.INDEX}")

    def wait_queryable(self, params):
        self.finish_index()
        self.warm()

    def after_start(self, params):
        self.index_ready()
        self.warm()

    def configure(self, sp):
        self.ef = sp["ef"]

    def prepare(self, q):
        return json.dumps({"size": K, "_source": False, "query": {"knn": {"vec": {
            "vector": q.tolist(), "k": K, "method_parameters": {"ef_search": self.ef}}}}}).encode()

    def query(self, body):
        return [int(h["_id"]) for h in self.http.call("POST", f"/{self.INDEX}/_search", body=body)["hits"]["hits"]]

    def replace(self, ids, vectors):
        for i, v in zip(ids, vectors):
            self.http.call("PUT", f"/{self.INDEX}/_doc/{int(i)}?refresh=true", {"vec": v.tolist()})

    def delete(self, ids):
        for i in ids:
            self.http.call("DELETE", f"/{self.INDEX}/_doc/{int(i)}?refresh=true")


# ------------------------------------------------------------------ PostgreSQL

class Postgres(ServerSystem):
    ENGINE = "postgres"
    READY_TIMEOUT = 120

    def pkg(self):
        return servers.system_package("postgres")

    def pre_start(self):
        pg = self.pg = self.pkg()
        cluster = self.data / "pg"
        if not (cluster / "PG_VERSION").exists():
            server_proc.chown_for_server(self.data)
            server_proc.as_server_user([str(Path(pg["bin"]) / "initdb"), "-D", str(cluster), "-U", "postgres", "-A", "trust",
                                        "-E", "UTF8", "--locale=C", "--no-instructions"], check=True, capture_output=True)

    def command(self):
        (self.port,) = self.ports(1)
        pg = self.pg
        settings = {"listen_addresses": "127.0.0.1", "port": self.port, "unix_socket_directories": "",
                    "max_parallel_workers_per_gather": 0, "max_parallel_maintenance_workers": max(self.threads - 1, 0),
                    "maintenance_work_mem": os.environ.get("HMS_BENCH_PG_MAINTENANCE_WORK_MEM", "4GB"),
                    "log_min_messages": "warning"}
        # hms-server ends each commit in F_FULLFSYNC on macOS; fsync_writethrough holds Postgres to
        # the same drive-cache flush. Elsewhere the default wal_sync_method already matches msync.
        if sys.platform == "darwin":
            settings["wal_sync_method"] = "fsync_writethrough"
        if pg.get("pgvector_share"):
            settings["extension_control_path"] = f"$system:{pg['pgvector_share']}"
            settings["dynamic_library_path"] = f"$libdir:{pg['pgvector_lib']}"
        argv = [Path(pg["bin"]) / "postgres", "-D", self.data / "pg", *(x for k, v in settings.items() for x in ("-c", f"{k}={v}"))]
        return argv, {}, None

    def _conn(self):
        import psycopg
        return psycopg.connect(host="127.0.0.1", port=self.port, user="postgres", dbname="postgres", autocommit=True, connect_timeout=5)

    def ready(self):
        with self._conn() as c:
            return c.execute("SELECT 1").fetchone()[0] == 1

    def connect(self):
        self.db = self._conn()
        self.db.prepare_threshold = 0  # every repeated statement runs server-side prepared

    def close(self):
        self.db.close()

    def checkpoint(self):
        self.db.execute("CHECKPOINT")


class PgVector(Postgres):
    BUILDS = {"hnsw_m16": {"index": "hnsw", "m": 16, "ef_construction": 64},
              "hnsw_m32e500": {"index": "hnsw", "m": 32, "ef_construction": 500},
              "ivfflat": {"index": "ivfflat"}}
    SWEEPS = {"hnsw_m16": [{"ef": e} for e in EF_SWEEP if e <= 1000], "hnsw_m32e500": [{"ef": e} for e in EF_SWEEP if e <= 1000],
              "ivfflat": [{"probes": p} for p in NPROBE_SWEEP]}

    def ops_class(self):
        return "vector_ip_ops" if self.angular else "vector_l2_ops"

    def operator(self):
        return "<#>" if self.angular else "<->"

    @staticmethod
    def lists(n):
        # pgvector's guidance: rows / 1000 up to 1M rows, sqrt(rows) above.
        return max(1, n // 1000) if n <= 1_000_000 else int(n ** 0.5)

    def create(self, params):
        self.db.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.db.execute(f"CREATE TABLE t (id integer PRIMARY KEY, v vector({self.dim}))")

    def ingest(self, train, params):
        with self.db.cursor() as cur, cur.copy("COPY t (id, v) FROM STDIN") as copy:
            for i, row in enumerate(train):
                copy.write_row((i, "[" + ",".join(map(repr, row.tolist())) + "]"))
        self.n = len(train)

    def wait_queryable(self, params):
        if params["index"] == "hnsw":
            self.db.execute(f"CREATE INDEX t_v ON t USING hnsw (v {self.ops_class()}) WITH (m = {params['m']}, ef_construction = {params['ef_construction']})")
        else:
            self.db.execute(f"CREATE INDEX t_v ON t USING ivfflat (v {self.ops_class()}) WITH (lists = {self.lists(self.n)})")
        self.db.execute("VACUUM ANALYZE t")
        self.checkpoint()

    def after_start(self, params):
        # A small table can make the planner prefer a scan plus sort; the row measures the index.
        self.db.execute("SET enable_seqscan = off")

    def configure(self, sp):
        if "ef" in sp:
            self.db.execute(f"SET hnsw.ef_search = {max(sp['ef'], K)}")
        else:
            self.db.execute(f"SET ivfflat.probes = {sp['probes']}")
        self.sql = f"SELECT id FROM t ORDER BY v {self.operator()} %s::vector LIMIT {K}"

    def prepare(self, q):
        return ("[" + ",".join(map(repr, q.tolist())) + "]",)

    def query(self, args):
        return [r[0] for r in self.db.execute(self.sql, args).fetchall()]

    def replace(self, ids, vectors):
        for i, v in zip(ids, vectors):
            self.db.execute("UPDATE t SET v = %s::vector WHERE id = %s", ("[" + ",".join(map(repr, v.tolist())) + "]", int(i)))

    def delete(self, ids):
        for i in ids:
            self.db.execute("DELETE FROM t WHERE id = %s", (int(i),))


class PgTsvector(Postgres):
    BUILDS = {"gin": {}}

    def create(self, params):
        self.db.execute("CREATE TABLE docs (n integer PRIMARY KEY, id text NOT NULL UNIQUE, body text NOT NULL, "
                        "tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED)")

    def ingest(self, documents):
        with self.db.cursor() as cur, cur.copy("COPY docs (n, id, body) FROM STDIN") as copy:
            for i, d in enumerate(documents):
                copy.write_row((i, d["id"], d["text"].replace("\x00", " ")))

    def wait_queryable(self, params):
        self.db.execute("CREATE INDEX docs_tsv ON docs USING gin (tsv)")
        self.db.execute("VACUUM ANALYZE docs")
        self.checkpoint()

    def prepare(self, text):
        return " | ".join(words(text))

    def search(self, expression):
        if not expression:
            return []
        rows = self.db.execute("SELECT id, ts_rank(tsv, q) AS r FROM docs, to_tsquery('english', %s) q WHERE tsv @@ q "
                               f"ORDER BY r DESC, id LIMIT {TEXT_K}", (expression,)).fetchall()
        return [(i, float(r)) for i, r in rows]

    def replace_doc(self, doc, n):
        self.db.execute("UPDATE docs SET body = %s WHERE id = %s", (doc["text"].replace("\x00", " ") + " benchmarkrevision", doc["id"]))

    def delete_doc(self, doc, n):
        self.db.execute("DELETE FROM docs WHERE id = %s", (doc["id"],))


class ParadeDb(PgTsvector):
    """pg_search (Tantivy inside PostgreSQL): BM25 over an english-stemmed field, OR query."""

    ENGINE = "paradedb"

    def command(self):
        argv, env, cwd = super().command()
        home = servers.artifact_dir("paradedb")
        share, lib = str((home / "share").resolve()), str((home / "lib").resolve())
        out = []
        for arg in argv:
            arg = str(arg)
            if arg.startswith("extension_control_path="):
                arg = f"{arg}:{share}"
            elif arg.startswith("dynamic_library_path="):
                arg = f"{arg}:{lib}"
            out.append(arg)
        if not any(a.startswith("extension_control_path=") for a in out):
            out += ["-c", f"extension_control_path=$system:{share}", "-c", f"dynamic_library_path=$libdir:{lib}"]
        # pg_search refuses to load otherwise; resolved through dynamic_library_path.
        out += ["-c", "shared_preload_libraries=pg_search"]
        return out, env, cwd

    def create(self, params):
        self.db.execute("CREATE EXTENSION IF NOT EXISTS pg_search CASCADE")
        self.db.execute("CREATE TABLE docs (n integer PRIMARY KEY, id text NOT NULL UNIQUE, body text NOT NULL)")

    def wait_queryable(self, params):
        self.db.execute("CREATE INDEX docs_bm25 ON docs USING bm25 (n, (body::pdb.simple('stemmer=english'))) WITH (key_field = 'n')")
        self.db.execute("VACUUM ANALYZE docs")
        self.checkpoint()

    def prepare(self, text):
        return " ".join(words(text))

    def search(self, query):
        if not query:
            return []
        rows = self.db.execute("SELECT id, pdb.score(n) AS s FROM docs WHERE body ||| %s "
                               f"ORDER BY s DESC, id LIMIT {TEXT_K}", (query,)).fetchall()
        return [(i, float(s)) for i, s in rows]


# ------------------------------------------------------------------ Redis 8 query engine

class RedisVss(ServerSystem):
    ENGINE = "redis"
    BUILDS = {"hnsw_m16": {"algo": "HNSW", "M": 16, "EF_CONSTRUCTION": 200}, "hnsw_m32e500": {"algo": "HNSW", "M": 32, "EF_CONSTRUCTION": 500},
              "flat": {"algo": "FLAT"}}
    SWEEPS = {"hnsw_m16": [{"ef": e} for e in EF_SWEEP], "hnsw_m32e500": [{"ef": e} for e in EF_SWEEP], "flat": [{}]}

    def command(self):
        (self.port,) = self.ports(1)
        pkg = servers.system_package("redis")
        # WORKERS 0: queries and index updates run on the main thread (one query, one thread).
        argv = [Path(pkg["bin"]) / "redis-server", "--port", str(self.port), "--bind", "127.0.0.1", "--dir", self.data,
                "--appendonly", "yes", "--appendfsync", "always", "--save", "", "--daemonize", "no", "--loglevel", "warning",
                "--loadmodule", pkg["module"], "WORKERS", "0"]
        return argv, {}, None

    def _client(self):
        import redis
        return redis.Redis(host="127.0.0.1", port=self.port, socket_connect_timeout=5)

    def ready(self):
        c = self._client()
        try:
            return c.ping()
        finally:
            c.close()

    def connect(self):
        self.r = self._client()

    def close(self):
        self.r.close()

    def create(self, params):
        attrs = ["TYPE", "FLOAT32", "DIM", self.dim, "DISTANCE_METRIC", "IP" if self.angular else "L2"]
        if params["algo"] == "HNSW":
            attrs += ["M", params["M"], "EF_CONSTRUCTION", params["EF_CONSTRUCTION"]]
        self.r.execute_command("FT.CREATE", "idx", "ON", "HASH", "PREFIX", 1, "v:", "SCHEMA", "vec", "VECTOR", params["algo"], len(attrs), *attrs)

    def ingest(self, train, params):
        for lo in range(0, len(train), 1000):
            pipe = self.r.pipeline(transaction=False)
            for i in range(lo, min(lo + 1000, len(train))):
                pipe.hset(f"v:{i}", "vec", train[i].tobytes())
            pipe.execute()

    def _indexed(self):
        raw = self.r.execute_command("FT.INFO", "idx")
        pairs = raw.items() if isinstance(raw, dict) else zip(raw[::2], raw[1::2])
        info = {k.decode() if isinstance(k, bytes) else k: v for k, v in pairs}
        indexing = int(info.get("indexing", 0))
        pct = float(info.get("percent_indexed", b"1"))
        return indexing == 0 and pct >= 1.0

    def wait_queryable(self, params):
        wait_until(self._indexed, 7200, 0.05, "redis index fully indexed")

    def after_start(self, params):
        # Loading the AOF re-creates the index definition and re-indexes every hash.
        wait_until(self._loaded, 7200, 0.05, "redis loaded")
        self.wait_queryable(params)

    def _loaded(self):
        return self.r.info("persistence").get("loading") == 0

    def configure(self, sp):
        self.ef = sp.get("ef")
        self.q = f"*=>[KNN {K} @vec $B EF_RUNTIME $EF]" if self.ef else f"*=>[KNN {K} @vec $B]"

    def prepare(self, q):
        params = ["B", q.astype(np.float32).tobytes()] + (["EF", max(self.ef, K)] if self.ef else [])
        return ("FT.SEARCH", "idx", self.q, "PARAMS", len(params), *params, "NOCONTENT", "SORTBY", "__vec_score", "LIMIT", 0, K, "DIALECT", 2)

    def query(self, args):
        out = self.r.execute_command(*args)
        keys = [r[b"id"] for r in out[b"results"]] if isinstance(out, dict) else out[1:]  # RESP3 map or RESP2 array
        return [int(key[2:]) for key in keys]

    def replace(self, ids, vectors):
        for i, v in zip(ids, vectors):
            self.r.hset(f"v:{int(i)}", "vec", v.astype(np.float32).tobytes())

    def delete(self, ids):
        for i in ids:
            self.r.delete(f"v:{int(i)}")


# ------------------------------------------------------------------ Meilisearch

class Meilisearch(ServerSystem):
    ENGINE = "meilisearch"
    BUILDS = {"default": {}}

    def command(self):
        (self.port,) = self.ports(1)
        argv = [servers.artifact_binary("meilisearch"), "--db-path", self.data / "ms", "--http-addr", f"127.0.0.1:{self.port}",
                "--no-analytics", "--env", "development", "--max-indexing-threads", str(self.threads),
                "--dump-dir", self.data / "dumps", "--snapshot-dir", self.data / "snapshots", "--log-level", "WARN"]
        return argv, {}, self.workdir

    def ready(self):
        return Http(self.port).probe("/health", lambda s, d: s == 200 and json.loads(d).get("status") == "available")

    def connect(self):
        self.http = Http(self.port)

    def task(self, out):
        uid = out["taskUid"]

        def done():
            t = self.http.call("GET", f"/tasks/{uid}")
            if t["status"] == "failed":
                raise RuntimeError(f"meilisearch task failed: {t.get('error')}")
            return t["status"] == "succeeded"
        wait_until(done, 7200, 0.0005, "meilisearch task")

    def create(self, params):
        self.task(self.http.call("POST", "/indexes", {"uid": "docs", "primaryKey": "pk"}))
        self.task(self.http.call("PATCH", "/indexes/docs/settings", {"searchableAttributes": ["body"], "displayedAttributes": ["doc"]}))

    def ingest(self, documents):
        for lo in range(0, len(documents), 10_000):
            batch = [{"pk": i, "doc": documents[i]["id"], "body": documents[i]["text"]} for i in range(lo, min(lo + 10_000, len(documents)))]
            self.task(self.http.call("POST", "/indexes/docs/documents", batch))

    def prepare(self, text):
        return json.dumps({"q": " ".join(words(text)), "limit": TEXT_K, "attributesToRetrieve": ["doc"], "showRankingScore": True}).encode()

    def search(self, body):
        return [(h["doc"], h["_rankingScore"]) for h in self.http.call("POST", "/indexes/docs/search", body=body)["hits"]]

    def replace_doc(self, doc, n):
        self.task(self.http.call("POST", "/indexes/docs/documents", [{"pk": n, "doc": doc["id"], "body": doc["text"] + " benchmarkrevision"}]))

    def delete_doc(self, doc, n):
        self.task(self.http.call("DELETE", f"/indexes/docs/documents/{n}"))


# ------------------------------------------------------------------ Typesense

class Typesense(ServerSystem):
    ENGINE = "typesense"
    BUILDS = {"default": {}}
    KEY = "hms-bench-local"

    def command(self):
        self.port, peering = self.ports(2)
        argv = [servers.artifact_binary("typesense"), "--data-dir", self.data / "ts", "--api-key", self.KEY,
                "--api-address", "127.0.0.1", "--api-port", str(self.port), "--peering-address", "127.0.0.1",
                "--peering-port", str(peering), "--enable-cors=false", "--log-dir", self.logs]
        (self.data / "ts").mkdir(parents=True, exist_ok=True)
        return argv, {}, self.workdir

    def headers(self):
        return {"X-TYPESENSE-API-KEY": self.KEY}

    def ready(self):
        return Http(self.port, self.headers()).probe("/health", lambda s, d: s == 200 and json.loads(d).get("ok") is True)

    def connect(self):
        self.http = Http(self.port, self.headers())

    def create(self, params):
        self.http.call("POST", "/collections", {"name": "docs", "fields": [
            {"name": "doc", "type": "string", "index": False, "optional": True},
            {"name": "body", "type": "string", "stem": True}]})

    def ingest(self, documents):
        for lo in range(0, len(documents), 10_000):
            lines = "\n".join(json.dumps({"id": str(i), "doc": documents[i]["id"], "body": documents[i]["text"]})
                              for i in range(lo, min(lo + 10_000, len(documents))))
            _, out = self.http.raw("POST", "/collections/docs/documents/import?action=create&batch_size=1000", lines.encode(), "text/plain")
            bad = [line for line in out.decode().splitlines() if not json.loads(line).get("success")]
            if bad:
                raise RuntimeError(f"typesense import errors: {bad[:2]}")
        (self.workdir / "documents.txt").write_text(str(len(documents)))

    def after_start(self, params):
        # /health turns ok before the raft log replay has re-indexed every document (searches
        # in that window miss stored documents), so the restarted server is ready when the
        # collection holds the built document count again.
        expected = int((self.workdir / "documents.txt").read_text())

        def loaded():
            try:
                return self.http.call("GET", "/collections/docs")["num_documents"] >= expected
            except HttpError:
                return False
        wait_until(loaded, 7200, 0.02, "typesense collection reloaded")

    def prepare(self, text):
        q = " ".join(words(text))
        # drop_tokens_threshold = k: Typesense matches all words first and drops words until k
        # hits exist, its closest setting to an OR query; ranking stays its own (text match).
        return "/collections/docs/documents/search?" + urllib.parse.urlencode(
            {"q": q, "query_by": "body", "per_page": TEXT_K, "include_fields": "doc", "drop_tokens_threshold": TEXT_K}) if q else None

    def search(self, path):
        if path is None:
            return []
        return [(h["document"]["doc"], h["text_match"]) for h in self.http.call("GET", path)["hits"]]

    def replace_doc(self, doc, n):
        self.http.call("POST", "/collections/docs/documents?action=upsert", {"id": str(n), "doc": doc["id"], "body": doc["text"] + " benchmarkrevision"})

    def delete_doc(self, doc, n):
        self.http.call("DELETE", f"/collections/docs/documents/{n}")


# ------------------------------------------------------------------ OpenSearch BM25

class OpenSearchBm25(OpenSearchBase):
    BUILDS = {"default": {}}

    def create(self, params):
        self.http.call("PUT", f"/{self.INDEX}", {
            "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"}},
            "mappings": {"properties": {"body": {"type": "text", "analyzer": "english"}}}})

    def ingest(self, documents):
        for lo in range(0, len(documents), 5000):
            lines = []
            for d in documents[lo:lo + 5000]:
                lines += [json.dumps({"index": {"_index": self.INDEX, "_id": d["id"]}}), json.dumps({"body": d["text"]})]
            self.bulk(lines)

    def wait_queryable(self, params):
        self.finish_index()

    def after_start(self, params):
        self.index_ready()

    def prepare(self, text):
        q = " ".join(words(text))
        return json.dumps({"size": TEXT_K, "_source": False, "query": {"match": {"body": q}}}).encode() if q else None

    def search(self, body):
        if body is None:
            return []
        return [(h["_id"], h["_score"]) for h in self.http.call("POST", f"/{self.INDEX}/_search", body=body)["hits"]["hits"]]

    def replace_doc(self, doc, n):
        self.http.call("PUT", f"/{self.INDEX}/_doc/{urllib.parse.quote(doc['id'], safe='')}?refresh=true", {"body": doc["text"] + " benchmarkrevision"})

    def delete_doc(self, doc, n):
        self.http.call("DELETE", f"/{self.INDEX}/_doc/{urllib.parse.quote(doc['id'], safe='')}?refresh=true")


ANN = {"hms-server": HmsServer, "qdrant-server": QdrantServer, "weaviate": WeaviateServer, "opensearch-knn": OpenSearchKnn,
       "pgvector": PgVector, "redis-vss": RedisVss}
TEXT = {"hms-server": HmsServerText, "meilisearch": Meilisearch, "typesense": Typesense, "opensearch-bm25": OpenSearchBm25, "pg-tsvector": PgTsvector,
        "paradedb": ParadeDb}


# ------------------------------------------------------------------ phases

def ann_build(a, sysm, meta):
    train = normalize(load_rows(Path(a.data) / "train.f32", meta["dim"], meta.get("n_train")), sysm.angular)
    params = sysm.BUILDS[a.build]
    server = sysm.start()
    try:
        start = time.perf_counter()
        sysm.create(params)
        sysm.ingest(train, params)
        sysm.wait_queryable(params)
        build_s = time.perf_counter() - start
        rss_after = server.sample()
        extra = {"segments": sysm.segments} if hasattr(sysm, "segments") else {}
    finally:
        sysm.stop()
    return {"build_s": build_s, "peak_rss_bytes": server.peak, "rss_after_build": rss_after, "n": len(train), **extra}


def ann_serve(a, sysm, meta):
    test = normalize(load_rows(Path(a.data) / "test.f32", meta["dim"], a.queries), sysm.angular)
    truth = np.fromfile(Path(a.data) / "neighbors.i32", dtype="<i4").reshape(-1, meta["n_neighbors"])[: len(test), :K]
    params = sysm.BUILDS[a.build]
    server = sysm.start()
    try:
        sysm.after_start(params)
        open_s = (time.time_ns() - server.started_ns) / 1e9
        sysm.configure(sysm.SWEEPS[a.build][0])
        sysm.query(sysm.prepare(test[0]))
        cold_start_s = (time.time_ns() - server.started_ns) / 1e9
        server.sampling(False)
        for q in test[:50]:
            sysm.query(sysm.prepare(q))
        rss_before = server.sample()
        configs = []
        for sp in sysm.SWEEPS[a.build]:
            sysm.configure(sp)
            bodies = [sysm.prepare(q) for q in test]
            found = [sysm.query(b) for b in bodies]
            hits = sum(len(set(int(x) for x in f[:K]) & set(int(x) for x in t)) for f, t in zip(found, truth))
            recall = hits / (K * len(test))
            repeats = []
            for _ in range(a.repeats if recall >= 0.5 else 1):
                lat = np.empty(len(test), dtype=np.int64)
                for i, b in enumerate(bodies):
                    t = time.perf_counter_ns()
                    sysm.query(b)
                    lat[i] = time.perf_counter_ns() - t
                order = np.sort(lat) / 1e3
                repeats.append({"qps": len(test) / (lat.sum() / 1e9), "p50_us": float(pctl(order, 0.5)), "p99_us": float(pctl(order, 0.99))})
            server.sample()
            configs.append({"params": sp, "recall_at_10": recall, "repeats": repeats, "load_1m": os.getloadavg()[0], "load_gate_met": None})
            if recall >= 0.999:
                break
        steady = server.sample()
    finally:
        sysm.stop()
    return {"open_s": open_s, "cold_start_s": cold_start_s, "rss_before_sweep": rss_before, "rss_steady": steady,
            "server_peak_rss_bytes": server.peak, "n_queries": len(test), "n": meta["n_train"], "configs": configs}


def ann_ops(a, sysm, meta):
    count = 100
    n = meta["n_train"]
    rng = np.random.default_rng(7)
    ids = rng.choice(n, size=2 * count, replace=False)
    fresh = normalize(rng.standard_normal((count, meta["dim"])).astype(np.float32), sysm.angular)
    params = sysm.BUILDS[a.build]
    sysm.start()
    out = {"count": count}
    try:
        sysm.after_start(params)
        for name, fn in (("replace", lambda: sysm.replace(ids[:count], fresh)), ("delete", lambda: sysm.delete(ids[count:]))):
            try:
                t = time.perf_counter()
                fn()
                elapsed = time.perf_counter() - t
                out[name] = {"supported": True, "apply_s": elapsed, "flush_s": 0.0, "per_op_ms_durable": elapsed / count * 1e3, "flush_required": False}
            except Unsupported as e:
                out[name] = {"supported": False, "reason": str(e)}
    finally:
        sysm.stop()
    return out


def text_build(a, sysm, data):
    server = sysm.start()
    try:
        start = time.perf_counter()
        sysm.create({})
        sysm.ingest(data["documents"])
        sysm.wait_queryable({})
        ingest_s = time.perf_counter() - start
    finally:
        sysm.stop()
    return {"documents": len(data["documents"]), "ingest_s": ingest_s, "peak_rss_bytes": server.peak}


def text_serve(a, sysm, data):
    queries = data["queries"]
    server = sysm.start()
    try:
        sysm.after_start({})
        open_s = (time.time_ns() - server.started_ns) / 1e9
        prepared = [sysm.prepare(q["text"]) for q in queries]
        sysm.search(prepared[0])
        cold_start_s = (time.time_ns() - server.started_ns) / 1e9
        server.sampling(False)
        for p in prepared[:16]:
            sysm.search(p)
        rows, passes = [], []
        for n in range(a.passes):
            ns = []
            for q, p in zip(queries, prepared):
                t = time.perf_counter_ns()
                found = sysm.search(p)
                ns.append(time.perf_counter_ns() - t)
                if n == 0:
                    rows.append({"id": q["id"], "results": [{"documentId": i, "lexicalScore": s} for i, s in found]})
            passes.append(ns)
            server.sample()
        steady = server.sample()
    finally:
        sysm.stop()
    return {"open_s": open_s, "cold_start_s": cold_start_s, "rows": rows, "passes_ns": passes, "rss_steady": steady,
            "peak_rss_bytes": server.peak}


def text_ops(a, sysm, data):
    docs = data["documents"]
    sysm.start()
    out = {"count": a.ops}
    try:
        sysm.after_start({})
        for name, lo in (("replace", 0), ("delete", a.ops)):
            fn = getattr(sysm, name + "_doc")
            t = time.perf_counter()
            for n in range(lo, lo + a.ops):
                fn(docs[n], n)
            out[name] = {"supported": True, "per_op_ms_durable": (time.perf_counter() - t) / a.ops * 1e3}
    finally:
        sysm.stop()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["ann", "text"])
    p.add_argument("mode", choices=["build", "serve", "ops"])
    p.add_argument("--system", required=True)
    p.add_argument("--build", default=None)
    p.add_argument("--data", help="ANN dataset directory")
    p.add_argument("--input", help="text input JSON")
    p.add_argument("--workdir", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--queries", type=int, default=None)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--ops", type=int, default=100)
    a = p.parse_args()
    workdir = Path(a.workdir)
    if a.mode == "build":
        shutil.rmtree(workdir, ignore_errors=True)
        workdir.mkdir(parents=True)
    if a.kind == "ann":
        meta = json.loads((Path(a.data) / "meta.json").read_text())
        sysm = ANN[a.system](workdir, a.threads, a.mode, meta["dim"], meta["metric"] == "angular")
        body = {"build": ann_build, "serve": ann_serve, "ops": ann_ops}[a.mode](a, sysm, meta)
    else:
        data = json.loads(Path(a.input).read_text())
        sysm = TEXT[a.system](workdir, a.threads, a.mode)
        body = {"build": text_build, "serve": text_serve, "ops": text_ops}[a.mode](a, sysm, data)
    body["client_peak_rss_bytes"] = peak_rss()
    body["server_log"] = str(sysm.server.log_path) if sysm.server else None
    print("RESULT " + json.dumps(body, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except socket.timeout as error:
        raise SystemExit(f"timeout: {error}")
