# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Runs inside one comparator's pinned environment (never imported by the orchestrator).

Modes, each a fresh process so peak RSS, open time and cold start are honest:
  build  build the index from train.f32 and persist it under --workdir
  serve  open the persisted index, run the recall/QPS/latency sweep one query at a time
  ops    time 100 replacements and 100 deletions on the persisted index

Data layout: <data>/{train.f32,test.f32,neighbors.i32,meta.json}, rows L2-normalised here
(angular metric), exactly as benchmarks/public/evaluate.py does for the other systems.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

T_IMPORT = time.time_ns()

import numpy as np  # noqa: E402

K = 10
EF_SWEEP = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
NPROBE_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256]


class Unsupported(Exception):
    """The library offers no such operation; the row records the reason."""


def rss_now():
    try:
        if sys.platform == "linux":
            return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        import subprocess
        return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout) * 1024
    except (OSError, ValueError):
        return 0


def peak_rss():
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def normalize(a, angular=True):
    if not angular:
        return np.ascontiguousarray(a.astype(np.float32))
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return np.ascontiguousarray((a / np.where(n == 0, 1, n)).astype(np.float32))


def sync_index(directory):
    for path in directory.rglob("*"):
        if path.is_file():
            with path.open("rb") as f:
                if sys.platform == "darwin":
                    import fcntl
                    fcntl.fcntl(f.fileno(), fcntl.F_FULLFSYNC)
                else:
                    os.fsync(f.fileno())
    if os.name == "posix":
        for path in [*sorted((p for p in directory.rglob("*") if p.is_dir()), reverse=True), directory]:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)


def load_rows(path, dim, count=None):
    rows = np.fromfile(path, dtype="<f4").reshape(-1, dim)
    return rows if count is None else rows[:count]


class Adapter:
    """Subclasses define BUILDS {id: params} and SWEEPS {id: [search params]}."""
    BUILDS = {}
    SWEEPS = {}
    VARIANTS = {}  # build id -> settings that only take effect in a fresh serving process
    PERSISTENT_FLUSH = False  # True when ops need an explicit save to be durable

    def pre_open(self, variant):
        pass

    def __init__(self, dim, workdir, threads, metric="angular"):
        self.dim, self.workdir, self.threads = dim, Path(workdir), threads
        self.angular = metric == "angular"  # angular data is L2-normalised, so inner product is cosine

    def build(self, train, params):
        raise NotImplementedError

    def open(self, params, n):
        raise NotImplementedError

    def configure(self, sp):
        pass

    def query(self, q):
        raise NotImplementedError

    def replace(self, ids, vectors):
        raise Unsupported("no in-place replacement")

    def delete(self, ids):
        raise Unsupported("no deletion")

    def flush(self):
        pass

    def version_note(self):
        return {}


class Warm(Adapter):
    BUILDS = {"persisted": {}}
    SWEEPS = {"persisted": [{"ef": 64}]}
    PERSISTENT_FLUSH = True

    def open(self, params, n):
        import subprocess
        binary = os.environ["WARM_PUBLIC_BENCH"]
        self.process = subprocess.Popen([binary, "ann-updates", "--action", "session", "--index", str(self.workdir / "index.updates")],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        if json.loads(self.process.stdout.readline()) != {"ready": True}:
            raise RuntimeError("WARM update session failed to open")

    def _request(self, request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        response = json.loads(self.process.stdout.readline())
        if "error" in response:
            raise RuntimeError(response["error"])

    def replace(self, ids, vectors):
        self._request({"op": "replace", "ids": [int(i) for i in ids], "vectors": vectors.tolist()})

    def delete(self, ids):
        self._request({"op": "delete", "ids": [int(i) for i in ids]})

    def flush(self):
        self._request({"op": "commit"})

    def close(self):
        self.process.stdin.close()
        if self.process.wait(timeout=30) != 0:
            raise RuntimeError("WARM update session exited unsuccessfully")


class Hnswlib(Adapter):
    # m32e500 and m48e500 match the build effort of the WARM graph (degree 64 at level 0 for
    # M=32, construction ef 500); analysis reports the best build per target recall.
    BUILDS = {"m16": {"M": 16, "ef_construction": 200}, "m32": {"M": 32, "ef_construction": 200},
              "m32e500": {"M": 32, "ef_construction": 500}, "m48e500": {"M": 48, "ef_construction": 500}}
    SWEEPS = {k: [{"ef": e} for e in EF_SWEEP] for k in BUILDS}
    PERSISTENT_FLUSH = True

    def build(self, train, params):
        import hnswlib
        idx = hnswlib.Index(space="ip" if self.angular else "l2", dim=self.dim)
        idx.init_index(max_elements=len(train), M=params["M"], ef_construction=params["ef_construction"], random_seed=100)
        idx.set_num_threads(self.threads)
        idx.add_items(train, np.arange(len(train)))
        idx.save_index(str(self.workdir / "index.bin"))

    def open(self, params, n):
        import hnswlib
        self.idx = hnswlib.Index(space="ip" if self.angular else "l2", dim=self.dim)
        self.idx.load_index(str(self.workdir / "index.bin"), max_elements=n + 1000)
        self.idx.set_num_threads(1)

    def configure(self, sp):
        self.idx.set_ef(max(sp["ef"], K))

    def query(self, q):
        return self.idx.knn_query(q, k=K)[0][0]

    def replace(self, ids, vectors):
        self.idx.add_items(vectors, np.asarray(ids))

    def delete(self, ids):
        for i in ids:
            self.idx.mark_deleted(int(i))

    def flush(self):
        self.idx.save_index(str(self.workdir / "index.bin"))
        sync_index(self.workdir)


class FaissBase(Adapter):
    PERSISTENT_FLUSH = True

    def _write(self, index):
        import faiss
        faiss.write_index(index, str(self.workdir / "index.faiss"))

    def open(self, params, n):
        import faiss
        faiss.omp_set_num_threads(1)
        self.index = faiss.read_index(str(self.workdir / "index.faiss"))

    def query(self, q):
        return self.index.search(q, K)[1][0]

    def flush(self):
        self._write(self.index)
        sync_index(self.workdir)


class FaissHnsw(FaissBase):
    BUILDS = {"m16": {"M": 16, "efConstruction": 200}, "m32": {"M": 32, "efConstruction": 200},
              "m32e500": {"M": 32, "efConstruction": 500}, "m48e500": {"M": 48, "efConstruction": 500}}
    SWEEPS = {k: [{"ef": e} for e in EF_SWEEP] for k in BUILDS}

    def build(self, train, params):
        import faiss
        faiss.omp_set_num_threads(self.threads)
        index = faiss.IndexHNSWFlat(self.dim, params["M"], faiss.METRIC_INNER_PRODUCT if self.angular else faiss.METRIC_L2)
        index.hnsw.efConstruction = params["efConstruction"]
        index.add(train)
        self._write(index)

    def configure(self, sp):
        self.index.hnsw.efSearch = max(sp["ef"], K)

    def replace(self, ids, vectors):
        raise Unsupported("IndexHNSWFlat cannot remove or replace vectors")

    def delete(self, ids):
        raise Unsupported("IndexHNSWFlat cannot remove vectors")


class FaissFlat(FaissBase):
    BUILDS = {"flat": {}}
    SWEEPS = {"flat": [{}]}

    def build(self, train, params):
        import faiss
        flat = faiss.IndexFlatIP(self.dim) if self.angular else faiss.IndexFlatL2(self.dim)
        index = faiss.IndexIDMap2(flat)
        index.add_with_ids(train, np.arange(len(train), dtype=np.int64))
        self._write(index)

    def replace(self, ids, vectors):
        self.index.remove_ids(np.asarray(ids, dtype=np.int64))
        self.index.add_with_ids(vectors, np.asarray(ids, dtype=np.int64))

    def delete(self, ids):
        self.index.remove_ids(np.asarray(ids, dtype=np.int64))


class FaissIvfpq(FaissBase):
    BUILDS = {"ivfpq": {"refine": False}, "ivfpq_refine": {"refine": True}}
    SWEEPS = {"ivfpq": [{"nprobe": p} for p in NPROBE_SWEEP],
              "ivfpq_refine": [{"nprobe": p, "k_factor": f} for f in (4, 16) for p in NPROBE_SWEEP]}

    def nlist(self, n):
        return int(2 ** round(np.log2(4 * np.sqrt(n))))

    def build(self, train, params):
        import faiss
        faiss.omp_set_num_threads(self.threads)
        n = len(train)
        flat = (lambda: faiss.IndexFlatIP(self.dim)) if self.angular else (lambda: faiss.IndexFlatL2(self.dim))
        quantizer = flat()
        pq_m = next(m for m in (25, 20, 16, 8) if self.dim % m == 0)
        ivf = faiss.IndexIVFPQ(quantizer, self.dim, self.nlist(n), pq_m, 8, faiss.METRIC_INNER_PRODUCT if self.angular else faiss.METRIC_L2)
        sample = train[np.random.default_rng(0).choice(n, size=min(n, 64 * self.nlist(n)), replace=False)]
        ivf.train(sample)
        index = faiss.IndexRefine(ivf, flat()) if params["refine"] else ivf
        index.add(train)
        self._write(index)

    def open(self, params, n):
        super().open(params, n)
        self.is_refine = params["refine"]

    def configure(self, sp):
        import faiss
        base = faiss.downcast_index(self.index.base_index) if self.is_refine else self.index
        base.nprobe = sp["nprobe"]
        if self.is_refine:
            self.index.k_factor = sp["k_factor"]

    def delete(self, ids):
        if self.is_refine:
            raise Unsupported("IndexRefine remove_ids is not defined for the refine store")
        self.index.remove_ids(np.asarray(ids, dtype=np.int64))

    def replace(self, ids, vectors):
        if self.is_refine:
            raise Unsupported("IndexRefineFlat remove_ids is not defined for the refine store")
        self.index.remove_ids(np.asarray(ids, dtype=np.int64))
        self.index.add_with_ids(vectors, np.asarray(ids, dtype=np.int64))


class Usearch(Adapter):
    BUILDS = {"m16": {"connectivity": 16, "expansion_add": 200, "dtype": "f32"},
              "m32": {"connectivity": 32, "expansion_add": 200, "dtype": "f32"},
              "m16_i8": {"connectivity": 16, "expansion_add": 200, "dtype": "i8"}}
    SWEEPS = {k: [{"ef": e} for e in EF_SWEEP] for k in BUILDS}
    PERSISTENT_FLUSH = True

    def _make(self, params):
        from usearch.index import Index
        return Index(ndim=self.dim, metric="cos" if self.angular else "l2sq", dtype=params["dtype"], connectivity=params["connectivity"],
                     expansion_add=params["expansion_add"], expansion_search=64)

    def build(self, train, params):
        index = self._make(params)
        index.add(np.arange(len(train)), train, threads=self.threads)
        index.save(str(self.workdir / "index.usearch"))

    def open(self, params, n):
        index = self._make(params)
        index.load(str(self.workdir / "index.usearch"))
        self.index = index

    def configure(self, sp):
        self.index.expansion_search = max(sp["ef"], K)

    def query(self, q):
        return self.index.search(q[0], K).keys

    def replace(self, ids, vectors):
        self.index.remove(np.asarray(ids))
        self.index.add(np.asarray(ids), vectors, threads=1)

    def delete(self, ids):
        self.index.remove(np.asarray(ids))

    def flush(self):
        self.index.save(str(self.workdir / "index.usearch"))
        sync_index(self.workdir)


class Annoy(Adapter):
    BUILDS = {"t100": {"n_trees": 100}}
    SWEEPS = {"t100": [{"search_k": s} for s in (100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200)]}

    def build(self, train, params):
        from annoy import AnnoyIndex
        index = AnnoyIndex(self.dim, "angular" if self.angular else "euclidean")
        index.on_disk_build(str(self.workdir / "index.ann"))
        for i, row in enumerate(train):
            index.add_item(i, row)
        index.build(params["n_trees"], n_jobs=self.threads)

    def open(self, params, n):
        from annoy import AnnoyIndex
        self.index = AnnoyIndex(self.dim, "angular" if self.angular else "euclidean")
        self.index.load(str(self.workdir / "index.ann"))

    def configure(self, sp):
        self.sk = sp["search_k"]

    def query(self, q):
        return self.index.get_nns_by_vector(q[0], K, search_k=self.sk)


class SqliteVec(Adapter):
    BUILDS = {"exact": {}}
    SWEEPS = {"exact": [{}]}

    def _connect(self):
        import sqlite3
        import sqlite_vec
        db = sqlite3.connect(str(self.workdir / "vec.db"))
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        db.execute("PRAGMA fullfsync = ON")
        db.execute("PRAGMA checkpoint_fullfsync = ON")
        return db

    def build(self, train, params):
        db = self._connect()
        db.execute(f"CREATE VIRTUAL TABLE v USING vec0(embedding float[{self.dim}] distance_metric={'cosine' if self.angular else 'l2'})")
        with db:
            db.executemany("INSERT INTO v(rowid, embedding) VALUES (?, ?)", ((i, row.tobytes()) for i, row in enumerate(train)))
        db.close()

    def open(self, params, n):
        self.db = self._connect()

    def query(self, q):
        rows = self.db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k = ?", (q[0].tobytes(), K)).fetchall()
        return [r[0] for r in rows]

    def replace(self, ids, vectors):
        with self.db:
            for i, row in zip(ids, vectors):
                self.db.execute("DELETE FROM v WHERE rowid = ?", (int(i),))
                self.db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (int(i), row.tobytes()))

    def delete(self, ids):
        with self.db:
            for i in ids:
                self.db.execute("DELETE FROM v WHERE rowid = ?", (int(i),))


class Lancedb(Adapter):
    BUILDS = {"ivf_pq": {"index_type": "IVF_PQ"}, "ivf_hnsw_sq": {"index_type": "IVF_HNSW_SQ"}}
    SWEEPS = {"ivf_pq": [{"nprobes": p, "refine_factor": r} for r in (None, 10) for p in NPROBE_SWEEP],
              "ivf_hnsw_sq": [{"nprobes": p, "ef": e} for p in (8, 32) for e in (32, 128, 512)]}

    def build(self, train, params):
        import lancedb
        import pyarrow as pa
        db = lancedb.connect(str(self.workdir / "db"))
        vectors = pa.FixedSizeListArray.from_arrays(pa.array(train.reshape(-1), type=pa.float32()), self.dim)
        table = db.create_table("t", pa.table({"id": pa.array(np.arange(len(train)), type=pa.int64()), "vector": vectors}))
        parts = max(1, int(2 ** round(np.log2(np.sqrt(len(train))))))
        kwargs = {"metric": "cosine" if self.angular else "l2", "vector_column_name": "vector", "index_type": params["index_type"], "num_partitions": parts}
        if params["index_type"] != "IVF_HNSW_SQ":
            kwargs["num_sub_vectors"] = 25 if self.dim % 25 == 0 else 16
        table.create_index(**kwargs)

    def open(self, params, n):
        import lancedb
        self.table = lancedb.connect(str(self.workdir / "db")).open_table("t")

    def configure(self, sp):
        self.sp = sp

    def query(self, q):
        s = self.table.search(q[0], vector_column_name="vector").metric("cosine" if self.angular else "l2").nprobes(self.sp["nprobes"]).limit(K)
        if self.sp.get("refine_factor"):
            s = s.refine_factor(self.sp["refine_factor"])
        if self.sp.get("ef"):
            s = s.ef(self.sp["ef"])
        return s.select(["id"]).to_arrow()["id"].to_numpy()

    def replace(self, ids, vectors):
        import pyarrow as pa
        ids = [int(i) for i in ids]
        self.table.delete(f"id IN ({','.join(map(str, ids))})")
        vec = pa.FixedSizeListArray.from_arrays(pa.array(vectors.reshape(-1), type=pa.float32()), self.dim)
        self.table.add(pa.table({"id": pa.array(ids, type=pa.int64()), "vector": vec}))

    def delete(self, ids):
        self.table.delete(f"id IN ({','.join(str(int(i)) for i in ids)})")


class Chroma(Adapter):
    BUILDS = {"m16": {"max_neighbors": 16, "ef_construction": 200}, "m32": {"max_neighbors": 32, "ef_construction": 200}}
    # ef_search is read when the segment loads, so each value needs its own serving process
    # (verified: modify() on a loaded collection leaves recall unchanged).
    SWEEPS = {k: [{}] for k in BUILDS}
    VARIANTS = {k: [{"ef": e} for e in (16, 32, 64, 128, 256, 512)] for k in BUILDS}

    def pre_open(self, variant):
        self._client().get_collection("vectors").modify(configuration={"hnsw": {"ef_search": variant["ef"]}})

    def _client(self):
        import chromadb
        return chromadb.PersistentClient(path=str(self.workdir / "db"))

    def build(self, train, params):
        client = self._client()
        col = client.create_collection("vectors", configuration={"hnsw": {"space": "cosine" if self.angular else "l2", "ef_construction": params["ef_construction"],
                                                                     "max_neighbors": params["max_neighbors"], "ef_search": 100}})
        batch = client.get_max_batch_size()
        for lo in range(0, len(train), batch):
            hi = min(lo + batch, len(train))
            col.add(ids=[str(i) for i in range(lo, hi)], embeddings=train[lo:hi])

    def open(self, params, n):
        self.col = self._client().get_collection("vectors")

    def query(self, q):
        return [int(i) for i in self.col.query(query_embeddings=q, n_results=K)["ids"][0]]

    def replace(self, ids, vectors):
        self.col.upsert(ids=[str(int(i)) for i in ids], embeddings=vectors)

    def delete(self, ids):
        self.col.delete(ids=[str(int(i)) for i in ids])


class Qdrant(Adapter):
    BUILDS = {"local_exact": {}}
    SWEEPS = {"local_exact": [{}]}

    def build(self, train, params):
        from qdrant_client import QdrantClient, models
        client = QdrantClient(path=str(self.workdir / "db"))
        client.create_collection("c", vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE if self.angular else models.Distance.EUCLID))
        client.upload_collection("c", vectors=train, ids=list(range(len(train))), batch_size=1024)
        client.close()

    def open(self, params, n):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(path=str(self.workdir / "db"))

    def query(self, q):
        return [p.id for p in self.client.query_points("c", query=q[0].tolist(), limit=K).points]

    def replace(self, ids, vectors):
        from qdrant_client import models
        self.client.upsert("c", points=models.Batch(ids=[int(i) for i in ids], vectors=vectors.tolist()))

    def delete(self, ids):
        from qdrant_client import models
        self.client.delete("c", points_selector=models.PointIdsList(points=[int(i) for i in ids]))


class Milvus(Adapter):
    BUILDS = {"hnsw_m16": {"index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}},
              "ivf_flat": {"index_type": "IVF_FLAT", "params": {"nlist": 512}}}
    SWEEPS = {"hnsw_m16": [{"ef": e} for e in EF_SWEEP[:11]], "ivf_flat": [{"nprobe": p} for p in NPROBE_SWEEP]}
    PERSISTENT_FLUSH = True

    def _client(self):
        from pymilvus import MilvusClient
        return MilvusClient(str(self.workdir / "milvus.db"))

    def build(self, train, params):
        client = self._client()
        index = client.prepare_index_params()
        index.add_index(field_name="vector", index_type=params["index_type"], metric_type="COSINE" if self.angular else "L2", params=params["params"])
        client.create_collection("c", dimension=self.dim, metric_type="COSINE" if self.angular else "L2", index_params=index, auto_id=False, primary_field_name="id", vector_field_name="vector")
        for lo in range(0, len(train), 5000):
            hi = min(lo + 5000, len(train))
            client.insert("c", [{"id": i, "vector": train[i]} for i in range(lo, hi)])
        client.flush("c") if hasattr(client, "flush") else None
        client.close()

    def open(self, params, n):
        self.client = self._client()
        self.client.load_collection("c")

    def configure(self, sp):
        self.sp = {"params": dict(sp)}

    def query(self, q):
        res = self.client.search("c", data=[q[0].tolist()], limit=K, search_params=self.sp, output_fields=[])
        return [h["id"] for h in res[0]]

    def replace(self, ids, vectors):
        self.client.upsert("c", [{"id": int(i), "vector": v} for i, v in zip(ids, vectors)])

    def delete(self, ids):
        self.client.delete("c", ids=[int(i) for i in ids])

    def flush(self):
        self.client.flush("c")


class Duckdb(Adapter):
    BUILDS = {"hnsw_m16": {"M": 16, "ef_construction": 200}}
    SWEEPS = {"hnsw_m16": [{"ef": e} for e in EF_SWEEP[:11]]}

    def _connect(self):
        import duckdb
        con = duckdb.connect(str(self.workdir / "db.duckdb"))
        con.execute("INSTALL vss; LOAD vss; SET hnsw_enable_experimental_persistence = true;")
        return con

    def build(self, train, params):
        import pyarrow as pa
        con = self._connect()
        try:
            con.execute(f"CREATE TABLE t(id INTEGER, v FLOAT[{self.dim}])")
            for lo in range(0, len(train), 20000):
                hi = min(lo + 20000, len(train))
                vec = pa.FixedSizeListArray.from_arrays(pa.array(train[lo:hi].reshape(-1), type=pa.float32()), self.dim)
                batch = pa.table({"id": pa.array(np.arange(lo, hi), type=pa.int32()), "v": vec})
                con.register("warm_ingest_batch", batch)
                try:
                    con.execute("INSERT INTO t SELECT * FROM warm_ingest_batch")
                finally:
                    con.unregister("warm_ingest_batch")
            con.execute(f"CREATE INDEX idx ON t USING HNSW (v) WITH (metric='{'cosine' if self.angular else 'l2sq'}', M={params['M']}, ef_construction={params['ef_construction']})")
        finally:
            con.close()

    def open(self, params, n):
        self.con = self._connect()

    def configure(self, sp):
        self.con.execute(f"SET hnsw_ef_search = {max(sp['ef'], K)}")

    def query(self, q):
        fn = "array_cosine_distance" if self.angular else "array_distance"
        rows = self.con.execute(f"SELECT id FROM t ORDER BY {fn}(v, ?::FLOAT[{self.dim}]) LIMIT {K}", [q[0].tolist()]).fetchall()
        return [r[0] for r in rows]

    def replace(self, ids, vectors):
        for i, row in zip(ids, vectors):
            self.con.execute(f"UPDATE t SET v = ?::FLOAT[{self.dim}] WHERE id = ?", [row.tolist(), int(i)])

    def delete(self, ids):
        for i in ids:
            self.con.execute("DELETE FROM t WHERE id = ?", [int(i)])

    def flush(self):
        self.con.execute("CHECKPOINT")


SYSTEMS = {
    "warm": Warm,
    "hnswlib": Hnswlib, "faiss-hnsw": FaissHnsw, "faiss-flat": FaissFlat, "faiss-ivfpq": FaissIvfpq,
    "usearch": Usearch, "annoy": Annoy, "sqlite-vec": SqliteVec, "lancedb": Lancedb, "chroma": Chroma,
    "qdrant-local": Qdrant, "milvus-lite": Milvus, "duckdb-vss": Duckdb,
}


class Ngt(Adapter):
    """NGT through ngtpy: ANNG built with batch_insert, then the ONNG optimiser (out/in degree
    pruning). The Python binding exposes no QG builder (QuantizedIndex only opens a `qg` directory
    written by the NGT command-line tool), so ONNG is the strongest graph the binding can build."""
    # onng_default: creation edges 10, outgoing 10, incoming 120 (the NGT README ONNG example).
    # onng_d64: creation edges 64 and 16+48 optimised edges, matching the degree-64 build effort of WARM.
    BUILDS = {"onng_default": {"creation": 10, "out": 10, "inn": 120}, "onng_d64": {"creation": 64, "out": 16, "inn": 48}}
    SWEEPS = {k: [{"epsilon": e} for e in (0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)] for k in BUILDS}
    PERSISTENT_FLUSH = True

    def build(self, train, params):
        import ngtpy
        anng, onng = self.workdir / "anng", self.workdir / "onng"
        ngtpy.create(str(anng), self.dim, edge_size_for_creation=params["creation"], edge_size_for_search=40,
                     distance_type="Cosine" if self.angular else "L2", object_type="Float")
        index = ngtpy.Index(str(anng))
        index.batch_insert(train, num_threads=self.threads)
        index.save()
        index.close()
        ngtpy.Optimizer(num_of_outgoings=params["out"], num_of_incomings=params["inn"]).execute(str(anng), str(onng))
        shutil.rmtree(anng)

    def open(self, params, n):
        import ngtpy
        self.index = ngtpy.Index(str(self.workdir / "onng"))
        path = self.workdir / "ids.json"
        self.object_ids = {int(k): v for k, v in json.loads(path.read_text()).items()} if path.exists() else {}
        self.dataset_ids = {v: k for k, v in self.object_ids.items()}

    def configure(self, sp):
        self.eps = sp["epsilon"]

    def query(self, q):
        found = self.index.search(q[0], K, epsilon=self.eps, with_distance=False)
        return [self.dataset_ids.get(i, i) for i in found] if self.dataset_ids else found

    def replace(self, ids, vectors):
        self.delete(ids)
        for i, v in zip(ids, vectors):
            # batch_insert returns None and appends new IDs; insert returns the allocated ID.
            oid = self.index.insert(v)
            self.object_ids[int(i)] = oid
            self.dataset_ids[oid] = int(i)
        self.index.build_index(num_threads=1)

    def delete(self, ids):
        for i in ids:
            oid = self.object_ids.pop(int(i), int(i))
            self.index.remove(oid)
            self.dataset_ids.pop(oid, None)

    def flush(self):
        self.index.save()
        (self.workdir / "ids.json").write_text(json.dumps(self.object_ids))
        sync_index(self.workdir)


class Scann(Adapter):
    """ScaNN: k-means tree (spherical for angular data) + anisotropic 4-bit-LUT scoring + exact reorder; the
    ann-benchmarks glove-100-angular scann1 and scann4 build arguments, with the leaf count scaled
    by sqrt(n) from its 2000 at 1.18M rows so subsets keep the same rows per leaf."""
    BUILDS = {"ab_scann1": {"leaf_scale": 1.84, "avq": 0.2, "dims": 2}, "ab_scann4": {"leaf_scale": 1.29, "avq": 0.15, "dims": 3}}
    _SWEEP = [{"leaves": a, "reorder": b} for a, b in ((1, 30), (2, 30), (4, 30), (8, 30), (16, 32), (30, 120), (50, 80), (75, 110),
                                                         (110, 120), (150, 200), (200, 300), (250, 500), (400, 300), (500, 500), (800, 1000))]
    SWEEPS = {"ab_scann1": _SWEEP, "ab_scann4": _SWEEP}

    def build(self, train, params):
        import scann
        leaves = max(8, int(params["leaf_scale"] * len(train) ** 0.5))
        metric = "dot_product" if self.angular else "squared_l2"
        searcher = (scann.scann_ops_pybind.builder(train, K, metric)
                    .set_n_training_threads(self.threads)
                    .tree(leaves, 1, training_sample_size=len(train), spherical=self.angular, quantize_centroids=True)
                    .score_ah(params["dims"], anisotropic_quantization_threshold=params["avq"])
                    .reorder(1).build())
        (self.workdir / "searcher").mkdir(parents=True, exist_ok=True)
        searcher.serialize(str(self.workdir / "searcher"))

    def open(self, params, n):
        import scann
        self.searcher = scann.scann_ops_pybind.load_searcher(str(self.workdir / "searcher"))

    def configure(self, sp):
        self.sp = sp

    def query(self, q):
        return self.searcher.search(q[0], K, self.sp["reorder"], self.sp["leaves"])[0]

    def replace(self, ids, vectors):
        raise Unsupported("the ScaNN searcher is immutable; there is no update or delete API")

    def delete(self, ids):
        raise Unsupported("the ScaNN searcher is immutable; there is no update or delete API")


class Pynndescent(Adapter):
    """PyNNDescent: NN-descent graph with diversification and degree pruning. The ann-benchmarks
    glove-100-angular NN-80-fast and NN-120-fast groups, plus a degree-64 build (n_neighbors 64,
    no extra pruning headroom). The pickle holds the prepared search graph; numba compiles on the
    first query of every process and that time is inside cold_start_s."""
    BUILDS = {"ab_nn80_fast": {"n": 80, "leaf": 20, "div": 1.0, "prune": 2.0}, "ab_nn120_fast": {"n": 120, "leaf": 20, "div": 1.0, "prune": 2.5},
              "d64": {"n": 64, "leaf": 20, "div": 1.0, "prune": 1.0}}
    SWEEPS = {k: [{"epsilon": e} for e in (0.0, 0.02, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.4)] for k in BUILDS}
    PERSISTENT_FLUSH = False

    def build(self, train, params):
        import pickle
        from pynndescent import NNDescent
        index = NNDescent(train, metric="cosine" if self.angular else "euclidean", n_neighbors=params["n"], leaf_size=params["leaf"],
                          diversify_prob=params["div"], pruning_degree_multiplier=params["prune"], n_jobs=self.threads, random_state=100)
        index.prepare()
        with open(self.workdir / "index.pkl", "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    def open(self, params, n):
        import pickle
        with open(self.workdir / "index.pkl", "rb") as f:
            self.index = pickle.load(f)

    def configure(self, sp):
        self.eps = sp["epsilon"]

    def query(self, q):
        return self.index.query(q, k=K, epsilon=self.eps)[0][0]

    def replace(self, ids, vectors):
        raise Unsupported("PyNNDescent has no deletion; update() appends new rows and cannot replace existing ids")

    def delete(self, ids):
        raise Unsupported("PyNNDescent has no deletion")


class Voyager(Adapter):
    """Spotify Voyager: an hnswlib-derived graph (own file format, auto-growing, optional float8/E4M3
    storage left at float32 here). m16e500 is the ann-benchmarks efConstruction; m32e500 and m48e500
    match the WARM build effort like the hnswlib rows. Cosine space normalises on insert."""
    BUILDS = {"m16": {"M": 16, "ef_construction": 200}, "m16e500": {"M": 16, "ef_construction": 500},
              "m32e500": {"M": 32, "ef_construction": 500}, "m48e500": {"M": 48, "ef_construction": 500}}
    SWEEPS = {k: [{"ef": e} for e in EF_SWEEP] for k in BUILDS}
    PERSISTENT_FLUSH = True

    def _space(self):
        import voyager
        return voyager.Space.Cosine if self.angular else voyager.Space.Euclidean

    def build(self, train, params):
        import voyager
        index = voyager.Index(self._space(), self.dim, M=params["M"], ef_construction=params["ef_construction"], random_seed=100, max_elements=len(train))
        index.add_items(train, num_threads=self.threads)
        index.save(str(self.workdir / "index.voyager"))

    def open(self, params, n):
        import voyager
        self.index = voyager.Index.load(str(self.workdir / "index.voyager"))

    def configure(self, sp):
        self.ef = max(sp["ef"], K)

    def query(self, q):
        return self.index.query(q[0], k=K, query_ef=self.ef)[0]

    def replace(self, ids, vectors):
        self.index.add_items(vectors, ids=[int(i) for i in ids], num_threads=1)

    def delete(self, ids):
        for i in ids:
            self.index.mark_deleted(int(i))

    def flush(self):
        self.index.save(str(self.workdir / "index.voyager"))
        sync_index(self.workdir)


class Diskann(Adapter):
    """diskannpy in-memory Vamana (StaticMemoryIndex; no SSD/PQ). r64_l100 and r64_l500 use degree 64
    with alpha 1.2; ab_r32_l125 is the ann-benchmarks glove-100-angular `vamana_125_32_1-2` group.
    The search list size L is the sweep axis."""
    BUILDS = {"r64_l100": {"R": 64, "L": 100, "alpha": 1.2}, "r64_l500": {"R": 64, "L": 500, "alpha": 1.2},
              "ab_r32_l125": {"R": 32, "L": 125, "alpha": 1.2}}
    SWEEPS = {k: [{"L": e} for e in EF_SWEEP] for k in BUILDS}

    def _metric(self):
        return "cosine" if self.angular else "l2"

    def build(self, train, params):
        import diskannpy
        diskannpy.build_memory_index(data=train, distance_metric=self._metric(), index_directory=str(self.workdir), complexity=params["L"],
                                     graph_degree=params["R"], alpha=params["alpha"], num_threads=self.threads, use_pq_build=False)

    def open(self, params, n):
        import diskannpy
        self.index = diskannpy.StaticMemoryIndex(index_directory=str(self.workdir), num_threads=1, initial_search_complexity=64,
                                                 distance_metric=self._metric(), vector_dtype=np.float32, dimensions=self.dim)

    def configure(self, sp):
        self.L = max(sp["L"], K)

    def query(self, q):
        return self.index.search(q[0], K, self.L).identifiers

    def replace(self, ids, vectors):
        raise Unsupported("StaticMemoryIndex is read-only; updates need the separate DynamicMemoryIndex build type")

    def delete(self, ids):
        raise Unsupported("StaticMemoryIndex is read-only; updates need the separate DynamicMemoryIndex build type")


SYSTEMS.update({"ngt": Ngt, "scann": Scann, "pynndescent": Pynndescent, "voyager": Voyager, "diskann": Diskann})


def wait_for_idle(max_load, max_wait):
    """(1-minute load, gate met). No gate configured (the default) records the load and None."""
    if max_load >= 1e8:
        return os.getloadavg()[0], None
    deadline = time.monotonic() + max_wait
    while True:
        load = os.getloadavg()[0]
        if load < max_load or time.monotonic() >= deadline:
            return load, load < max_load
        time.sleep(2)


def pctl(sorted_values, q):
    at = (len(sorted_values) - 1) * q
    lo = int(at)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (at - lo)


def run_build(a, adapter, meta):
    train = normalize(load_rows(Path(a.data) / "train.f32", meta["dim"], meta.get("n_train")), adapter.angular)
    rss_loaded = rss_now()
    start = time.perf_counter()
    adapter.build(train, adapter.BUILDS[a.build])
    build_s = time.perf_counter() - start
    return {"build_s": build_s, "rss_before_build": rss_loaded, "rss_after_build": rss_now(), "n": len(train)}


def run_serve(a, adapter, meta):
    n = meta["n_train"]
    test = normalize(load_rows(Path(a.data) / "test.f32", meta["dim"], a.queries), adapter.angular)
    truth = np.fromfile(Path(a.data) / "neighbors.i32", dtype="<i4").reshape(-1, meta["n_neighbors"])[: len(test), :K]
    variant = json.loads(a.variant) if a.variant else {}
    if variant:
        adapter.pre_open(variant)
    start = time.perf_counter()
    adapter.open(adapter.BUILDS[a.build], n)
    open_s = time.perf_counter() - start
    adapter.configure(adapter.SWEEPS[a.build][0])
    adapter.query(test[0:1])
    cold_start_s = (time.time_ns() - a.t0_ns) / 1e9
    for q in test[:50]:
        adapter.query(q[None])
    rss_before_sweep = rss_now()
    configs = []
    best_recall = 0.0
    for sp in adapter.SWEEPS[a.build]:
        adapter.configure(sp)
        found = [list(adapter.query(q[None])) for q in test]
        hits = sum(len(set(int(x) for x in f[:K]) & set(int(x) for x in t)) for f, t in zip(found, truth))
        recall = hits / (K * len(test))
        load, gate = wait_for_idle(a.max_load, a.max_wait)
        repeats = []
        for _ in range(a.repeats if recall >= 0.5 else 1):
            lat = np.empty(len(test), dtype=np.int64)
            for i, q in enumerate(test):
                qq = q[None]
                t = time.perf_counter_ns()
                adapter.query(qq)
                lat[i] = time.perf_counter_ns() - t
            order = np.sort(lat) / 1e3
            repeats.append({"qps": len(test) / (lat.sum() / 1e9), "p50_us": float(pctl(order, 0.5)), "p99_us": float(pctl(order, 0.99))})
        configs.append({"params": {**variant, **sp}, "recall_at_10": recall, "repeats": repeats, "load_1m": load, "load_gate_met": gate})
        best_recall = max(best_recall, recall)
        if recall >= 0.999:
            break
    return {"open_s": open_s, "cold_start_s": cold_start_s, "rss_before_sweep": rss_before_sweep, "rss_steady": rss_now(),
            "n_queries": len(test), "n": n, "configs": configs}


def run_ops(a, adapter, meta):
    count = 100
    n = meta["n_train"]
    rng = np.random.default_rng(7)
    ids = rng.choice(n, size=2 * count, replace=False)
    replace_ids, delete_ids = ids[:count], ids[count:]
    fresh = normalize(rng.standard_normal((count, meta["dim"])).astype(np.float32), adapter.angular)
    adapter.open(adapter.BUILDS[a.build], n)
    adapter.configure(adapter.SWEEPS[a.build][len(adapter.SWEEPS[a.build]) // 2])
    out = {"count": count}
    for name, fn in (("replace", lambda i: adapter.replace(replace_ids[i:i + 1], fresh[i:i + 1])),
                     ("delete", lambda i: adapter.delete(delete_ids[i:i + 1]))):
        try:
            apply_s, flush_s = 0.0, 0.0
            for i in range(count):
                t = time.perf_counter()
                fn(i)
                apply_s += time.perf_counter() - t
                t = time.perf_counter()
                adapter.flush()
                flush_s += time.perf_counter() - t
            out[name] = {"supported": True, "apply_s": apply_s, "flush_s": flush_s, "per_op_ms_durable": (apply_s + flush_s) / count * 1e3,
                         "flush_required": adapter.PERSISTENT_FLUSH, "operations_per_flush": 1}
        except Unsupported as e:
            out[name] = {"supported": False, "reason": str(e)}
    if isinstance(adapter, Warm):
        adapter.close()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["build", "serve", "ops", "versions"])
    p.add_argument("--system", required=True)
    p.add_argument("--build", default=None)
    p.add_argument("--data")
    p.add_argument("--workdir")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--queries", type=int, default=None)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-load", type=float, default=1e9)
    p.add_argument("--max-wait", type=float, default=0)
    p.add_argument("--t0-ns", type=int, default=T_IMPORT)
    p.add_argument("--variant", default=None, help="JSON settings applied before open (see Adapter.VARIANTS)")
    a = p.parse_args()
    if a.mode == "versions":
        print(json.dumps({"builds": {k: v for k, v in SYSTEMS[a.system].BUILDS.items()}}))
        return
    meta = json.loads((Path(a.data) / "meta.json").read_text())
    workdir = Path(a.workdir)
    if a.mode == "build":
        shutil.rmtree(workdir, ignore_errors=True)
        workdir.mkdir(parents=True)
    adapter = SYSTEMS[a.system](meta["dim"], workdir, a.threads, meta["metric"])
    body = {"build": run_build, "serve": run_serve, "ops": run_ops}[a.mode](a, adapter, meta)
    body["peak_rss_bytes_self"] = peak_rss()
    print("RESULT " + json.dumps(body, allow_nan=False))


if __name__ == "__main__":
    main()
