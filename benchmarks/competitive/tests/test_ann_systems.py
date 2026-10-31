# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Every ANN system in the registry is fully wired: adapter, sweeps, pinned environment, lock, facts."""

import importlib.util
import os
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from warmbench import envs, registry  # noqa: E402

WORKER = Path(__file__).resolve().parents[1] / "warmbench" / "ann_worker.py"


def _adapters():
    spec = importlib.util.spec_from_file_location("ann_worker_under_test", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SYSTEMS


@pytest.fixture
def synced(monkeypatch):
    synced = set()
    fsync = os.fsync

    def record(fd):
        stat = os.fstat(fd)
        synced.add((stat.st_dev, stat.st_ino))

    def sync(fd):
        fsync(fd)
        record(fd)

    monkeypatch.setattr(os, "fsync", sync)
    if sys.platform == "darwin":
        import fcntl
        original = fcntl.fcntl

        def fullsync(fd, command, *args):
            result = original(fd, command, *args)
            if command == fcntl.F_FULLFSYNC:
                record(fd)
            return result

        monkeypatch.setattr(fcntl, "fcntl", fullsync)
    return synced


def assert_index_synced(directory, synced):
    paths = [directory, *directory.rglob("*")]
    assert {(p.stat().st_dev, p.stat().st_ino) for p in paths} <= synced


@pytest.mark.parametrize("metric", ["angular", "euclidean"])
def test_ngt_mutations_preserve_dataset_ids(tmp_path, metric, synced):
    ngtpy = pytest.importorskip("ngtpy")
    cls = _adapters()["ngt"]
    adapter = cls(16, tmp_path, 1, metric)
    rng = np.random.default_rng(7)
    train = rng.standard_normal((200, 16)).astype(np.float32)
    fresh = rng.standard_normal((1, 16)).astype(np.float32)
    if metric == "angular":
        train /= np.linalg.norm(train, axis=1, keepdims=True)
        fresh /= np.linalg.norm(fresh, axis=1, keepdims=True)
    params = cls.BUILDS["onng_default"]
    path = str(tmp_path / "onng")
    ngtpy.create(path, 16, distance_type="Cosine" if metric == "angular" else "L2")
    index = ngtpy.Index(path)
    assert index.batch_insert(train, num_threads=1) is None
    index.save()
    index.close()
    adapter.open(params, len(train))
    adapter.configure({"epsilon": 1.0})
    assert adapter.query(train[199:200])[0] == 199
    deleted = {0, 5, 199}
    adapter.delete(sorted(deleted))
    for row in train:
        assert not deleted.intersection(adapter.query(row[None]))
    adapter.replace([7], fresh)
    assert adapter.query(fresh)[0] == 7
    adapter.flush()
    assert_index_synced(tmp_path, synced)
    adapter.index.close()
    adapter.open(params, len(train))
    assert adapter.query(fresh)[0] == 7
    adapter.replace([7], train[7:8])
    assert adapter.query(train[7:8])[0] == 7
    adapter.delete([7])
    for row in np.concatenate([train, fresh]):
        assert not (deleted | {7}).intersection(adapter.query(row[None]))
    adapter.index.close()


@pytest.mark.parametrize("system, package, build", [
    ("hnswlib", "hnswlib", "m16"),
    ("faiss-flat", "faiss", "flat"),
    ("voyager", "voyager", "m16"),
    ("sqlite-vec", "sqlite_vec", "exact"),
    ("usearch", "usearch", "m16"),
])
@pytest.mark.parametrize("metric", ["angular", "euclidean"])
def test_ann_mutations_preserve_ids_and_survive_reopen(tmp_path, system, package, build, metric, synced):
    pytest.importorskip(package)
    cls = _adapters()[system]
    adapter = cls(16, tmp_path, 1, metric)
    rng = np.random.default_rng(17)
    train = rng.standard_normal((200, 16)).astype(np.float32)
    fresh = rng.standard_normal((1, 16)).astype(np.float32)
    train[1:3] = 0
    train[1, 0], train[2, 0], train[2, 1] = 1, 10, 1
    if metric == "angular":
        train /= np.linalg.norm(train, axis=1, keepdims=True)
        fresh /= np.linalg.norm(fresh, axis=1, keepdims=True)
    params = cls.BUILDS[build]
    adapter.build(train, params)
    adapter.open(params, len(train))
    adapter.configure(cls.SWEEPS[build][-1])
    query = np.zeros((1, 16), dtype=np.float32)
    query[0, 0] = 10
    if metric == "angular":
        query /= np.linalg.norm(query)
    assert adapter.query(query)[0] == (1 if metric == "angular" else 2)
    adapter.delete([0, 5, 199])
    assert adapter.query(train[198:199])[0] == 198
    adapter.replace([7], fresh)
    assert adapter.query(fresh)[0] == 7
    adapter.flush()
    if cls.PERSISTENT_FLUSH:
        assert_index_synced(tmp_path, synced)
    if system == "sqlite-vec":
        assert adapter.db.execute("PRAGMA fullfsync").fetchone() == (1,)
        assert adapter.db.execute("PRAGMA checkpoint_fullfsync").fetchone() == (1,)
    adapter.open(params, len(train))
    adapter.configure(cls.SWEEPS[build][-1])
    assert adapter.query(fresh)[0] == 7
    adapter.delete([7])
    for row in np.concatenate([train, fresh]):
        assert not {0, 5, 7, 199}.intersection(adapter.query(row[None]))


def test_faiss_refine_delete_reports_unsupported(tmp_path):
    faiss = pytest.importorskip("faiss")
    adapter = _adapters()["faiss-ivfpq"](16, tmp_path, 1)
    adapter.index = faiss.IndexRefine(faiss.IndexFlatL2(16), faiss.IndexFlatL2(16))
    adapter.is_refine = True
    with pytest.raises(Exception, match="IndexRefine.*remove_ids") as error:
        adapter.delete([0])
    assert type(error.value).__name__ == "Unsupported"


def test_milvus_mutations_flush_the_wal_before_returning(tmp_path, synced):
    milvus = pytest.importorskip("milvus_lite")
    from milvus_lite.schema import CollectionSchema, DataType, FieldSchema

    db = milvus.MilvusLite(str(tmp_path / "db"))
    schema = CollectionSchema(fields=[FieldSchema("id", DataType.INT64, is_primary=True),
                                      FieldSchema("vector", DataType.FLOAT_VECTOR, dim=2)])
    col = db.create_collection("c", schema)
    col.insert([{"id": i, "vector": [float(i), 1.0]} for i in range(10)])
    col.flush()

    # Use the core in process; MilvusClient would start a gRPC server.
    class CoreClient:
        def upsert(self, name, rows):
            db.get_collection(name).upsert([{**row, "vector": row["vector"].tolist()} for row in rows])

        def delete(self, name, ids):
            db.get_collection(name).delete(ids)

        def flush(self, name):
            db.get_collection(name).flush()

    adapter = _adapters()["milvus-lite"](2, tmp_path, 1, "euclidean")
    adapter.client = CoreClient()
    adapter.replace([7], np.array([[0.0, 10.0]], dtype=np.float32))
    adapter.delete([0, 5, 9])
    wal = [p for p in (tmp_path / "db").rglob("wal/*") if p.is_file()]
    assert wal
    dirty = {(p.stat().st_dev, p.stat().st_ino) for p in wal}
    synced.clear()
    adapter.flush()
    assert dirty <= synced
    db.close()
    with milvus.MilvusLite(str(tmp_path / "db")) as reopened:
        rows = reopened.get_collection("c").get([0, 5, 7, 9])
        assert [row["id"] for row in rows] == [7]
        assert rows[0]["vector"] == [0.0, 10.0]


def test_ann_ops_commit_each_mutation_before_starting_the_next(tmp_path):
    from warmbench import ann_worker as worker
    train = np.random.default_rng(7).standard_normal((200, 16)).astype(np.float32)
    train.tofile(tmp_path / "train.f32")
    path = tmp_path / "control.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE v (rowid INTEGER PRIMARY KEY, embedding BLOB)")
        db.executemany("INSERT INTO v VALUES (?, ?)", [(i, row.tobytes()) for i, row in enumerate(train)])

    commits = []

    class SqliteControl(worker.SqliteVec):
        def open(self, params, n):
            self.db = sqlite3.connect(path)
            self.db.set_trace_callback(lambda sql: commits.append(sql) if sql == "COMMIT" else None)

    adapter = SqliteControl(16, tmp_path, 1, "euclidean")
    try:
        result = worker.run_ops(SimpleNamespace(data=tmp_path, build="exact"), adapter,
                                {"n_train": len(train), "dim": 16})
    finally:
        adapter.db.close()
    assert len(commits) == 200
    assert result["replace"]["supported"] and result["delete"]["supported"]
    assert result["replace"]["operations_per_flush"] == result["delete"]["operations_per_flush"] == 1
    with sqlite3.connect(path) as db:
        rows = dict(db.execute("SELECT rowid, embedding FROM v"))
    assert len(rows) == 100
    assert all(vector != train[i].tobytes() for i, vector in rows.items())


def test_every_ann_system_is_wired_end_to_end():
    adapters = _adapters()
    for name in registry.ANN_SYSTEMS:
        reg = registry.SYSTEMS[name]
        assert name in adapters, name
        cls = adapters[name]
        assert set(cls.SWEEPS) == set(cls.BUILDS), name
        assert all(cls.SWEEPS[b] for b in cls.BUILDS), name
        assert reg["env"] in envs.PINS, name
        assert (envs.REQUIREMENTS / f"{reg['env']}.txt").read_text().strip(), name
        assert name in registry.TIMING_BOUNDARY and name in registry.FACTS, name


def test_interpreter_override_is_per_env_and_defaults_to_the_shared_python():
    assert envs.python_for("hnswlib") == envs.PYTHON
    assert all(name in envs.PINS for name in envs.PYTHON_FOR)
    assert envs.python_for("voyager") != envs.PYTHON
