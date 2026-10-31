# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""chroma, sqlite-vec, lancedb and qdrant sources.

Verified against REAL local instances (real client libraries writing real stores) except
`test_qdrant_mock_*`, which drives the real qdrant-client against the REST mock for the
server code path.
"""

import sqlite3

import pytest

from writerslogic_warm_sdk.importers.sources.base import ColumnMap

from .conftest import DIM, assert_faithful, assert_resumable, vec
from .mockserver import MockApi

N = 40


@pytest.fixture(autouse=True)
def _state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("WARM_IMPORT_STATE_DIR", str(tmp_path / "state"))


def warm_docs(server, tenant, vector, **flt):
    sink = server.sink(tenant)
    try:
        return sink.query(vector=vector, top_k=100, filter=flt or None)
    finally:
        sink.close()


# ---- chroma -------------------------------------------------------------------------


def make_chroma(path):
    chromadb = pytest.importorskip("chromadb")
    client = chromadb.PersistentClient(path=str(path))
    col = client.create_collection("docs")
    col.add(
        ids=[f"d{i}" for i in range(N)],
        embeddings=[vec(i) for i in range(N)],
        documents=[f"chroma document {i}" for i in range(N)],
        metadatas=[{"tag": i % 3, "flag": bool(i % 2), "created": f"2024-01-{1 + i % 28:02d}T00:00:00Z"} for i in range(N)],
    )
    other = client.create_collection("other")
    other.add(ids=["x"], embeddings=[vec(500)], documents=["lives elsewhere"], metadatas=[{"k": "v"}])
    return str(path)


def test_chroma_real_roundtrip(tmp_path, run, server):
    from writerslogic_warm_sdk.importers.sources.chroma import ChromaSource

    path = make_chroma(tmp_path / "chroma")
    src = ChromaSource(path=path, created_at_key="created", page_size=7)
    assert src.count() == N + 1
    report, tenant = run(src, batch_size=9)
    assert_faithful(report, N + 1)
    assert report.stats.namespaces == {"docs": N, "other": 1}
    hits = {m["id"]: m for m in warm_docs(server, tenant, vec(3), hms_namespace="docs")}
    meta = hits["docs::d3"]["metadata"]
    assert meta["tag"] == 0 and meta["flag"] is True and meta["warm_source"] == "chroma"
    assert meta["warm_created_at"] == "2024-01-04T00:00:00+00:00" and meta["created"] == "2024-01-04T00:00:00Z"
    assert "docs::x" not in hits and warm_docs(server, tenant, vec(500), hms_namespace="other")[0]["text"] == "lives elsewhere"


def test_chroma_real_resume(tmp_path, server):
    from writerslogic_warm_sdk.importers.sources.chroma import ChromaSource

    path = make_chroma(tmp_path / "chroma")
    assert_resumable(lambda: ChromaSource(path=path, page_size=6), server, tmp_path, N + 1)


def test_chroma_requires_exactly_one_location(tmp_path):
    from writerslogic_warm_sdk.importers.model import ImporterError
    from writerslogic_warm_sdk.importers.sources.chroma import ChromaSource

    with pytest.raises(ImporterError):
        ChromaSource()


# ---- sqlite-vec ---------------------------------------------------------------------


def make_sqlite_vec(path, kind="float"):
    sqlite_vec = pytest.importorskip("sqlite_vec")
    con = sqlite3.connect(path)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.execute(f"CREATE VIRTUAL TABLE v USING vec0(embedding {kind}[{DIM}])")
    con.execute("CREATE TABLE notes(rowid INTEGER PRIMARY KEY, uid TEXT, body TEXT, tag INTEGER, created REAL, space TEXT)")
    for i in range(N):
        data = sqlite_vec.serialize_float32(vec(i)) if kind == "float" else sqlite_vec.serialize_int8(
            [int(x * 100) for x in vec(i)])
        marker = "?" if kind == "float" else "vec_int8(?)"
        con.execute(f"INSERT INTO v(rowid, embedding) VALUES (?, {marker})", (i + 1, data))
        con.execute("INSERT INTO notes VALUES (?,?,?,?,?,?)",
                    (i + 1, f"u{i}", f"sqlite-vec note {i}", i % 5, 1_700_000_000 + i, "a" if i % 2 else "b"))
    con.commit()
    con.close()
    return str(path)


def sqlite_vec_source(db, **kw):
    from writerslogic_warm_sdk.importers.sources.sqlite_vec import SqliteVecSource

    cmap = ColumnMap(["id=uid", "text=body", "created_at=created", "namespace=space"])
    return SqliteVecSource(db, "v", None, "notes", "rowid", cmap, **kw)


def test_sqlite_vec_real_roundtrip(tmp_path, run, server):
    db = make_sqlite_vec(tmp_path / "v.db")
    src = sqlite_vec_source(db, page_size=11)
    assert src.count() == N
    report, tenant = run(src, batch_size=8)
    assert_faithful(report, N)
    assert report.stats.namespaces == {"a": N // 2, "b": N // 2}
    hit = warm_docs(server, tenant, vec(4), warm_source_id="u4")[0]
    assert hit["id"] == "b::u4" and hit["text"] == "sqlite-vec note 4"
    assert hit["metadata"]["tag"] == 4 and hit["metadata"]["warm_created_at"].startswith("2023-11-14T22:13:24")


def test_sqlite_vec_real_resume_uses_rowid_cursor(tmp_path, server):
    db = make_sqlite_vec(tmp_path / "v.db")
    assert sqlite_vec_source(db).seekable
    assert_resumable(lambda: sqlite_vec_source(db, page_size=6), server, tmp_path, N)


def test_sqlite_vec_real_int8_vectors_decode(tmp_path):
    db = make_sqlite_vec(tmp_path / "v8.db", kind="int8")
    recs = list(sqlite_vec_source(db).records())
    assert len(recs) == N and recs[2].vector == [float(int(x * 100)) for x in vec(2)]
    assert [r.cursor for r in recs[:3]] == [1, 2, 3]
    assert [r.id for r in sqlite_vec_source(db).records(cursor=N - 1)] == [f"u{N - 1}"]


def test_sqlite_vec_rejects_a_plain_table(tmp_path):
    from writerslogic_warm_sdk.importers.model import ImporterError
    from writerslogic_warm_sdk.importers.sources.sqlite_vec import SqliteVecSource

    db = make_sqlite_vec(tmp_path / "v.db")
    src = SqliteVecSource(db, "notes", None, None, "rowid", ColumnMap())
    with pytest.raises(ImporterError, match="not a vec0"):
        list(src.records())


# ---- lancedb ------------------------------------------------------------------------


def make_lancedb(path):
    lancedb = pytest.importorskip("lancedb")
    db = lancedb.connect(str(path))
    db.create_table("notes", data=[
        {"id": f"n{i}", "text": f"lance note {i}", "vector": vec(i), "tag": i % 4,
         "created": f"2024-02-{1 + i % 28:02d}T12:00:00Z", "nested": {"a": i, "b": [i, i + 1]}}
        for i in range(N)
    ])
    return str(path)


def lancedb_source(uri, **kw):
    from writerslogic_warm_sdk.importers.sources.lancedb import LanceDbSource

    return LanceDbSource(uri, ["notes"], None, ColumnMap(["created_at=created"]), **kw)


def test_lancedb_real_roundtrip(tmp_path, run, server):
    uri = make_lancedb(tmp_path / "lance")
    src = lancedb_source(uri, page_size=7)
    assert src.count() == N
    report, tenant = run(src, batch_size=10)
    assert_faithful(report, N)
    hit = warm_docs(server, tenant, vec(5), warm_source_id="n5")[0]
    assert hit["id"] == "notes::n5" and hit["metadata"]["nested"] == {"a": 5, "b": [5, 6]}
    assert hit["metadata"]["hms_namespace"] == "notes" and hit["metadata"]["warm_created_at"].startswith("2024-02-06")


def test_lancedb_real_resume(tmp_path, server):
    uri = make_lancedb(tmp_path / "lance")
    assert_resumable(lambda: lancedb_source(uri, page_size=6), server, tmp_path, N)


def test_lancedb_all_tables_by_default(tmp_path):
    from writerslogic_warm_sdk.importers.sources.lancedb import LanceDbSource

    uri = make_lancedb(tmp_path / "lance")
    import lancedb

    lancedb.connect(uri).create_table("more", data=[{"id": "m", "text": "t", "vector": vec(1)}])
    recs = list(LanceDbSource(uri, [], None, ColumnMap()).records())
    assert len(recs) == N + 1 and {r.namespace for r in recs} == {"notes", "more"}


# ---- qdrant -------------------------------------------------------------------------


def make_qdrant(path, named=False):
    qc = pytest.importorskip("qdrant_client")
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = qc.QdrantClient(path=str(path))
    cfg = {"dense": VectorParams(size=DIM, distance=Distance.COSINE), "alt": VectorParams(size=4, distance=Distance.DOT)} \
        if named else VectorParams(size=DIM, distance=Distance.COSINE)
    client.create_collection("mem", vectors_config=cfg)
    client.upsert("mem", points=[
        PointStruct(
            id=i + 1,
            vector={"dense": vec(i), "alt": [1.0, 0.0, 0.0, float(i)]} if named else vec(i),
            payload={"text": f"qdrant memory {i}", "tag": i % 3, "tenant": "t1" if i % 2 else "t2",
                     "created": 1_700_000_000 + i, "nested": {"k": i}},
        )
        for i in range(N)
    ])
    client.close()
    return str(path)


def qdrant_source(path, **kw):
    from writerslogic_warm_sdk.importers.sources.qdrant import QdrantSource

    return QdrantSource(path=path, page_size=7, created_at_field="created", namespace_field="tenant", **kw)


def test_qdrant_real_roundtrip(tmp_path, run, server):
    path = make_qdrant(tmp_path / "q")
    src = qdrant_source(path)
    assert src.count() == N
    report, tenant = run(src, batch_size=9)
    assert_faithful(report, N)
    assert report.stats.namespaces == {"t1": N // 2, "t2": N // 2}
    hit = warm_docs(server, tenant, vec(6), warm_source_id="7")[0]
    assert hit["id"] == "t2::7" and hit["text"] == "qdrant memory 6" and hit["metadata"]["nested"] == {"k": 6}
    assert hit["metadata"]["warm_created_at"].startswith("2023-11-14T22:13:26")


def test_qdrant_real_resume_by_point_id(tmp_path, server):
    path = make_qdrant(tmp_path / "q")
    assert qdrant_source(path).seekable
    assert_resumable(lambda: qdrant_source(path), server, tmp_path, N)


def test_qdrant_real_cursor_resumes_after_the_given_point(tmp_path):
    path = make_qdrant(tmp_path / "q")
    first = list(qdrant_source(path).records())
    assert len(first) == N
    rest = list(qdrant_source(path).records(cursor=first[19].cursor))
    assert [r.id for r in rest] == [r.id for r in first[20:]]


def test_qdrant_real_named_vectors_need_a_choice(tmp_path):
    from writerslogic_warm_sdk.importers.model import ImporterError

    path = make_qdrant(tmp_path / "q", named=True)
    with pytest.raises(ImporterError, match="--vector-name"):
        list(qdrant_source(path).records())
    recs = list(qdrant_source(path, vector_name="dense").records())
    assert len(recs) == N and len(recs[0].vector) == DIM


KEY = "qd-test-key-123"


def test_qdrant_description_preserves_port_without_credentials():
    from writerslogic_warm_sdk.importers.sources.qdrant import QdrantSource

    url = "http://127.0.0.1:40123"
    with QdrantSource(url=url, api_key=KEY) as source:
        description = source.describe()
    assert description["url"] == url
    assert "api_key" not in description
    assert KEY not in str(description)


def qdrant_mock(points_by_collection):
    api = MockApi(required_auth=("api-key", KEY))

    @api.route("GET", r"/collections")
    def collections(req):
        return {"result": {"collections": [{"name": n} for n in points_by_collection]}, "status": "ok", "time": 0.0}

    @api.route("POST", r"/collections/(?P<c>[^/]+)/points/scroll")
    def scroll(req):
        pts = sorted(points_by_collection[req.match["c"]], key=lambda p: p["id"])
        offset, limit = req.body.get("offset"), req.body["limit"]
        assert req.body["with_payload"] is True and req.body["with_vector"] is True
        pts = [p for p in pts if offset is None or p["id"] >= offset]
        page, rest = pts[:limit], pts[limit:]
        return {"result": {"points": page, "next_page_offset": rest[0]["id"] if rest else None},
                "status": "ok", "time": 0.0}

    @api.route("POST", r"/collections/(?P<c>[^/]+)/points/count")
    def count(req):
        return {"result": {"count": len(points_by_collection[req.match["c"]])}, "status": "ok", "time": 0.0}

    return api


def test_qdrant_mock_server_roundtrip(tmp_path, run, server):
    pytest.importorskip("qdrant_client")
    from writerslogic_warm_sdk.importers.sources.qdrant import QdrantSource

    points = {"mem": [{"id": i + 1, "vector": vec(i), "payload": {"page_content": f"lc doc {i}",
                      "metadata": {"src": f"s{i}", "tag": i % 2}}} for i in range(N)]}
    with qdrant_mock(points) as api:
        src = QdrantSource(url=api.url, api_key=KEY, page_size=9)
        assert "api_key" not in src.describe() and KEY not in str(src.describe())
        assert src.count() == N
        report, tenant = run(src, batch_size=12)
        assert_faithful(report, N)
        api.assert_clean()
    hit = warm_docs(server, tenant, vec(2), warm_source_id="3")[0]
    assert hit["text"] == "lc doc 2" and hit["metadata"]["src"] == "s2" and hit["metadata"]["hms_namespace"] == "mem"


def test_qdrant_mock_server_resume(tmp_path, server):
    pytest.importorskip("qdrant_client")
    from writerslogic_warm_sdk.importers.sources.qdrant import QdrantSource

    points = {"mem": [{"id": i + 1, "vector": vec(i), "payload": {"text": f"t{i}"}} for i in range(N)]}
    with qdrant_mock(points) as api:
        assert_resumable(lambda: QdrantSource(url=api.url, api_key=KEY, page_size=5), server, tmp_path, N)
        api.assert_clean()


def test_qdrant_mock_wrong_key_is_not_leaked_and_fails(tmp_path):
    pytest.importorskip("qdrant_client")
    from writerslogic_warm_sdk.importers.common import scrub
    from writerslogic_warm_sdk.importers.sources.qdrant import QdrantSource

    with qdrant_mock({"mem": []}) as api:
        src = QdrantSource(url=api.url, api_key="wrong-key-value")
        from writerslogic_warm_sdk.importers.common import register_secret

        register_secret("wrong-key-value")
        with pytest.raises(Exception) as err:
            src.count()
        assert "wrong-key-value" not in scrub(str(err.value))
        assert api.auth_failures >= 1
