# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""pgvector (real Postgres + pgvector), Redis (real redis-server, hash path), MongoDB (in-process fake).

Real: Postgres 18 with the pgvector extension, redis-server (core, no RediSearch module).
Mock: RediSearch FT.INFO reply and JSON documents (canned), MongoDB collection (fake of find/sort/limit;
Atlas $vectorSearch is never called by the importer, it only reads documents).
"""

import os
import shutil
import socket
import struct
import subprocess
import time

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("redis")

from holographic_sdk.importers import ImporterError
from holographic_sdk.importers.sources.base import ColumnMap
from holographic_sdk.importers.sources.mongodb import MongoDbSource, decode_id, encode_id
from holographic_sdk.importers.sources.pgvector import PgVectorSource, parse_pg_vector, quote_ident
from holographic_sdk.importers.sources.redis import RedisSource, parse_ft_info

from .conftest import assert_faithful, assert_resumable, vec

PG_BIN = "/opt/homebrew/opt/postgresql@18/bin"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------- pgvector (real)


def relocated_postgres(root):
    """A private Postgres tree with the pgvector files merged in.

    The Homebrew postgresql@18 here keeps its data files in share/postgresql, while the binaries look in
    share/postgresql@18, and pgvector's files cannot be linked into the keg. The binaries relocate relative
    to their own directory, so copy them beside symlinked data files. Returns the bin dir, or skips."""
    import glob

    cellar = os.path.realpath("/opt/homebrew/opt/postgresql@18")
    share = next((p for p in (f"{cellar}/share/postgresql@18", f"{cellar}/share/postgresql") if os.path.isdir(p)), None)
    lib = next((p for p in (f"{cellar}/lib/postgresql@18", f"{cellar}/lib/postgresql") if os.path.isdir(p)), None)
    if not share or not lib:
        pytest.skip("postgresql@18 data files not found")
    vec_share = glob.glob("/opt/homebrew/Cellar/pgvector/*/share/postgresql@18/extension")
    vec_lib = glob.glob("/opt/homebrew/Cellar/pgvector/*/lib/postgresql@18")
    bin_dir = root / "Cellar/postgresql@18/18.4/bin"
    bin_dir.mkdir(parents=True)
    for exe in ("initdb", "postgres", "pg_ctl"):
        shutil.copy2(f"{cellar}/bin/{exe}", bin_dir / exe)
    new_share, new_lib = root / "share/postgresql@18", root / "lib/postgresql@18"
    for src, dst, extra in ((share, new_share, vec_share), (lib, new_lib, vec_lib)):
        dst.mkdir(parents=True)
        for entry in os.listdir(src):
            if entry != "extension":
                os.symlink(os.path.join(src, entry), dst / entry)
        if dst is new_share:
            (dst / "extension").mkdir()
            for entry in os.listdir(f"{src}/extension"):
                os.symlink(os.path.join(src, "extension", entry), dst / "extension" / entry)
        for e in extra:
            for entry in os.listdir(e):
                target = dst / "extension" / entry if dst is new_share else dst / entry
                if not target.exists():
                    os.symlink(os.path.join(e, entry), target)
    return str(bin_dir)


@pytest.fixture(scope="module")
def pg(tmp_path_factory):
    if not os.path.exists(f"{PG_BIN}/initdb"):
        pytest.skip("postgresql@18 not installed")
    root = tmp_path_factory.mktemp("pg")
    bindir = relocated_postgres(root)
    data = root / "data"
    subprocess.run([f"{bindir}/initdb", "-D", str(data), "-A", "trust", "-U", "hms"], check=True, capture_output=True)
    port = free_port()
    subprocess.run(
        [f"{bindir}/pg_ctl", "-D", str(data), "-o", f"-p {port} -c listen_addresses=127.0.0.1 -c unix_socket_directories=", "-w", "start",
         "-l", str(root / "log")], check=True, capture_output=True)
    dsn = f"host=127.0.0.1 port={port} user=hms dbname=postgres"
    import psycopg

    con = psycopg.connect(dsn, autocommit=True)
    try:
        con.execute("CREATE EXTENSION vector")
        has_vector = True
    except psycopg.Error:
        has_vector = False
    try:
        yield dsn, con, has_vector
    finally:
        con.close()
        subprocess.run([f"{bindir}/pg_ctl", "-D", str(data), "-m", "immediate", "-w", "stop"], capture_output=True)


def vtext(v):
    return "[" + ",".join(repr(x) for x in v) + "]"


@pytest.fixture
def pg_table(pg):
    dsn, con, has_vector = pg
    if not has_vector:
        pytest.skip("pgvector extension not loadable")
    con.execute("DROP TABLE IF EXISTS docs")
    con.execute("CREATE TABLE docs (id integer PRIMARY KEY, content text, embedding vector(32), meta jsonb, "
                "ns text, created_at timestamptz)")
    for i in range(1, 41):
        con.execute("INSERT INTO docs VALUES (%s,%s,%s::vector,%s::jsonb,%s,%s)",
                    (i, f"pg doc {i}", vtext(vec(i)), f'{{"tag": {i % 4}, "nested": {{"k": "v{i}"}}}}',
                     "even" if i % 2 == 0 else "odd", f"2024-03-{(i % 28) + 1:02d}T12:00:00+00:00"))
    return dsn


def pg_source(dsn, **kw):
    cmap = ColumnMap(["text=content", "vector=embedding", "metadata=meta", "namespace=ns", "created_at=created_at"])
    return PgVectorSource(dsn, "docs", None, "id", cmap, **kw)


def test_pgvector_real_roundtrip(pg_table, run, server):
    report, tenant = run(pg_source(pg_table, page_size=7), batch_size=9)
    assert_faithful(report, 40)
    assert report.source_count == 40
    sink = server.sink(tenant)
    hit = sink.query(vector=vec(6), top_k=1, filter={"hms_source_id": "6"})[0]
    sink.close()
    assert hit["id"] == "even::6" and hit["text"] == "pg doc 6"
    m = hit["metadata"]
    assert m["hms_namespace"] == "even" and m["tag"] == 2 and m["nested"] == {"k": "v6"}
    assert m["hms_created_at"].startswith("2024-03-07")


def test_pgvector_real_resume(pg_table, server, tmp_path):
    assert_resumable(lambda: pg_source(pg_table, page_size=6), server, tmp_path, 40)


def test_pgvector_real_query_mode_and_text_keys(pg, server, run):
    dsn, con, has_vector = pg
    if not has_vector:
        pytest.skip("pgvector extension not loadable")
    con.execute("DROP TABLE IF EXISTS tdocs")
    con.execute("CREATE TABLE tdocs (slug text PRIMARY KEY, body text, emb halfvec(32))")
    for i in range(10):
        con.execute("INSERT INTO tdocs VALUES (%s,%s,%s::halfvec)", (f"s{i:02d}", f"body {i}", vtext(vec(i))))
    src = PgVectorSource(dsn, "tdocs", None, "slug", ColumnMap(["text=body", "vector=emb"]), page_size=3)
    assert [r.id for r in src.records()] == [f"s{i:02d}" for i in range(10)]
    resumed = [r.id for r in src.records(cursor="s04")]
    assert resumed == [f"s{i:02d}" for i in range(5, 10)]
    q = PgVectorSource(dsn, None, "SELECT slug, body, emb FROM tdocs ORDER BY slug",
                       "slug", ColumnMap(["id=slug", "text=body", "vector=emb"]))
    assert [r.id for r in q.records()] == [f"s{i:02d}" for i in range(10)]
    assert not q.seekable


def test_pgvector_real_sparsevec_and_identifier_guard(pg):
    dsn, con, has_vector = pg
    if not has_vector:
        pytest.skip("pgvector extension not loadable")
    text = con.execute("SELECT '{1:0.5,3:1}/5'::sparsevec::text AS t").fetchone()[0]
    assert parse_pg_vector(text) == [0.5, 0.0, 1.0, 0.0, 0.0]
    with pytest.raises(ImporterError):
        quote_ident('docs"; DROP TABLE docs; --')


def test_pgvector_secret_in_dsn_not_described():
    from holographic_sdk.importers.common import scrub

    src = PgVectorSource.from_args(
        type("A", (), {"dsn": "postgresql://u:hunter2pw@db.supabase.co:5432/postgres", "table": "t",
                       "query": None, "key_column": "id", "map": [], "page_size": 10})(), {})
    assert "hunter2pw" not in str(src.describe())
    assert "hunter2pw" not in scrub("could not connect postgresql://u:hunter2pw@db")


# ---------------------------------------------------------------- redis (real hashes)


@pytest.fixture(scope="module")
def redis_server(tmp_path_factory):
    binary = shutil.which("redis-server")
    if not binary:
        pytest.skip("redis-server not installed")
    port = free_port()
    proc = subprocess.Popen([binary, "--port", str(port), "--bind", "127.0.0.1", "--save", "", "--appendonly", "no",
                             "--dir", str(tmp_path_factory.mktemp("redis"))], stdout=subprocess.DEVNULL)
    import redis

    r = redis.Redis(port=port)
    for _ in range(100):
        try:
            r.ping()
            break
        except redis.ConnectionError:
            time.sleep(0.05)
    try:
        yield f"redis://127.0.0.1:{port}", r
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture
def redis_docs(redis_server):
    url, r = redis_server
    r.flushall()
    for i in range(30):
        r.hset(f"doc:{i}", mapping={"content": f"redis doc {i}", "embedding": struct.pack("<32f", *vec(i)),
                                    "tag": str(i % 3), "ns": "a" if i % 2 else "b", "ts": str(1_700_000_000 + i)})
    r.hset("other:1", mapping={"content": "not ours"})
    return url


def redis_source(url, **kw):
    return RedisSource(url, "doc:", None, "hash", None, "content", "embedding", "ns", "ts", None, None, **kw)


def test_redis_real_roundtrip(redis_docs, run, server):
    report, tenant = run(redis_source(redis_docs, page_size=4), batch_size=7)
    assert_faithful(report, 30)
    sink = server.sink(tenant)
    hit = sink.query(vector=struct.unpack("<32f", struct.pack("<32f", *vec(5))), top_k=1)[0]
    sink.close()
    assert hit["id"] == "a::5" and hit["metadata"]["tag"] == "2" and hit["metadata"]["hms_namespace"] == "a"
    assert hit["metadata"]["hms_created_at"].startswith("2023-11-14")


def test_redis_real_resume(redis_docs, server, tmp_path):
    assert_resumable(lambda: redis_source(redis_docs, page_size=4), server, tmp_path, 30)


def test_redis_real_without_redisearch_reports_clearly(redis_docs):
    src = RedisSource(redis_docs, "", "idx", "auto", None, None, None, None, None, None, None)
    with pytest.raises(ImporterError, match="FT.INFO idx failed"):
        list(src.records())


def test_redis_url_password_masked(redis_docs):
    src = RedisSource.from_args(type("A", (), {"url": "redis://:p4ssw0rd@h:6379/0", "prefix": "", "index": None,
                                               "key_type": "auto", "id_field": None, "text_field": "content",
                                               "vector_field": None, "namespace_field": None, "created_field": None,
                                               "updated_field": None, "metadata_fields": None, "page_size": 5})(), {})
    assert "p4ssw0rd" not in str(src.describe())


def test_redis_mock_ft_info_discovery_and_json_documents():
    resp2 = [b"index_name", b"idx", b"index_definition", [b"key_type", b"JSON", b"prefixes", [b"item:"]],
             b"attributes", [[b"identifier", b"$.emb", b"attribute", b"emb", b"type", b"VECTOR", b"DIM", b"4"]],
             b"num_docs", b"2"]
    info = parse_ft_info(resp2)
    assert info["prefixes"] == ["item:"] and info["key_type"] == "JSON" and info["vector_attr"] == "$.emb"
    assert info["vector_dim"] == 4 and info["num_docs"] == 2

    import json

    class FakeRedis:
        docs = {b"item:a": {"content": "hello", "emb": [1, 0, 0, 0], "k": 1}, b"item:b": "{not json"}

        def execute_command(self, *a):
            if a[0] == "FT.INFO":
                return resp2
            if a[0] == "JSON.GET":
                d = self.docs[a[1]]
                return d if isinstance(d, str) else json.dumps([d])
            raise AssertionError("unexpected Redis command: " + a[0])

        def scan(self, cursor, match, count):
            return 0, list(self.docs)

        def pipeline(self, transaction):
            outer = self

            class P:
                def __init__(self):
                    self.q = []

                def execute_command(self, *a):
                    self.q.append(outer.execute_command(*a))

                def execute(self):
                    return self.q

            return P()

        def type(self, key):
            return b"ReJSON-RL"

        def dbsize(self):
            return 2

    src = RedisSource("redis://x", "", "idx", "auto", None, "content", None, None, None, None, None, client=FakeRedis())
    recs = list(src.records())
    assert src.count() == 2
    assert recs[0].id == "a" and recs[0].vector == [1, 0, 0, 0] and recs[0].metadata == {"k": 1}
    assert recs[1].error and "unreadable JSON" in recs[1].error  # reported, not dropped


# ---------------------------------------------------------------- mongodb (mock)


class FakeCursor:
    def __init__(self, docs):
        self.docs = docs

    def sort(self, key, direction):
        self.docs = sorted(self.docs, key=lambda d: d[key], reverse=direction < 0)
        return self

    def limit(self, n):
        self.docs = self.docs[:n]
        return self

    def __iter__(self):
        return iter(self.docs)


class FakeCollection:
    """Implements exactly the find/count_documents surface the importer uses, including $and/$gt on _id."""

    def __init__(self, docs):
        self.docs = docs
        self.queries = []

    def _match(self, doc, q):
        for k, cond in q.items():
            if k == "$and":
                if not all(self._match(doc, c) for c in cond):
                    return False
            elif isinstance(cond, dict) and "$gt" in cond:
                if not doc[k] > cond["$gt"]:
                    return False
            elif doc.get(k) != cond:
                return False
        return True

    def find(self, q):
        self.queries.append(q)
        return FakeCursor([d for d in self.docs if self._match(d, q)])

    def count_documents(self, q):
        return len([d for d in self.docs if self._match(d, q)])


def mongo_docs(n=25):
    import datetime as dt

    return [{"_id": f"m{i:03d}", "text": f"mongo doc {i}", "embedding": vec(i), "tenant": "x" if i % 2 else "y",
             "added": dt.datetime(2024, 1, 1 + i % 20, 8, 0), "meta": {"k": i}, "score": float("inf") if i == 3 else i}
            for i in range(n)]


def mongo_source(docs, **kw):
    return MongoDbSource("mongodb://u:pw12345@h/db", "db", "c", "embedding", "text", "_id", "tenant", "added", None,
                         None, None, page_size=kw.pop("page_size", 6), collection_obj=FakeCollection(docs))


def test_mongodb_mock_roundtrip(run, server):
    docs = mongo_docs()
    report, tenant = run(mongo_source(docs), batch_size=8)
    assert_faithful(report, 25)
    assert report.stats.coercions["non_finite_float"] == 1  # inf kept as a string, counted
    sink = server.sink(tenant)
    hit = sink.query(vector=vec(7), top_k=1)[0]
    sink.close()
    assert hit["id"] == "x::m007" and hit["metadata"]["meta"] == {"k": 7} and hit["metadata"]["hms_created_at"].startswith("2024-01-08")


def test_mongodb_mock_resume_and_keyset_cursor(server, tmp_path):
    docs = mongo_docs()
    src = mongo_source(docs)
    assert_resumable(lambda: mongo_source(docs), server, tmp_path, 25)
    ids = [r.id for r in src.records(cursor=encode_id("m009"))]
    assert ids[0] == "m010" and len(ids) == 15
    assert any("$gt" in str(q) for q in src._coll.queries)
    assert "pw12345" not in str(src.describe())


def test_mongodb_objectid_cursor_roundtrip_and_unsupported_id():
    bson = pytest.importorskip("bson")
    oid = bson.ObjectId()
    assert decode_id(encode_id(oid)) == oid
    src = MongoDbSource("mongodb://h", "d", "c", "embedding", "text", "_id", None, None, None, None, None,
                        collection_obj=FakeCollection([{"_id": 1.5, "text": "t", "embedding": [1.0]}]))
    rec = list(src.records())[0]
    assert rec.error and "unsupported _id" in rec.error
