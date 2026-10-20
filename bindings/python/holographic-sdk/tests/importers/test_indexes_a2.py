# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""FAISS, hnswlib, Annoy and Milvus Lite sources, all against real local library instances."""

import json
import tempfile

import numpy as np
import pytest

from holographic_sdk.importers import ImporterError, ImportOptions, run_import

from .conftest import DIM, assert_faithful, assert_resumable, vec

N = 30


def matrix(n=N, dim=DIM):
    return np.asarray([vec(i, dim) for i in range(n)], dtype="float32")


def sidecars(tmp_path, n=N):
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps([f"doc-{i}" for i in range(n)]))
    texts = tmp_path / "texts.jsonl"
    texts.write_text("".join(json.dumps(f"text of document {i}\nsecond line") + "\n" for i in range(n)))
    meta = tmp_path / "meta.jsonl"
    meta.write_text("".join(json.dumps({"n": i, "tag": ["a", "b"][i % 2], "created_at": 1_700_000_000 + i}) + "\n"
                            for i in range(n)))
    return str(ids), str(texts), str(meta)


def hms_get(server, tenant, i):
    sink = server.sink(tenant)
    try:
        return sink.query(vector=vec(i), top_k=1)[0]
    finally:
        sink.close()


# ---- FAISS (real faiss-cpu) ---------------------------------------------------------

faiss = pytest.importorskip("faiss")
from holographic_sdk.importers.sources.annoy import AnnoySource  # noqa: E402
from holographic_sdk.importers.sources.faiss import FaissSource  # noqa: E402
from holographic_sdk.importers.sources.hnswlib import HnswlibSource  # noqa: E402
from holographic_sdk.importers.sources.milvus import MilvusSource  # noqa: E402


def test_faiss_real_roundtrip_flat_with_sidecars(tmp_path, run, server):
    index = tmp_path / "flat.faiss"
    flat = faiss.IndexFlatIP(DIM)
    flat.add(matrix())
    faiss.write_index(flat, str(index))
    ids, texts, meta = sidecars(tmp_path)
    src = FaissSource(str(index), ids, texts, meta, namespace="ns1", created_at_key="created_at")
    assert src.count() == N
    report, tenant = run(src, batch_size=7)
    assert_faithful(report, N)
    top = hms_get(server, tenant, 4)
    assert top["id"] == "ns1::doc-4"
    assert top["text"] == "text of document 4\nsecond line"
    md = top["metadata"]
    assert md["n"] == 4 and md["tag"] == "a" and md["hms_namespace"] == "ns1" and md["hms_source_id"] == "doc-4"
    assert md["hms_created_at"].startswith("2023-11-14T22:13:2")  # 1_700_000_004 s


def test_faiss_real_resume(tmp_path, server):
    index = tmp_path / "flat.faiss"
    flat = faiss.IndexFlatL2(DIM)
    flat.add(matrix())
    faiss.write_index(flat, str(index))
    assert_resumable(lambda: FaissSource(str(index)), server, tmp_path, N)


def test_faiss_real_idmap_native_ids(tmp_path, run, server):
    index = tmp_path / "idmap.faiss"
    idx = faiss.IndexIDMap2(faiss.IndexFlatL2(DIM))
    idx.add_with_ids(matrix(), np.arange(N, dtype="int64") + 1000)
    faiss.write_index(idx, str(index))
    report, tenant = run(FaissSource(str(index)))
    assert_faithful(report, N)
    assert hms_get(server, tenant, 7)["id"] == "1007"


def test_faiss_real_ivf_native_ids_and_resume(tmp_path, run, server):
    index = tmp_path / "ivf.faiss"
    data = matrix(200)
    quant = faiss.IndexFlatL2(DIM)
    ivf = faiss.IndexIVFFlat(quant, DIM, 4)
    ivf.train(data)
    ivf.add_with_ids(data, np.arange(200, dtype="int64") * 3)
    faiss.write_index(ivf, str(index))
    report, tenant = run(FaissSource(str(index)), sample=50)
    assert_faithful(report, 200)
    assert hms_get(server, tenant, 11)["id"] == "33"
    assert_resumable(lambda: FaissSource(str(index)), server, tmp_path, 200, batch_size=20)


def test_faiss_ivf_row_sidecars_are_refused(tmp_path):
    index = tmp_path / "ivf.faiss"
    data = matrix(100)
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(DIM), DIM, 2)
    ivf.train(data)
    ivf.add(data)
    faiss.write_index(ivf, str(index))
    _, texts, _ = sidecars(tmp_path, 100)
    with pytest.raises(ImporterError, match="list by list"):
        list(FaissSource(str(index), texts=texts).records())


def test_faiss_real_hnsw(tmp_path, run):
    index = tmp_path / "hnsw.faiss"
    h = faiss.IndexHNSWFlat(DIM, 8)
    h.add(matrix())
    faiss.write_index(h, str(index))
    report, _ = run(FaissSource(str(index)))
    assert_faithful(report, N)


def test_faiss_short_sidecar_is_an_error_not_a_loss(tmp_path):
    index = tmp_path / "flat.faiss"
    flat = faiss.IndexFlatL2(DIM)
    flat.add(matrix())
    faiss.write_index(flat, str(index))
    _, texts, _ = sidecars(tmp_path, N - 5)
    with pytest.raises(ImporterError, match="fewer rows"):
        list(FaissSource(str(index), texts=texts).records())


# ---- hnswlib (real) -----------------------------------------------------------------

hnswlib = pytest.importorskip("hnswlib")


def build_hnsw(path, space="l2", n=N):
    p = hnswlib.Index(space=space, dim=DIM)
    p.init_index(max_elements=n + 10, ef_construction=100, M=16)
    p.add_items(matrix(n), np.arange(n) * 2)  # non-contiguous labels
    p.save_index(str(path))
    return p


def test_hnswlib_real_roundtrip_with_sidecars(tmp_path, run, server):
    path = tmp_path / "h.bin"
    build_hnsw(path)
    ids, texts, meta = sidecars(tmp_path)
    src = HnswlibSource(str(path), DIM, "l2", ids, texts, meta, namespace="ns2")
    assert src.count() == N
    report, tenant = run(src)
    assert_faithful(report, N)
    top = hms_get(server, tenant, 9)
    assert top["id"] == "ns2::doc-9" and top["metadata"]["n"] == 9


@pytest.mark.parametrize("space", ["ip", "cosine"])
def test_hnswlib_real_other_spaces(tmp_path, run, space):
    path = tmp_path / "h.bin"
    build_hnsw(path, space)
    report, _ = run(HnswlibSource(str(path), DIM, space))
    assert_faithful(report, N)  # cosine indexes store normalized vectors; that is what is carried


def test_hnswlib_real_native_labels_and_resume(tmp_path, run, server):
    path = tmp_path / "h.bin"
    build_hnsw(path)
    report, tenant = run(HnswlibSource(str(path), DIM, "l2"), sample=10)
    assert_faithful(report, N)
    assert hms_get(server, tenant, 5)["id"] == "10"
    assert_resumable(lambda: HnswlibSource(str(path), DIM, "l2"), server, tmp_path, N)


def test_hnswlib_wrong_dim_is_a_clear_error(tmp_path):
    path = tmp_path / "h.bin"
    build_hnsw(path)
    with pytest.raises(ImporterError, match="cannot load"):
        HnswlibSource(str(path), DIM + 1, "l2").count()


def test_hnswlib_real_deleted_labels_are_counted_as_skipped(tmp_path, run):
    """mark_deleted keeps a label in get_ids_list but its vector is unreadable; the pipeline must report it."""
    path = tmp_path / "h.bin"
    p = build_hnsw(path)
    p.mark_deleted(4)  # label 4 is row 2
    p.save_index(str(path))
    report, _ = run(HnswlibSource(str(path), DIM, "l2"), sample=0)
    assert report.stats.skips == {"empty": 1} and report.stats.skip_samples["empty"] == ["4"]
    assert report.stats.counts["docs_acked"] == N - 1


# ---- Annoy (real) -------------------------------------------------------------------

annoy = pytest.importorskip("annoy")


def build_annoy(path, metric="angular", items=range(N)):
    a = annoy.AnnoyIndex(DIM, metric)
    for i in items:
        a.add_item(i, vec(i))
    a.build(10)
    a.save(str(path))


def test_annoy_real_roundtrip_with_sidecars(tmp_path, run, server):
    path = tmp_path / "a.ann"
    build_annoy(path)
    ids, texts, meta = sidecars(tmp_path)
    src = AnnoySource(str(path), DIM, "angular", ids, texts, meta, namespace="ns3")
    assert src.count() == N
    report, tenant = run(src)
    assert_faithful(report, N)
    top = hms_get(server, tenant, 12)
    assert top["id"] == "ns3::doc-12" and top["metadata"]["tag"] == "a"


def test_annoy_real_native_ids_and_resume(tmp_path, run, server):
    path = tmp_path / "a.ann"
    build_annoy(path, "euclidean")
    report, tenant = run(AnnoySource(str(path), DIM, "euclidean"))
    assert_faithful(report, N)
    assert hms_get(server, tenant, 3)["id"] == "3"
    assert_resumable(lambda: AnnoySource(str(path), DIM, "euclidean"), server, tmp_path, N)


def test_annoy_unfilled_item_ids_are_counted_as_skipped_not_lost(tmp_path, run):
    path = tmp_path / "a.ann"
    build_annoy(path, items=[*range(10), *range(12, 20)])  # items 10 and 11 were never added
    report, _ = run(AnnoySource(str(path), DIM, "angular"))
    assert report.stats.skips == {"empty": 2}
    assert sorted(report.stats.skip_samples["empty"]) == ["10", "11"]
    assert report.stats.counts["docs_acked"] == 18


def test_annoy_wrong_dim_is_a_clear_error(tmp_path):
    path = tmp_path / "a.ann"
    build_annoy(path)
    with pytest.raises(ImporterError, match="Annoy index"):
        AnnoySource(str(path), DIM + 3, "angular").count()


# ---- Milvus Lite (real pymilvus + milvus-lite) -------------------------------------

pymilvus = pytest.importorskip("pymilvus")


def build_milvus(db, n=30, partitions=True):
    from pymilvus import DataType, MilvusClient

    c = MilvusClient(db)
    schema = c.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("doc_id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("text", DataType.VARCHAR, max_length=4000)
    schema.add_field("n", DataType.INT64)
    schema.add_field("ts", DataType.INT64)
    params = c.prepare_index_params()
    params.add_index("embedding", index_type="FLAT", metric_type="COSINE")
    c.create_collection("docs", schema=schema, index_params=params)
    rows = [{"doc_id": f"m{i:03d}", "embedding": vec(i), "text": f"milvus text {i}", "n": i,
             "ts": 1_700_000_000 + i, "extra": {"i": i, "even": i % 2 == 0}} for i in range(n)]
    if partitions:
        c.create_partition("docs", "tenant_b")
        c.insert("docs", rows[: n // 2])
        c.insert("docs", rows[n // 2:], partition_name="tenant_b")
    else:
        c.insert("docs", rows)
    c.flush("docs")
    c.close()


@pytest.fixture
def milvus_db():
    d = tempfile.mkdtemp(prefix="mv")
    return f"{d}/m.db"


def test_milvus_real_roundtrip_partitions_and_dynamic_fields(milvus_db, run, server):
    build_milvus(milvus_db)
    src = MilvusSource(milvus_db, "docs", created_at_field="ts", batch_size=8)
    assert src.count() == 30
    report, tenant = run(src, batch_size=7)
    assert_faithful(report, 30)
    assert set(report.stats.namespaces) == {"", "tenant_b"}
    default = hms_get(server, tenant, 3)
    assert default["id"] == "m003" and default["metadata"]["hms_namespace"] == ""
    assert default["metadata"]["extra"] == {"i": 3, "even": False} and default["metadata"]["n"] == 3
    assert default["text"] == "milvus text 3"
    assert default["metadata"]["hms_created_at"].startswith("2023-11-14T22:13:2")
    other = hms_get(server, tenant, 20)
    assert other["id"] == "tenant_b::m020" and other["metadata"]["hms_namespace"] == "tenant_b"


def test_milvus_real_resume_across_partitions(milvus_db, server, tmp_path):
    build_milvus(milvus_db)
    assert_resumable(lambda: MilvusSource(milvus_db, "docs", batch_size=4), server, tmp_path, 30, batch_size=5)


def test_milvus_real_resume_inside_a_partition_uses_keyset_cursor(milvus_db):
    build_milvus(milvus_db, partitions=False)
    src = MilvusSource(milvus_db, "docs", batch_size=4)
    first = []
    for rec in src.records():
        first.append(rec)
        if len(first) == 11:
            break
    rest = [r.id for r in src.records(first[-1].cursor)]
    assert [r.id for r in first] + rest == [f"m{i:03d}" for i in range(30)]
    src.close()


def test_milvus_missing_collection_and_ambiguous_vector_field(milvus_db):
    build_milvus(milvus_db, n=4, partitions=False)
    with pytest.raises(ImporterError, match="does not exist"):
        list(MilvusSource(milvus_db, "nope").records())
    with pytest.raises(ImporterError, match="no field"):
        list(MilvusSource(milvus_db, "docs", vector_field="absent").records())


def test_milvus_describe_has_no_credentials():
    d = MilvusSource("http://user:hunter2pw@milvus.example:19530", "c", token="tok-secret-123",
                     password="pw-secret-123").describe()
    assert "hunter2pw" not in json.dumps(d) and "secret" not in json.dumps(d)


def test_sources_describe_is_stable_and_secret_free(tmp_path):
    path = tmp_path / "a.ann"
    build_annoy(path)
    a = AnnoySource(str(path), DIM, "angular").describe()
    assert a == AnnoySource(str(path), DIM, "angular").describe()
