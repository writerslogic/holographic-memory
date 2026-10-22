# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for holographic_memory.VectorIndex (the quantized graph index binding).

Run against an installed build of the extension:
    maturin develop --release && pytest tests/python
"""
import numpy as np
import pytest

import holographic_memory as hm

N, DIM, K = 4000, 48, 10


@pytest.mark.parametrize("metric", ["angular", "euclidean", "ip"])
@pytest.mark.parametrize("bits", [4, 8])
def test_compact_adjacency_preserves_trained_results(metric, bits):
    rng = np.random.default_rng(24301)
    base = rng.normal(size=(128, 16)).astype(np.float32)
    construction = rng.normal(size=(32, 16)).astype(np.float32)
    queries = rng.normal(size=(8, 16)).astype(np.float32)
    fixed = hm.VectorIndex(metric, degree=8, build_ef=32, threads=1, code_bits=bits)
    compact = hm.VectorIndex(metric, degree=8, build_ef=32, threads=1,
                             code_bits=bits, compact_adjacency=True)
    for index in (fixed, compact):
        index.fit(base, training_queries=construction)
        index.set_query_arguments(64, 64)
    assert compact.index_bytes < fixed.index_bytes
    np.testing.assert_array_equal(fixed.batch_query(queries, 16), compact.batch_query(queries, 16))


def clustered(n, dim, seed):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(32, dim)).astype(np.float32)
    rows = centres[rng.integers(0, len(centres), n)] + 0.6 * rng.normal(size=(n, dim))
    return rows.astype(np.float32)


def exact_top_k(data, queries, k):
    data = data.astype(np.float64)
    unit = data / np.linalg.norm(data, axis=1, keepdims=True)
    scores = queries.astype(np.float64) @ unit.T
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


@pytest.fixture(scope="module")
def data():
    return clustered(N, DIM, 7)


@pytest.fixture(scope="module")
def queries():
    return clustered(200, DIM, 8)


@pytest.fixture(scope="module")
def index(data):
    idx = hm.VectorIndex("angular", degree=32, build_ef=128, threads=2)
    idx.fit(data)
    return idx


def recall(found, truth):
    return np.mean([len(set(f) & set(t)) / len(t) for f, t in zip(found, truth)])


def test_recall_against_exact_search(index, data, queries):
    truth = exact_top_k(data, queries, K)
    index.set_query_arguments(ef=128, rerank=64)
    found = [index.query(q, K) for q in queries]
    assert recall(found, truth) >= 0.95
    index.set_query_arguments(ef=10, rerank=0)
    low = recall([index.query(q, K) for q in queries], truth)
    assert low < recall(found, truth)


def test_results_are_unique_in_range_int64(index, queries):
    index.set_query_arguments(ef=64)
    for q in queries[:50]:
        ids = index.query(q, K)
        assert ids.dtype == np.int64
        assert len(ids) == K
        assert len(set(ids.tolist())) == K
        assert ids.min() >= 0 and ids.max() < N


def test_batch_query_matches_single_queries(index, queries):
    index.set_query_arguments(ef=40, rerank=32)
    batch = index.batch_query(queries, K)
    assert len(batch) == len(queries)
    for q, ids in zip(queries, batch):
        np.testing.assert_array_equal(ids, index.query(q, K))


def test_input_conversion_gives_identical_results(index, data, queries):
    index.set_query_arguments(ef=40)
    q = queries[3]
    expected = index.query(q, K)
    np.testing.assert_array_equal(index.query(q.astype(np.float64), K), expected)
    np.testing.assert_array_equal(index.query(q.tolist(), K), expected)
    strided = np.repeat(q, 2)[::2]
    assert not strided.flags.c_contiguous
    np.testing.assert_array_equal(index.query(strided, K), expected)

    fortran = hm.VectorIndex(degree=32, build_ef=128, threads=2)
    fortran.fit(np.asfortranarray(data))
    fortran.set_query_arguments(ef=40)
    np.testing.assert_array_equal(fortran.query(q, K), expected)


def test_euclidean_ranks_by_distance_not_angle(data, queries):
    # Rows with very different norms: the angular order is not the Euclidean one.
    scaled = data * np.random.default_rng(9).uniform(0.2, 5.0, (len(data), 1)).astype(np.float32)
    d2 = ((queries.astype(np.float64)[:, None, :] - scaled.astype(np.float64)[None]) ** 2).sum(-1)
    truth = np.argsort(d2, axis=1, kind="stable")[:, :K]
    idx = hm.VectorIndex("euclidean", degree=32, build_ef=128, threads=2)
    idx.fit(scaled)
    assert repr(idx).startswith("VectorIndex(metric='euclidean'")
    idx.set_query_arguments(ef=128, rerank=64)
    assert recall([idx.query(q, K) for q in queries], truth) >= 0.95
    assert recall([idx.query(q, K) for q in queries], exact_top_k(scaled, queries, K)) < 0.9


@pytest.mark.parametrize("code_bits", [4, 8])
def test_inner_product_ranks_by_dot_product(data, queries, code_bits):
    scaled = data * np.random.default_rng(10).uniform(0.2, 5.0, (len(data), 1)).astype(np.float32)
    dots = queries.astype(np.float64) @ scaled.astype(np.float64).T
    truth = np.argsort(-dots, axis=1, kind="stable")[:, :K]
    idx = hm.VectorIndex("ip", degree=32, build_ef=128, threads=2, code_bits=code_bits)
    idx.fit(scaled)
    assert repr(idx).startswith("VectorIndex(metric='ip'")
    # The reduction puts every row on one sphere, a harder graph than the raw data: more effort.
    idx.set_query_arguments(ef=500, rerank=500)
    found = [idx.query(q, K) for q in queries]
    assert recall(found, truth) >= 0.95
    batch = idx.batch_query(queries, K)
    assert all(np.array_equal(a, b) for a, b in zip(found, batch))
    assert recall(found, exact_top_k(scaled, queries, K)) < 0.9
    with pytest.raises(ValueError, match="dimensions"):
        idx.query(np.ones(DIM + 1, np.float32), K)


def test_default_traversal_codes_match_explicit_four_bits():
    rows = clustered(128, 17, 31)
    probes = clustered(8, 17, 32)
    default = hm.VectorIndex("ip", degree=16, build_ef=64, threads=1)
    explicit = hm.VectorIndex("ip", degree=16, build_ef=64, threads=1, code_bits=4)
    for idx in (default, explicit):
        idx.fit(rows)
        idx.set_query_arguments(ef=64, rerank=32)
    assert default.index_bytes == explicit.index_bytes
    for actual, expected in zip(default.batch_query(probes, K), explicit.batch_query(probes, K)):
        np.testing.assert_array_equal(actual, expected)


def test_omitted_patience_matches_explicit_zero_and_resets_early_stop():
    rows = clustered(256, 16, 51)
    probes = clustered(8, 16, 52)
    idx = hm.VectorIndex("ip", degree=16, build_ef=64, threads=1, code_bits=8)
    idx.fit(rows)
    idx.set_query_arguments(ef=64, rerank=32)
    expected = idx.batch_query(probes, K)
    idx.set_query_arguments(ef=64, rerank=32, patience=0)
    for actual, reference in zip(idx.batch_query(probes, K), expected):
        np.testing.assert_array_equal(actual, reference)
    idx.set_query_arguments(ef=64, rerank=32, patience=1)
    idx.query(probes[0], K)
    idx.set_query_arguments(ef=64, rerank=32)
    for actual, reference in zip(idx.batch_query(probes, K), expected):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("patience", [1, 4, 8, 16])
def test_positive_patience_preserves_query_result_contract(patience):
    rows = clustered(256, 16, 53)
    learn = clustered(32, 16, 54)
    probes = clustered(8, 16, 55)
    idx = hm.VectorIndex("ip", degree=16, build_ef=64, threads=1, code_bits=8)
    idx.fit(rows, training_queries=learn)
    idx.set_query_arguments(ef=64, rerank=32, patience=patience)
    for query, actual in zip(probes, idx.batch_query(probes, K)):
        assert actual.dtype == np.int64
        assert len(actual) == len(np.unique(actual)) == K
        assert np.all((actual >= 0) & (actual < len(rows)))
        np.testing.assert_array_equal(actual, idx.query(query, K))


def test_negative_patience_rejects_without_changing_query_arguments():
    rows = clustered(256, 16, 56)
    probes = clustered(8, 16, 57)
    idx = hm.VectorIndex("ip", degree=16, build_ef=64, threads=1)
    idx.fit(rows)
    idx.set_query_arguments(ef=96, rerank=64, patience=4)
    expected = idx.batch_query(probes, K)
    with pytest.raises(OverflowError):
        idx.set_query_arguments(ef=1, rerank=1, patience=-1)
    for actual, reference in zip(idx.batch_query(probes, K), expected):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("code_bits", [3, 4, 5, 7, 8, 9])
def test_traversal_code_width_boundaries(code_bits):
    if code_bits in (4, 8):
        idx = hm.VectorIndex("ip", degree=4, build_ef=16, threads=1, code_bits=code_bits)
        rows = np.array([[3, 0], [0, 2], [1, 1], [-1, 0], [0, 0]], dtype=np.float32)
        probe = np.array([1, 0.2], dtype=np.float32)
        idx.fit(rows)
        idx.set_query_arguments(ef=len(rows), rerank=len(rows))
        expected = np.argsort(-(rows.astype(np.float64) @ probe), kind="stable")
        np.testing.assert_array_equal(idx.query(probe, len(rows)), expected)
        np.testing.assert_array_equal(idx.batch_query(probe[None], len(rows))[0], expected)
    else:
        with pytest.raises(ValueError, match="code_bits must be 4 or 8"):
            hm.VectorIndex(code_bits=code_bits)


@pytest.mark.parametrize("metric", ["angular", "euclidean", "ip"])
def test_omitted_training_queries_match_explicit_none(metric):
    rows = clustered(128, 7, 41)
    probes = clustered(8, 7, 42)
    omitted = hm.VectorIndex(metric, degree=16, build_ef=64, threads=1)
    explicit = hm.VectorIndex(metric, degree=16, build_ef=64, threads=1)
    omitted.fit(rows)
    explicit.fit(rows, training_queries=None)
    assert omitted.index_bytes == explicit.index_bytes
    for index in (omitted, explicit):
        index.set_query_arguments(ef=32, rerank=32)
    for actual, expected in zip(omitted.batch_query(probes, K), explicit.batch_query(probes, K)):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("metric", ["angular", "euclidean", "ip"])
def test_training_queries_preserve_fortran_row_coordinates(metric):
    rows = clustered(128, 7, 43)
    learn = clustered(32, 7, 44)
    probes = clustered(8, 7, 45)
    contiguous = hm.VectorIndex(metric, degree=16, build_ef=64, threads=1)
    fortran = hm.VectorIndex(metric, degree=16, build_ef=64, threads=1)
    contiguous.fit(rows, training_queries=learn)
    fortran.fit(np.asfortranarray(rows), training_queries=np.asfortranarray(learn))
    assert contiguous.index_bytes == fortran.index_bytes
    for index in (contiguous, fortran):
        index.set_query_arguments(ef=32, rerank=32)
    for actual, expected in zip(contiguous.batch_query(probes, K), fortran.batch_query(probes, K)):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("invalid", [
    np.ones(7, dtype=np.float32),
    np.empty((0, 7), dtype=np.float32),
    np.ones((2, 6), dtype=np.float32),
    np.ones((2, 8), dtype=np.float32),
    np.full((2, 7), np.nan, dtype=np.float32),
    np.full((2, 7), np.inf, dtype=np.float32),
    np.full((2, 7), -np.inf, dtype=np.float32),
    np.zeros((2, 7), dtype=np.float32),
])
def test_invalid_training_queries_preserve_existing_index(invalid):
    rows = clustered(96, 7, 46)
    index = hm.VectorIndex("angular", degree=16, build_ef=32, threads=1)
    index.fit(rows)
    index.set_query_arguments(ef=32, rerank=32)
    before = index.query(rows[0], K)
    size = index.index_bytes
    with pytest.raises((TypeError, ValueError)):
        index.fit(rows[:32], training_queries=invalid)
    assert len(index) == len(rows)
    assert index.dim == rows.shape[1]
    assert index.index_bytes == size
    np.testing.assert_array_equal(index.query(rows[0], K), before)


def test_training_queries_inner_product_matches_asymmetric_l2_projection():
    rng = np.random.default_rng(47)
    rows = clustered(256, 5, 48) * rng.uniform(0.1, 7.0, (256, 1)).astype(np.float32)
    learn = clustered(64, 5, 49) * rng.uniform(0.1, 7.0, (64, 1)).astype(np.float32)
    probes = clustered(16, 5, 50)
    norms2 = (rows.astype(np.float64) ** 2).sum(axis=1)
    augmented_rows = np.column_stack((rows, np.sqrt(norms2.max() - norms2))).astype(np.float32)
    augmented_learn = np.column_stack((learn, np.zeros(len(learn), dtype=np.float32)))
    augmented_probes = np.column_stack((probes, np.zeros(len(probes), dtype=np.float32)))
    inner_product = hm.VectorIndex("ip", degree=16, build_ef=64, threads=1, code_bits=8)
    euclidean = hm.VectorIndex("euclidean", degree=16, build_ef=64, threads=1, code_bits=8)
    inner_product.fit(rows, training_queries=learn)
    euclidean.fit(augmented_rows, training_queries=augmented_learn)
    for index in (inner_product, euclidean):
        index.set_query_arguments(ef=32, rerank=32)
    for actual, expected in zip(
        inner_product.batch_query(probes, K), euclidean.batch_query(augmented_probes, K)
    ):
        np.testing.assert_array_equal(actual, expected)
    inner_product.set_query_arguments(ef=len(rows), rerank=len(rows))
    truth = np.argsort(-(probes.astype(np.float64) @ rows.astype(np.float64).T), axis=1)[:, :K]
    assert recall(inner_product.batch_query(probes, K), truth) >= 0.99


def test_k_above_size_returns_every_row_once():
    small = clustered(12, 8, 3)
    idx = hm.VectorIndex(degree=4, build_ef=16)
    idx.fit(small)
    ids = idx.query(small[0], 50)
    assert sorted(ids.tolist()) == list(range(12))


def test_rejects_invalid_arguments(index):
    with pytest.raises(ValueError, match="metric"):
        hm.VectorIndex("hamming")
    with pytest.raises(ValueError, match="even"):
        hm.VectorIndex(degree=7)
    with pytest.raises(ValueError):
        hm.VectorIndex(build_ef=0)
    with pytest.raises(RuntimeError, match="fit"):
        hm.VectorIndex().query(np.ones(4, np.float32), 1)
    with pytest.raises(ValueError, match="dimensions"):
        index.query(np.ones(DIM + 1, np.float32), K)
    with pytest.raises(ValueError, match="dimensions"):
        index.batch_query(np.ones((2, DIM + 1), np.float32), K)
    with pytest.raises(ValueError, match="zero"):
        index.query(np.zeros(DIM, np.float32), K)
    with pytest.raises(ValueError):
        index.query(np.ones(DIM, np.float32), 0)
    with pytest.raises(ValueError):
        index.set_query_arguments(ef=0)
    with pytest.raises(TypeError):
        hm.VectorIndex().fit(np.ones(8, np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        hm.VectorIndex().fit(np.full((4, 4), np.nan, np.float32))


def test_reports_size(index):
    assert len(index) == N
    assert index.dim == DIM
    assert index.index_bytes > 0
