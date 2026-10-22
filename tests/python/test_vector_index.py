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


def test_inner_product_ranks_by_dot_product(data, queries):
    scaled = data * np.random.default_rng(10).uniform(0.2, 5.0, (len(data), 1)).astype(np.float32)
    dots = queries.astype(np.float64) @ scaled.astype(np.float64).T
    truth = np.argsort(-dots, axis=1, kind="stable")[:, :K]
    idx = hm.VectorIndex("ip", degree=32, build_ef=128, threads=2)
    idx.fit(scaled)
    assert repr(idx).startswith("VectorIndex(metric='ip'")
    idx.set_query_arguments(ef=128, rerank=64)
    found = [idx.query(q, K) for q in queries]
    assert recall(found, truth) >= 0.95
    batch = idx.batch_query(queries, K)
    assert all(np.array_equal(a, b) for a, b in zip(found, batch))
    assert recall(found, exact_top_k(scaled, queries, K)) < 0.9
    with pytest.raises(ValueError, match="dimensions"):
        idx.query(np.ones(DIM + 1, np.float32), K)


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
