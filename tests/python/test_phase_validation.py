# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

import holographic_memory as hm


@pytest.mark.parametrize("dim,n", [(3, 16), (5, 16), (4, 15), (4, 17)])
def test_incompatible_members_raise_value_error(dim, n):
    a = hm.PhaseHVec.random(4, 16, 1)
    b = hm.PhaseHVec.random(dim, n, 2)
    for action in [
        lambda: hm.PhaseHVec.bundle([a, b]),
        lambda: hm.resonator_factorize(a, [[a, b]]),
        lambda: hm.resonator_factorize(a, [[b]]),
        lambda: hm.resonator_factorize(a, [[a], [b]]),
    ]:
        with pytest.raises(ValueError, match="dimension or resolution"):
            action()


def test_empty_collections_follow_the_declared_contract():
    a = hm.PhaseHVec.random(4, 16, 1)
    with pytest.raises(ValueError, match="empty list"):
        hm.PhaseHVec.bundle([])
    for books in [[[]], [[a], []], [[], [a]]]:
        with pytest.raises(ValueError, match="must not be empty"):
            hm.resonator_factorize(a, books)
    assert hm.resonator_factorize(a, []) == []


@pytest.mark.parametrize("dim", [65535, 65536, 65537])
def test_dimension_limit(dim):
    if dim > 65536:
        with pytest.raises(ValueError, match="phase dimensions"):
            hm.PhaseHVec.random(dim, 16, 1)
    else:
        assert hm.PhaseHVec.random(dim, 16, 1).dim == dim


@pytest.mark.parametrize("count", [65535, 65536, 65537])
def test_bundle_item_limit_without_cloning_vectors(count):
    a = hm.PhaseHVec.random(0, 16, 1)
    if count > 65536:
        with pytest.raises(ValueError, match="phase items"):
            hm.PhaseHVec.bundle([a] * count)
    else:
        assert hm.PhaseHVec.bundle([a] * count).phases() == []


@pytest.mark.parametrize("count", [63, 64, 65])
def test_bundle_element_limit(count):
    a = hm.PhaseHVec.random(65536, 2, 1)
    if count > 64:
        with pytest.raises(ValueError, match="phase elements"):
            hm.PhaseHVec.bundle([a] * count)
    else:
        assert hm.PhaseHVec.bundle([a] * count).phases() == a.phases()


@pytest.mark.parametrize("count", [63, 64, 65])
def test_factor_limit(count):
    a = hm.PhaseHVec.random(1, 16, 1)
    if count > 64:
        with pytest.raises(ValueError, match="resonator factors"):
            hm.resonator_factorize(a, [[a]] * count, max_iter=0)
    else:
        assert len(hm.resonator_factorize(a, [[a]] * count, max_iter=0)) == count


@pytest.mark.parametrize("iterations", [999, 1000, 1001])
def test_iteration_limit(iterations):
    a = hm.PhaseHVec.random(1, 16, 1)
    if iterations > 1000:
        with pytest.raises(ValueError, match="resonator iterations"):
            hm.resonator_factorize(a, [[a]], max_iter=iterations)
    else:
        assert hm.resonator_factorize(a, [[a]], max_iter=iterations) == [(0, pytest.approx(1.0))]


def test_work_admission_and_valid_algebra():
    large = hm.PhaseHVec.random(65536, 16, 1)
    with pytest.raises(ValueError, match="resonator work"):
        hm.resonator_factorize(large, [[large]] * 64)
    a = hm.PhaseHVec.random(128, 16, 1)
    b = hm.PhaseHVec.random(128, 16, 2)
    assert a.bind(b).unbind(b).phases() == a.phases()
    assert hm.PhaseHVec.bundle([a]).phases() == a.phases()
    assert hm.PhaseHVec.bundle([a, b]).phases() == hm.PhaseHVec.bundle([b, a]).phases()
    factors = hm.resonator_factorize(a.bind(b), [[a], [b]])
    assert [index for index, _ in factors] == [0, 0]
    assert all(score == pytest.approx(1.0) for _, score in factors)


@pytest.mark.parametrize("entries", [33332, 33333, 33334])
def test_work_limit_accounts_for_codebook_scans_and_iterations(entries):
    a = hm.PhaseHVec.random(1, 16, 1)
    # (3 * entries + factors**2) * dimensions * (iterations + 1).
    if entries > 33333:
        with pytest.raises(ValueError, match="resonator work"):
            hm.resonator_factorize(a, [[a] * entries], max_iter=999)
    else:
        assert hm.resonator_factorize(a, [[a] * entries], max_iter=999) == [(0, pytest.approx(1.0))]


@pytest.mark.parametrize("resolution", [1, 2, 3, 65535, 65536, 65537])
def test_resolution_boundaries(resolution):
    if resolution < 2 or resolution > 65536:
        with pytest.raises(ValueError, match="resolution"):
            hm.PhaseHVec.random(1, resolution, 1)
    else:
        assert hm.PhaseHVec.random(1, resolution, 1).n == resolution
