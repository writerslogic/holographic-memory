# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from connectivity import components, escape_control


def test_small_sink_gets_one_data_only_escape_without_changing_existing_edges():
    lengths = np.array([2, 2, 2, 1, 1], dtype=np.uint32)
    flat = np.array([1, 2, 0, 2, 0, 1, 4, 3], dtype=np.uint32)
    train = np.array([[1., 0.], [0., 1.], [-1., 0.], [1., .01], [1., .01]])
    original = train.copy()
    new_lengths, new_flat, bridges = escape_control((5, 2, 2, *([0] * 7)), lengths, flat, train, maximum=2)
    assert bridges == [{"source": 3, "target": 0, "component_size": 2}]
    assert new_lengths.tolist() == [2, 2, 2, 2, 1]
    assert new_flat.tolist() == [1, 2, 0, 2, 0, 1, 4, 0, 3]
    labels, sizes, sinks = components(new_lengths, new_flat)
    assert all(sizes[label] > 2 for label in sinks)
    assert np.array_equal(train, original)
    assert flat.tolist() == [1, 2, 0, 2, 0, 1, 4, 3]


def test_connected_graph_is_unchanged():
    lengths, flat = np.ones(4, dtype=np.uint32), np.array([1, 2, 3, 0], dtype=np.uint32)
    result = escape_control((4, 2, 2, *([0] * 7)), lengths, flat, np.ones((4, 2)), maximum=2)
    assert np.array_equal(result[0], lengths)
    assert np.array_equal(result[1], flat)
    assert result[2] == []


def test_escape_control_never_replaces_a_full_neighbor_list():
    lengths = np.full(7, 2, dtype=np.uint32)
    flat = np.array([1, 2, 2, 3, 3, 0, 0, 1, 5, 6, 4, 6, 4, 5], dtype=np.uint32)
    with pytest.raises(ValueError, match="free edge slot"):
        escape_control((7, 2, 2, *([0] * 7)), lengths, flat, np.ones((7, 2)), maximum=3)


def test_escape_control_rejects_euclidean_vectors():
    with pytest.raises(ValueError, match="angular"):
        escape_control((4, 2, 2, *([0] * 6), 1), np.ones(4, dtype=np.uint32),
                       np.array([1, 2, 3, 0], dtype=np.uint32), np.ones((4, 2)))
