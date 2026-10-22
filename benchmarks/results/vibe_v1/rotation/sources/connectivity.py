# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Inspect directed graph components; optionally create a data-only escape control."""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components

from check import sha


def read_graph(path):
    if path.stat().st_size > 1 << 28:
        raise ValueError("diagnostic cache exceeds 256 MiB")
    raw = path.read_bytes()
    if raw[:8] != b"HMSGRF04" or len(raw) < 104:
        raise ValueError("unsupported or truncated graph cache")
    header = struct.unpack_from("<10Q", raw, 8)
    n, dim, degree = header[:3]
    if not 100 <= n <= 250000 or not 1 <= dim <= 8192 or not 1 <= degree <= 256:
        raise ValueError("unsupported graph shape")
    count, = struct.unpack_from("<Q", raw, 88)
    if count != n:
        raise ValueError("adjacency count differs from corpus")
    lengths = np.frombuffer(raw, dtype="<u4", count=n, offset=96).copy()
    offset = 96 + 4 * n
    edges, = struct.unpack_from("<Q", raw, offset)
    if edges != int(lengths.sum()) or np.any(lengths > degree):
        raise ValueError("invalid adjacency lengths")
    end = offset + 8 + edges * 4
    if end > len(raw):
        raise ValueError("truncated adjacency")
    flat = np.frombuffer(raw, dtype="<u4", count=edges, offset=offset + 8).copy()
    if np.any(flat >= n):
        raise ValueError("adjacency ID outside corpus")
    return header, lengths, flat, raw[:88], raw[end:]


def components(lengths, flat):
    n = len(lengths)
    matrix = csr_array((np.ones(len(flat), dtype=np.uint8), flat.astype(np.int32),
                        np.r_[0, np.cumsum(lengths)].astype(np.int32)), shape=(n, n))
    count, labels = connected_components(matrix, directed=True, connection="strong")
    sizes = np.bincount(labels, minlength=count)
    source_labels = np.repeat(labels, lengths)
    outgoing = np.zeros(count, dtype=bool)
    outgoing[source_labels[source_labels != labels[flat]]] = True
    return labels, sizes, np.flatnonzero(~outgoing)


def escape_control(header, lengths, flat, train, maximum=100):
    if header[9] != 0:
        raise ValueError("escape control requires angular vectors")
    if train.shape != tuple(header[:2]) or not np.isfinite(train).all():
        raise ValueError("invalid training vectors")
    labels, sizes, sinks = components(lengths, flat)
    main = int(np.argmax(sizes))
    if sizes[main] <= maximum:
        raise ValueError("no large component for an escape control")
    normalized = train.astype(np.float64)
    norms = np.linalg.norm(normalized, axis=1)
    normalized /= np.where(norms > 0, norms, 1)[:, None]
    main_ids = np.flatnonzero(labels == main)
    main_vectors = normalized[main_ids]
    offsets = np.r_[0, np.cumsum(lengths)]
    adjacency = [flat[offsets[i]:offsets[i + 1]].tolist() for i in range(len(lengths))]
    bridges = []
    for label in sinks:
        if label == main or sizes[label] > maximum:
            continue
        members = np.flatnonzero(labels == label)
        available = members[lengths[members] < header[2]]
        if not len(available):
            raise ValueError("sink has no free edge slot")
        source = int(available[0])
        target = int(main_ids[np.argmax(main_vectors @ normalized[source])])
        adjacency[source].append(target)
        bridges.append({"source": source, "target": target, "component_size": int(sizes[label])})
    new_lengths = np.asarray([len(row) for row in adjacency], dtype="<u4")
    new_flat = np.asarray([node for row in adjacency for node in row], dtype="<u4")
    _, new_sizes, new_sinks = components(new_lengths, new_flat)
    if any(new_sizes[label] <= maximum for label in new_sinks):
        raise ValueError("small sink remains after escape control")
    return new_lengths, new_flat, bridges


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--escape-cache", type=Path)
    args = parser.parse_args()
    header, lengths, flat, prefix, suffix = read_graph(args.cache)
    labels, sizes, sinks = components(lengths, flat)
    report = {"cache_sha256": sha(args.cache), "nodes": header[0], "edges": len(flat),
              "strong_components": len(sizes), "largest_component": int(sizes.max()),
              "sink_components": [{"size": int(sizes[label]),
                                   "ids": np.flatnonzero(labels == label).tolist() if sizes[label] <= 100 else None}
                                  for label in sinks], "bridge_policy": None}
    if args.escape_cache:
        if args.data is None or args.escape_cache.exists():
            raise ValueError("requires data and a new output cache")
        train_path = args.data / "train.f32"
        train = np.fromfile(train_path, dtype="<f4").reshape(header[0], header[1])
        new_lengths, new_flat, bridges = escape_control(header, lengths, flat, train)
        args.escape_cache.write_bytes(prefix + struct.pack("<Q", len(new_lengths)) + new_lengths.tobytes()
                                     + struct.pack("<Q", len(new_flat)) + new_flat.tobytes() + suffix)
        report.update({"escape_cache_sha256": sha(args.escape_cache), "training_sha256": sha(train_path),
                       "bridges": bridges, "bridge_policy": "small sink representative to closest large-component vector; no queries or truth used; graph degree unchanged"})
    args.out.write_text(json.dumps(report, indent=2) + "\n")
