# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np


def fbin(path):
    with path.open('rb') as stream:
        n, dim = struct.unpack('<II', stream.read(8))
    assert n > 0 and 0 < dim <= 1 << 20
    assert path.stat().st_size == 8 + n * dim * 4
    return np.memmap(path, mode='r', dtype='<f4', offset=8, shape=(n, dim))


def live_mapping(n, changed):
    initial = n * 9 // 10
    step = n // 100
    small = step // 10
    source = np.arange(initial + (step if changed else 0), dtype=np.int64)
    if changed:
        source[:step] = -1
        source[step:step + small] = np.arange(initial + step, initial + step + small)
        source[:small] = np.arange(initial + step + small, initial + step + 2 * small)
    return source


def brute(rows, queries, source):
    query = queries.astype(np.float64)
    best_d = np.full((len(query), 11), np.inf)
    best_i = np.full((len(query), 11), -1, dtype=np.int64)
    for first in range(0, len(source), 16384):
        ids = np.flatnonzero(source[first:first + 16384] >= 0) + first
        block = rows[source[ids]].astype(np.float64)
        assert np.isfinite(block).all()
        distance = np.einsum('ij,ij->i', block, block)[None, :] - 2 * query @ block.T
        take = min(11, len(ids))
        chosen = np.argpartition(distance, take - 1, axis=1)[:, :take]
        dd = np.take_along_axis(distance, chosen, axis=1)
        ii = ids[chosen]
        for qid in range(len(query)):
            threshold = dd[qid].max()
            tied = np.flatnonzero(distance[qid] <= threshold)
            if len(tied) > take:
                order = np.lexsort((ids[tied], distance[qid, tied]))[:take]
                chosen_ties = tied[order]
                dd[qid] = distance[qid, chosen_ties]
                ii[qid] = ids[chosen_ties]
        all_d = np.concatenate((best_d, dd), axis=1)
        all_i = np.concatenate((best_i, ii), axis=1)
        order = np.lexsort((all_i, all_d), axis=1)[:, :11]
        best_d = np.take_along_axis(all_d, order, axis=1)
        best_i = np.take_along_axis(all_i, order, axis=1)
    assert (best_i >= 0).all()
    return best_i[:, :10], best_d


def self_check():
    rng = np.random.default_rng(912)
    rows = rng.integers(-2, 3, (101, 5)).astype(np.float32)
    queries = rng.integers(-2, 3, (13, 5)).astype(np.float32)
    source = np.arange(90)
    source[::3] = -1
    source[1::7] = 99
    actual, _ = brute(rows, queries, source)
    ids = np.flatnonzero(source >= 0)
    expected = np.array([ids[np.lexsort((ids, ((rows[source[ids]].astype(np.float64) - q) ** 2).sum(axis=1)))[:10]] for q in queries])
    assert np.array_equal(actual, expected)


def bound_ceiling(rows, queries, source, truth_distances, capacity):
    query = queries.astype(np.float64)
    cutoff = truth_distances[:, 9] + np.einsum('ij,ij->i', query, query)
    box_counts = np.zeros(len(query), dtype=int)
    ball_counts = np.zeros(len(query), dtype=int)
    either_counts = np.zeros(len(query), dtype=int)
    parts = 0
    for first in range(0, len(source), capacity):
        source_ids = source[first:first + capacity]
        source_ids = source_ids[source_ids >= 0]
        if not len(source_ids):
            continue
        parts += 1
        low = np.full(rows.shape[1], np.inf)
        high = np.full(rows.shape[1], -np.inf)
        total = np.zeros(rows.shape[1], dtype=np.float64)
        for at in range(0, len(source_ids), 16384):
            block = rows[source_ids[at:at + 16384]].astype(np.float64)
            low = np.minimum(low, block.min(axis=0))
            high = np.maximum(high, block.max(axis=0))
            total += block.sum(axis=0)
        center = total / len(source_ids)
        radius = 0.0
        for at in range(0, len(source_ids), 16384):
            block = rows[source_ids[at:at + 16384]].astype(np.float64)
            radius = max(radius, np.linalg.norm(block - center, axis=1).max())
        radius += 1e-8 * max(1.0, radius)
        box = np.linalg.norm(np.maximum(np.maximum(low - query, query - high), 0), axis=1) ** 2
        ball = np.maximum(np.linalg.norm(query - center, axis=1) - radius, 0) ** 2
        margin = 1e-8 * np.maximum(1.0, cutoff)
        a = box > cutoff + margin
        b = ball > cutoff + margin
        box_counts += a
        ball_counts += b
        either_counts += a | b
    return {'capacity': capacity, 'parts': parts,
            'box_fraction': float(box_counts.mean() / parts),
            'ball_fraction': float(ball_counts.mean() / parts),
            'either_fraction': float(either_counts.mean() / parts),
            'max_pruned': int(either_counts.max())}


def main(args):
    self_check()
    if args.mode == 'self-check':
        print('oracle self-check passed')
        return
    rows = fbin(args.data)
    queries = fbin(args.queries)[args.start:args.start + 128]
    assert args.rows in (1000000, 3000000) and args.start in (512, 640)
    assert len(rows) >= args.rows and rows.shape[1] == queries.shape[1]
    assert len(queries) == 128 and np.isfinite(queries).all()
    assert all(capacity in (250000, 500000) for capacity in args.capacities)
    data_sha = hashlib.file_digest(args.data.open('rb'), 'sha256').hexdigest()
    query_sha = hashlib.file_digest(args.queries.open('rb'), 'sha256').hexdigest()
    identity = {'rows': args.rows, 'query_start': args.start, 'data_sha': data_sha, 'query_sha': query_sha, 'oracle': 'float64_l2_stable_id_v1'}
    args.output.mkdir(exist_ok=True, parents=True)
    for changed in (False, True):
        phase = 'churn' if changed else 'before'
        source = live_mapping(args.rows, changed)
        truth, distances = brute(rows, queries, source)
        np.savez_compressed(args.output / f'{args.rows}-{args.start}-{phase}.npz', ids=truth, distances=distances, identity=json.dumps(identity, sort_keys=True), source=source)
        print(json.dumps({'event': 'truth', 'identity': identity, 'phase': phase, 'load': os.getloadavg(), 'ties_10_11': int(np.isclose(distances[:, 9], distances[:, 10], rtol=1e-12, atol=1e-12).sum())}), flush=True)
        bounds = [bound_ceiling(rows, queries, source, distances, capacity) for capacity in args.capacities]
        print(json.dumps({'event': 'bound_ceiling', 'identity': identity, 'phase': phase, 'bounds': bounds}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['truth', 'self-check'], default='truth')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--queries', type=Path)
    parser.add_argument('--rows', type=int)
    parser.add_argument('--start', type=int, default=512)
    parser.add_argument('--capacities', type=int, nargs='+', default=[250000, 500000])
    parser.add_argument('--output', type=Path)
    main(parser.parse_args())
