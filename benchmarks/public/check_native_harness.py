# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.2.6", "h5py==3.14.0"]
# ///
"""Check native ANN evidence independently of the benchmark producer."""

import argparse
import copy
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {'nytimes-256-angular', 'glove-100-angular'}
SYSTEMS = {'hms', 'faiss_hnsw', 'faiss_ivfpq', 'hnswlib'}
COMMITS = {'faiss_hnsw': '7ea7339886edf8b4d9e719f593a5dfccff158274',
           'faiss_ivfpq': '7ea7339886edf8b4d9e719f593a5dfccff158274',
           'hnswlib': '3f3429661187e4c24a490a0f148fc6bc89042b3d'}
MODES = {'single_query_loop', 'batch_throughput', 'one_query_latency'}
ALLOWED_EXTERNAL = [Path.home() / '.cache/hms-bench',
                    Path.home() / '.cache/uv',
                    Path.home() / 'Library/Caches/uv',
                    Path('/usr/lib'), Path('/System/Library'),
                    Path('/Volumes/A/.hms-target-native-harness'),
                    Path('/Volumes/A/hms-native-harness')]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name, minimum=None):
    require(type(value) in (int, float) and math.isfinite(value), f'{name} must be finite')
    require(minimum is None or value >= minimum, f'{name} below minimum')
    return value


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f'invalid {name}')
    return value


def digest(data):
    return hashlib.sha256(data).hexdigest()


def path(value):
    require(isinstance(value, str) and value and '\x00' not in value, 'invalid artifact path')
    result = (ROOT / value).resolve()
    require(result.is_relative_to(ROOT) or any(result.is_relative_to(p.resolve())
            for p in ALLOWED_EXTERNAL), 'artifact path outside declared evidence roots')
    return result


def replace_fingerprint(value, source, fingerprint):
    if isinstance(value, dict):
        if value.get('path') == source and 'sha256' in value:
            value['sha256'] = fingerprint
        for child in value.values():
            replace_fingerprint(child, source, fingerprint)
    elif isinstance(value, list):
        for child in value:
            replace_fingerprint(child, source, fingerprint)


def valid_predictions(labels, count):
    ordered = np.sort(labels, axis=1)
    return np.all((labels >= -1) & (labels < count)) and np.all(
        (ordered[:, 1:] < 0) | (np.diff(ordered, axis=1) != 0))


class Files:
    def __init__(self):
        self.hash_cache = {}
        self.recall_cache = {}
        self.train_cache = {}
        self.tuning_cache = {}

    def verify(self, metadata, overrides=None, read=False):
        require(isinstance(metadata, dict), 'artifact metadata missing')
        require(isinstance(metadata.get('sha256'), str)
                and re.fullmatch('[0-9a-f]{64}', metadata['sha256']), 'invalid fingerprint')
        source = path(metadata['path'])
        data = (overrides or {}).get(source)
        size = integer(metadata['bytes'], 'artifact bytes') if 'bytes' in metadata else None
        if data is not None:
            actual_size, fingerprint = len(data), digest(data)
        else:
            state = (source.stat().st_size, source.stat().st_mtime_ns)
            cached = self.hash_cache.get(source)
            if cached is None or cached[0] != state:
                hasher = hashlib.sha256()
                with source.open('rb') as handle:
                    while chunk := handle.read(8 * 1024 * 1024):
                        hasher.update(chunk)
                cached = (state, hasher.hexdigest())
                self.hash_cache[source] = cached
            actual_size, fingerprint = state[0], cached[1]
        require(size is None or size == actual_size, 'artifact byte count differs')
        require(fingerprint == metadata['sha256'], f'artifact fingerprint differs: {metadata["path"]}')
        return data if data is not None else (source.read_bytes() if read else source)

    def array(self, metadata, dtype, shape, overrides):
        require(metadata.get('dtype') == dtype and metadata.get('shape') == list(shape),
                'array dtype or dimensions differ')
        expected = math.prod(shape) * np.dtype(dtype).itemsize
        require(metadata['bytes'] == expected and expected <= 1024 ** 3,
                'array allocation size differs or exceeds evidence bound')
        return np.frombuffer(self.verify(metadata, overrides, read=True), dtype=dtype).reshape(shape)


def timing(row, locks):
    timer = row['timer']
    require(timer['clock'] == 'perf_counter_ns', 'timer clock differs')
    start = integer(timer['start_ns'], 'timer start')
    end = integer(timer['end_ns'], 'timer end', start + 1)
    require(integer(timer['elapsed_ns'], 'elapsed', 1) == end - start, 'timer interval differs')
    begin = finite(timer['start_unix_time'], 'wall start', 0)
    finish = finite(timer['end_unix_time'], 'wall end', begin)
    require(abs((finish - begin) - (end - start) / 1e9) < 0.1,
            'wall and monotonic timing intervals differ')
    samples = row['load_samples']
    require(len(samples) >= 2, 'missing load boundary samples')
    previous = -1
    for sample in samples:
        stamp = integer(sample['monotonic_ns'], 'load sample timestamp')
        require(stamp >= previous, 'load timestamps out of order')
        previous = stamp
        finite(sample['unix_time'], 'load wall timestamp', 0)
        require(finite(sample['one_minute'], 'load', 0) < 3, 'load gate not met')
    require(samples[0]['monotonic_ns'] <= start
            and samples[-1]['monotonic_ns'] >= end, 'load samples do not enclose timing')
    require(row['load_gate_met'] is True and row['preliminary'] is False
            and row['timing_claim'] is True, 'final row is not gated evidence')
    lock = locks[row['lock_id']]
    require(lock['timer']['start_ns'] <= samples[0]['monotonic_ns']
            and lock['timer']['end_ns'] >= samples[-1]['monotonic_ns'], 'timing outside held lock')
    return end - start


def measured_recall(row, dataset, files, overrides):
    n = integer(row['n_queries'], 'query count', 1)
    start = integer(row['query_start'], 'query start')
    stop = integer(row['query_end'], 'query end', start + 1)
    require(stop - start == n and stop <= dataset['n_test'], 'query interval differs')
    labels = files.array(row['predictions'], '<i8', (n, 10), overrides)
    distances = files.array(row['distances'], '<f4', (n, 10), overrides)
    hits = files.array(row['hits'], '<i2', (n,), overrides)
    require(valid_predictions(labels, dataset['n_train']), 'prediction ID outside dataset or duplicated')
    key = (dataset['hdf5']['sha256'], row['predictions']['sha256'], start, stop)
    computed = files.recall_cache.get(key)
    if computed is None:
        with h5py.File(path(dataset['hdf5']['path']), 'r') as data:
            require(data['train'].shape == (dataset['n_train'], dataset['dim'])
                    and data['test'].shape == (dataset['n_test'], dataset['dim']), 'dataset shape differs')
            queries = np.asarray(data['test'][start:stop], dtype=np.float32)
            threshold = np.asarray(data['distances'][start:stop, 9], dtype=np.float32) + np.float32(0.001)
            truth = np.asarray(data['neighbors'][start:stop, :10], dtype=np.int64)
            train = files.train_cache.get(dataset['hdf5']['sha256'])
            if train is None:
                train = np.asarray(data['train'], dtype=np.float32)
                files.train_cache[dataset['hdf5']['sha256']] = train
            actual = np.full((n, 10), np.inf, dtype=np.float32)
            for i, query in enumerate(queries):
                qnorm = np.sum(query * query) ** 0.5
                require(qnorm > 0, 'angular query has zero norm')
                for j, identifier in enumerate(labels[i]):
                    if identifier == -1:
                        continue
                    vector = train[identifier]
                    norm = qnorm * (np.sum(vector * vector) ** 0.5)
                    actual[i, j] = np.nan if norm == 0 else 1 - np.dot(query, vector) / norm
            counts = np.sum(actual <= threshold[:, None], axis=1).astype(np.int16)
            strict = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(labels, truth)) / (n * 10)
            computed = actual, counts, strict
            files.recall_cache[key] = computed
    actual, counts, strict = computed
    require(np.allclose(distances, actual, atol=3e-7, rtol=0, equal_nan=True),
            'reported distances differ from original vector distances')
    require(np.array_equal(hits, counts), 'distance-threshold recall hits differ')
    recall = float(counts.sum()) / (n * 10)
    require(abs(finite(row['recall_at_10'], 'recall', 0) - recall) < 1e-12
            and abs(finite(row['strict_id_recall_at_10'], 'strict recall', 0) - strict) < 1e-12,
            'reported recall differs')
    return recall


def frontiers(observations, resources, curve):
    result = []
    resource_by_build = {r['build_id']: r for r in resources.values()}
    point_by_id = {p['point_id']: p for p in curve['points']}
    for dataset in sorted(DATASETS):
        for mode in sorted(MODES):
            for target in (0.90, 0.95):
                chosen = {}
                for system in sorted(SYSTEMS):
                    eligible = []
                    for key, rounds in observations.items():
                        if key[0] == dataset and key[1] == system and key[3] == mode and len(rounds) == 11:
                            if rounds[0]['recall'] >= target:
                                values = [rounds[i]['qps'] for i in range(11)]
                                eligible.append((float(np.median(values)), key[2], values, rounds[0]['recall']))
                    if eligible:
                        qps, point, values, recall = max(eligible, key=lambda item: (item[0], item[1]))
                        resource = resource_by_build[point_by_id[point]['build_id']]
                        chosen[system] = {'point_id': point, 'recall_at_10': recall,
                                          'median_qps': qps, 'round_qps': values,
                                          'index_bytes': resource['index_bytes'],
                                          'serving_file_bytes': resource['serving_file_bytes']}
                    else:
                        chosen[system] = None
                for system in sorted(SYSTEMS - {'hms'}):
                    hms, competitor = chosen['hms'], chosen[system]
                    row = {'dataset': dataset, 'mode': mode, 'target': target,
                           'competitor': system, 'hms': hms, 'them': competitor,
                           'margin_fraction': None, 'paired_ratio': None,
                           'paired_ratio_ci95': None, 'status': 'target_not_reached'}
                    if hms and competitor:
                        ratios = np.asarray(hms['round_qps']) / np.asarray(competitor['round_qps'])
                        logs = np.log(ratios)
                        rng = np.random.default_rng(20261009)
                        indices = rng.integers(0, 11, size=(10000, 11))
                        interval = np.quantile(np.exp(logs[indices].mean(axis=1)), [0.025, 0.975]).tolist()
                        margin = hms['median_qps'] / competitor['median_qps'] - 1
                        row.update(margin_fraction=margin, paired_ratio=float(np.exp(logs.mean())),
                                   paired_ratio_ci95=interval,
                                   status='ahead' if interval[0] > 1 else ('behind' if interval[1] < 1 else 'tied'))
                    result.append(row)
    return result


def check_tuning(tuning, protocol, files, overrides):
    cache_key = digest(json.dumps([tuning, protocol], sort_keys=True, allow_nan=False).encode())
    if not overrides and cache_key in files.tuning_cache:
        return files.tuning_cache[cache_key]
    require(tuning['complete'] is True and tuning['test_opened'] is False,
            'calibration incomplete or test-derived')
    expected = {(name, variant['build_id']): variant for name, dataset in protocol['datasets'].items()
                for variant in dataset['variants']}
    actual = {(item['dataset'], item['build_id']): item for item in tuning['records']}
    require(len(actual) == len(tuning['records']) and set(actual) == set(expected),
            'calibration build coverage differs')
    checked_splits = {}
    for key, record in actual.items():
        dataset = protocol['datasets'][key[0]]
        variant = expected[key]
        require(all(record[field] == variant[field]
                    for field in ('system', 'build_id', 'build_config', 'search_grid')),
                'calibration variant differs from preregistration')
        split = record['split']
        query_count = protocol['tuning']['holdout_queries']
        base_count = split['base_ids']['shape'][0]
        query_ids = files.array(split['query_ids'], '<i8', (query_count,), overrides)
        base_ids = files.array(split['base_ids'], '<i8', (base_count,), overrides)
        require(len(np.unique(query_ids)) == query_count and len(np.unique(base_ids)) == base_count
                and np.intersect1d(query_ids, base_ids).size == 0
                and np.all((query_ids >= 0) & (query_ids < dataset['n_train']))
                and np.all((base_ids >= 0) & (base_ids < dataset['n_train'])),
                'train calibration split overlaps or has invalid IDs')
        truth = files.array(split['truth_ids'], '<i8', (query_count, 10), overrides)
        true_distances = files.array(split['truth_distances'], '<f4', (query_count, 10), overrides)
        split_key = (key[0], split['query_ids']['sha256'], split['base_ids']['sha256'])
        if split_key not in checked_splits:
            files.verify(dataset['hdf5'], overrides)
            with h5py.File(path(dataset['hdf5']['path']), 'r') as data:
                base = np.asarray(data['train'][np.sort(base_ids)], dtype=np.float32)
                queries = np.asarray(data['train'][np.sort(query_ids)], dtype=np.float32)
            require(np.array_equal(base_ids, np.sort(base_ids))
                    and np.array_equal(query_ids, np.sort(query_ids)), 'train split IDs must be sorted')
            require(np.all((truth >= 0) & (truth < base_count))
                    and np.all(np.diff(np.sort(truth, axis=1), axis=1) != 0), 'calibration truth IDs invalid')
            unit_base = base / np.linalg.norm(base, axis=1, keepdims=True)
            unit_queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
            exact_tenth = np.empty(query_count, dtype=np.float32)
            for start in range(0, query_count, 32):
                matrix = 1 - unit_queries[start:start + 32] @ unit_base.T
                exact_tenth[start:start + len(matrix)] = np.partition(matrix, 9, axis=1)[:, 9]
            actual_truth = np.empty_like(true_distances)
            for i, query in enumerate(queries):
                qnorm = np.sum(query * query) ** 0.5
                for j, identifier in enumerate(truth[i]):
                    vector = base[identifier]
                    actual_truth[i, j] = 1 - np.dot(query, vector) / (qnorm * np.sum(vector * vector) ** 0.5)
            require(np.allclose(actual_truth, true_distances, atol=3e-7, rtol=0)
                    and np.allclose(true_distances[:, 9], exact_tenth, atol=1e-6, rtol=0)
                    and np.all(np.diff(true_distances, axis=1) >= -3e-7),
                    'calibration truth is not exact angular top ten')
            checked_splits[split_key] = base, queries, truth, true_distances
        base, queries, truth, true_distances = checked_splits[split_key]
        points = {point['point_id']: point for point in record['points']}
        require(len(points) == len(record['points'])
                and set(points) == {point['point_id'] for point in variant['search_grid']},
                'calibration search grid incomplete')
        for frozen_point in variant['search_grid']:
            point = points[frozen_point['point_id']]
            require(point['search_config'] == frozen_point['search_config'], 'calibration point changed')
            labels = files.array(point['predictions'], '<i8', (query_count, 10), overrides)
            distances = files.array(point['distances'], '<f4', (query_count, 10), overrides)
            hits = files.array(point['hits'], '<i2', (query_count,), overrides)
            require(valid_predictions(labels, base_count), 'calibration prediction invalid')
            actual_distances = np.full_like(distances, np.inf)
            for i, query in enumerate(queries):
                qnorm = np.sum(query * query) ** 0.5
                for j, identifier in enumerate(labels[i]):
                    if identifier == -1:
                        continue
                    vector = base[identifier]
                    actual_distances[i, j] = 1 - np.dot(query, vector) / (qnorm * np.sum(vector * vector) ** 0.5)
            actual_hits = np.sum(actual_distances <= true_distances[:, 9, None] + np.float32(0.001), axis=1)
            strict = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(labels, truth)) / (query_count * 10)
            require(np.allclose(actual_distances, distances, atol=3e-7, rtol=0)
                    and np.array_equal(actual_hits, hits)
                    and abs(float(actual_hits.sum()) / (query_count * 10) - point['recall_at_10']) < 1e-12
                    and abs(strict - point['strict_id_recall_at_10']) < 1e-12,
                    'calibration distances or scores differ')
    summary = {'calibration_builds_checked': len(actual), 'train_splits_checked': len(checked_splits)}
    if not overrides:
        files.tuning_cache[cache_key] = summary
    return summary


def check(report, files, overrides=None, allow_incomplete=False):
    overrides = overrides or {}
    require(report['schema_version'] == 1 and (report['complete'] is True or allow_incomplete),
            'report incomplete or schema differs')
    prereg = json.loads(files.verify(report['preregistration'], overrides, read=True))
    mandatory = json.loads(files.verify(prereg['mandatory_preregistration'], overrides, read=True))
    require(mandatory['ann_recall_targets'] == [0.9, 0.95] and mandatory['ann_paired_rounds'] == 11
            and mandatory['load_threshold'] == 3 and mandatory['heldout_authorized'] is False,
            'preregistered conditions differ')
    require(prereg['frozen'] is True and prereg['targets'] == [0.9, 0.95]
            and prereg['k'] == 10 and prereg['epsilon'] == 0.001
            and prereg['timing']['rounds'] == 11 and prereg['timing']['load_gate'] == 3
            and prereg['timing']['single_thread'] is True
            and prereg['timing']['max_lock_seconds'] == 900
            and set(prereg['timing']['modes']) == MODES,
            'native protocol differs from mandatory conditions')
    require(prereg['commits']['ann_benchmarks'] == '2e081ad32c1eccab72dcb739ad886c310b90f715'
            and prereg['versions'] == {'numpy': '2.2.6', 'h5py': '3.14.0',
                                      'faiss-cpu': '1.15.1', 'hnswlib': '0.8.0'},
            'native implementation or recall pins differ')
    for metadata in prereg['producer_artifacts'] + prereg['references'] + [prereg['bridge']]:
        files.verify(metadata, overrides)
    receipt = json.loads(files.verify(prereg['bridge_build'], overrides, read=True))
    require(receipt['exit_code'] == 0 and receipt['source_identity_verified'] is True
            and receipt['source_before'] == receipt['source_after']
            and receipt['environment']['RUSTFLAGS'] == '-C target-cpu=native'
            and receipt['compiler'].startswith('rustc 1.96.0 ')
            and receipt['binary'] == prereg['bridge'], 'frozen bridge build receipt differs')
    original = {}
    helper = None
    for metadata in receipt['source_before']:
        data = files.verify(metadata, overrides, read=True)
        source = Path(metadata['path'])
        if source.parent.parts[-3:] == ('src', 'core', 'qgraph'):
            original[source.name] = data
        elif source.name == 'vertex_support.rs':
            helper = data
    generated = {}
    for metadata in receipt['generated_sources']:
        generated[Path(metadata['path']).name] = files.verify(metadata, overrides, read=True)
    require(set(original) == set(generated) == {'mod.rs', 'vertex.rs', 'kernels.rs', 'build_kernels.rs'}
            and helper is not None, 'bridge omits or duplicates qgraph sources')
    for name, data in original.items():
        expected_source = data + b'\n' + helper if name == 'vertex.rs' else data
        require(generated[name] == expected_source, 'compiled bridge changes qgraph implementation')
    tuning = json.loads(files.verify(report['tuning'], overrides, read=True))
    curve = json.loads(files.verify(report['frozen_curve'], overrides, read=True))
    require(tuning['test_opened'] is False and tuning['complete'] is True and curve['round_count'] == 11
            and set(curve['modes']) == MODES, 'tuning or frozen run scope differs')
    require(curve['test_driven_curve_expansion'] is False
            and curve['tuning'] == report['tuning']
            and curve['preregistration'] == report['preregistration']
            and tuning['preregistration'] == report['preregistration'], 'frozen lineage differs')
    tuning_summary = check_tuning(tuning, prereg, files, overrides)
    frozen = finite(curve['frozen_at_unix'], 'freeze timestamp', 0)
    points = {p['point_id']: p for p in curve['points']}
    require(len(points) == len(curve['points']) and points, 'duplicate or empty frozen points')
    calibrated = {(item['dataset'], item['build_id']): item for item in tuning['records']}
    require({(point['dataset'], point['build_id']) for point in points.values()} == set(calibrated),
            'frozen curve omits calibrated builds')
    for key, record in calibrated.items():
        selected = {point['point_id']: point for point in points.values()
                    if (point['dataset'], point['build_id']) == key}
        require(set(selected) == set(record['retained_point_ids']), 'frozen curve differs from train selection')
        for candidate in record['search_grid']:
            if candidate['point_id'] in selected:
                point = selected[candidate['point_id']]
                require(point['search_config'] == candidate['search_config']
                        and point['system'] == record['system']
                        and point['build_config'] == record['build_config'], 'frozen point differs from calibration')
    require(set(report['datasets']) == DATASETS and SYSTEMS <= set(report['systems']),
            'required dataset/system missing')
    for metadata in report['artifacts']:
        files.verify(metadata, overrides)
    for name, dataset in report['datasets'].items():
        for field in ('n_train', 'n_test', 'dim'):
            integer(dataset[field], f'{name} {field}', 1)
        source = files.verify(dataset['hdf5'], overrides)
        require(source == (Path.home() / '.cache/hms-bench/downloads' / f'{name}.hdf5').resolve(),
                'dataset source differs from pinned HDF5')
    for name, system in report['systems'].items():
        require(system['architecture'] == 'arm64' and system['thread_environment']
                and all(str(value) == '1' for value in system['thread_environment'].values()),
                f'{name}: architecture or thread settings differ')
        require(re.fullmatch('[0-9a-f]{40}', system['source_commit']), 'system source pin absent')
        require(name not in COMMITS or system['source_commit'] == COMMITS[name], 'competitor commit differs')
        for item in system['module_artifacts']:
            files.verify(item, overrides)
    resources = {r['resource_id']: r for r in report['resources']}
    require(len(resources) == len(report['resources']), 'duplicate resource ID')
    for resource in resources.values():
        files.verify(resource['index_artifact'], overrides)
        require(resource['index_bytes'] == resource['index_artifact']['bytes'], 'index byte total differs')
        require(resource['serving_file_bytes'] == sum(a['bytes'] for a in resource['serving_artifacts']),
                'serving byte total differs')
        for item in resource['serving_artifacts'] + resource['bootstrap_sidecars']:
            files.verify(item, overrides)
        require(any(item == resource['index_artifact'] for item in resource['serving_artifacts']),
                'serving artifacts omit index')
        require(all(any(item['sha256'] == module['sha256'] and item['bytes'] == module['bytes']
                        for item in resource['serving_artifacts'])
                    for module in report['systems'][resource['system']]['module_artifacts']),
                'serving artifacts omit native modules')
        for field in ('index_payload_bytes', 'index_allocated_bytes'):
            if resource[field] is not None:
                integer(resource[field], field)
        workspace, rss = resource['workspace'], resource['rss']
        for field in ('persistent_bytes', 'query_interface_peak_bytes', 'observed_rss_growth_peak_bytes',
                      'observed_rss_endpoint_growth_bytes'):
            if workspace[field] is not None:
                integer(workspace[field], field)
        for value in rss.values():
            if value is not None:
                integer(value, 'RSS')
        require(rss['peak_build_process_bytes'] >= rss['after_build_bytes'], 'build RSS peak below observation')
        build = resource['build']
        require((build['load_gate_met'] is True and build['preliminary'] is False and build['timing_claim'] is True)
                or (build['preliminary'] is True and build['timing_claim'] is False),
                'ungated build timer is not marked preliminary')
    locks = {item['lock_id']: item for item in report['locks']}
    require(len(locks) == len(report['locks']), 'duplicate lock ID')
    for lock in locks.values():
        clock = lock['timer']
        integer(clock['start_ns'], 'lock acquire')
        integer(clock['end_ns'], 'lock release', clock['start_ns'] + 1)
        require(clock['elapsed_ns'] == clock['end_ns'] - clock['start_ns']
                and clock['elapsed_ns'] < 900_000_000_000,
                'timing lock held for 15 minutes or more')
        require(lock['status'] == 'released' and lock['exit_code'] == 0
                and lock['under_15_minutes'] is True
                and abs(lock['hold_seconds'] - clock['elapsed_ns'] / 1e9) < 1e-9,
                'timing lock incomplete or duration differs')
        require(lock['path'] == '/Volumes/A/.hms-target/timing.lock' and lock['pid'] > 0,
                'timing lock identity differs')
    chronological_locks = sorted(locks.values(), key=lambda item: item['timer']['start_ns'])
    require(all(a['timer']['end_ns'] <= b['timer']['start_ns']
                for a, b in zip(chronological_locks, chronological_locks[1:])), 'timing locks overlap')
    observations = defaultdict(dict)
    query_ranges = {}
    round_order = defaultdict(list)
    for row in report['rows']:
        point = points[row['point_id']]
        require(all(row[field] == point[field] for field in ('dataset', 'system', 'build_id', 'search_config')),
                'row differs from frozen point')
        resource = resources[row['resource_id']]
        require(all(resource[field] == point[field] for field in ('dataset', 'system', 'build_id', 'build_config')),
                'row resource differs from frozen build')
        integer(row['round'], 'round')
        require(row['round'] < 11 and row['mode'] in MODES, 'round or mode outside protocol')
        require(row['timer']['start_unix_time'] >= frozen, 'test row precedes frozen configuration')
        require(type(row['order_index']) is int and 0 <= row['order_index'] < len(row['order']),
                'invalid order index')
        order = row['order']
        require(len(order) == len(set(order)) and row['point_id'] == order[row['order_index']]
                and set(order) == {key for key, value in points.items() if value['dataset'] == row['dataset']},
                'paired round order omits or duplicates points')
        base_order = [p['point_id'] for p in curve['points'] if p['dataset'] == row['dataset']]
        require(order == (base_order if row['round'] % 2 == 0 else base_order[::-1]),
                'round order does not alternate')
        elapsed = timing(row, locks)
        rss, workspace = row['rss'], row['workspace']
        require(all(type(value) is int and value >= 0 for value in rss.values())
                and rss['peak_serving_process_bytes'] >= max(rss.values()), 'row RSS peak below observation')
        require(workspace['observed_rss_growth_peak_bytes'] is None
                and workspace['observed_rss_endpoint_growth_bytes'] == max(
                    0, rss['after_query_bytes'] - rss['before_query_bytes']), 'RSS endpoint growth differs')
        round_order[(row['dataset'], row['round'], row['mode'])].append(
            (row['timer']['start_ns'], row['timer']['end_ns'], row['order_index']))
        recall = measured_recall(row, report['datasets'][row['dataset']], files, overrides)
        if row['mode'] == 'one_query_latency':
            samples = files.array(row['latencies_ns'], '<i8', (row['n_queries'],), overrides)
            require(np.all(samples > 0) and int(samples.sum()) <= elapsed,
                    'latency samples exceed loop interval or include invalid duration')
            require(row['batch_size'] == 1, 'scalar latency uses a batch')
        else:
            require(row['latencies_ns'] is None, 'throughput row contains scalar latency data')
        if row['mode'] == 'single_query_loop':
            require(row['batch_size'] == 1, 'single-query loop batch size differs')
        key = (row['dataset'], row['system'], row['point_id'], row['mode'])
        query_range = (row['query_start'], row['query_end'])
        require(row['dataset'] not in query_ranges or query_ranges[row['dataset']] == query_range,
                'systems, modes or rounds use different query ranges')
        query_ranges[row['dataset']] = query_range
        require(row['round'] not in observations[key], 'duplicate paired round')
        observations[key][row['round']] = {'qps': row['n_queries'] * 1e9 / elapsed, 'recall': recall}
    for measured in round_order.values():
        measured.sort()
        require(all(a[1] <= b[0] and a[2] < b[2] for a, b in zip(measured, measured[1:])),
                'observed timing order differs from paired round metadata')
    if report['complete']:
        require(all(query_ranges.get(name) == (0, data['n_test'])
                    for name, data in report['datasets'].items()), 'final run omits test queries')
        for point in points.values():
            for mode in MODES:
                rounds = observations[(point['dataset'], point['system'], point['point_id'], mode)]
                require(set(rounds) == set(range(11)), 'frozen curve lacks 11 paired rounds')
                require(len({value['recall'] for value in rounds.values()}) == 1,
                        'identical frozen point changes recall between rounds')
    maxima = []
    for resource in report['resources']:
        matching = [row for row in report['rows'] if row['resource_id'] == resource['resource_id']]
        maxima.append({'resource_id': resource['resource_id'],
                       'peak_serving_process_bytes': max((r['rss']['peak_serving_process_bytes'] for r in matching), default=None),
                       'persistent_workspace_bytes': max((r['workspace']['persistent_bytes'] for r in matching
                                                          if r['workspace']['persistent_bytes'] is not None), default=None),
                       'query_interface_peak_bytes': max((r['workspace']['query_interface_peak_bytes'] for r in matching), default=None),
                       'observed_rss_endpoint_growth_bytes': max((r['workspace']['observed_rss_endpoint_growth_bytes'] for r in matching), default=None)})
    require(report['serving_resource_maxima'] == maxima, 'serving resource maxima differ from row observations')
    computed_frontiers = frontiers(observations, resources, curve) if report['complete'] else []
    if 'frontiers' in report:
        require(report['frontiers'] == computed_frontiers, 'frontier selection, margin or paired confidence interval differs')
    return {'passed': True, 'complete': report['complete'], 'rows_checked': len(report['rows']),
            'points_checked': len(points), 'resources_checked': len(resources),
            'recall_definition': 'angular distance <= ground-truth tenth distance + 0.001',
            'datasets': sorted(report['datasets']), 'systems': sorted(report['systems']),
            'frontiers': computed_frontiers, **tuning_summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--tamper-tests', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--tuning-only', action='store_true')
    args = parser.parse_args()
    report_bytes = args.report.read_bytes()
    report = json.loads(report_bytes)
    files = Files()
    if args.tuning_only:
        protocol = json.loads(files.verify(report['preregistration'], read=True))
        summary = {'passed': True, 'test_opened': False,
                   **check_tuning(report, protocol, files, {})}
    else:
        summary = check(report, files, allow_incomplete=args.allow_incomplete)
    summary['report_sha256'] = digest(report_bytes)
    summary['checker_sha256'] = digest(Path(__file__).read_bytes())
    if args.tamper_tests:
        require(not args.tuning_only, 'use full measurements for tamper tests')
        require(report['rows'], 'tamper tests require real measurements')
        mutations = []
        damaged = copy.deepcopy(report)
        damaged['rows'][0]['timer']['elapsed_ns'] += 1
        mutations.append(('interval', damaged, {}))
        damaged = copy.deepcopy(report)
        damaged['rows'][0]['recall_at_10'] = -1
        mutations.append(('score', damaged, {}))
        damaged = copy.deepcopy(report)
        damaged['rows'][0]['load_samples'][0]['one_minute'] = 3
        mutations.append(('load_boundary', damaged, {}))
        damaged = copy.deepcopy(report)
        damaged['rows'][0]['predictions']['sha256'] = '0' * 64
        mutations.append(('fingerprint', damaged, {}))
        artifact = report['rows'][0]['predictions']
        raw = bytearray(path(artifact['path']).read_bytes())
        raw[0] ^= 1
        mutations.append(('artifact_bytes', report, {path(artifact['path']): bytes(raw)}))
        damaged = copy.deepcopy(report)
        raw = bytearray(path(artifact['path']).read_bytes())
        raw[:8] = report['datasets'][report['rows'][0]['dataset']]['n_train'].to_bytes(8, 'little', signed=True)
        replace_fingerprint(damaged, artifact['path'], digest(raw))
        mutations.append(('prediction_with_valid_fingerprint', damaged,
                          {path(artifact['path']): bytes(raw)}))
        if report['complete']:
            damaged = copy.deepcopy(report)
            damaged['frontiers'] = copy.deepcopy(summary['frontiers'])
            comparable = next((r for r in damaged['frontiers'] if r['paired_ratio_ci95']), None)
            require(comparable is not None, 'no matched frontier for confidence interval tamper test')
            comparable['paired_ratio_ci95'][0] += 0.01
            mutations.append(('paired_confidence_interval', damaged, {}))
        passed = []
        for name, damaged, overrides in mutations:
            try:
                check(damaged, files, overrides, args.allow_incomplete)
            except (ValueError, KeyError, IndexError):
                passed.append(name)
            else:
                raise ValueError(f'tamper was accepted: {name}')
        summary['tamper_tests'] = {'passed': passed, 'count': len(passed)}
    rendered = json.dumps(summary, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
