# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest


def export(tmp_path, count_delta=0, duplicate=False, invalid_recall=False, matched_subset=False,
           glass_metadata=False, duplicate_csv_row=False):
    rows = []
    native = []
    dataset = 'fashion-mnist-784-euclidean' if glass_metadata else 'glove-100-angular'
    competitor, competitor_count = ('glass', 198) if glass_metadata else ('scann', 84)
    for algorithm, count in [('hms', 108 + count_delta), (competitor, competitor_count)]:
        for index in range(count):
            param = str(index if not duplicate or index else 1)
            if matched_subset and algorithm == 'hms':
                param = f'HMS(degree={32 + 16 * (index // 36)}, ef={index})'
            if glass_metadata and algorithm == 'glass':
                param = f'Glass(build={index // 11})'
            rows.append({'dataset': dataset, 'algorithm': algorithm,
                         'parameters': param,
                         'k-nn': 1.01 if invalid_recall else 0.90 + index / 2000,
                         'qps': count - index, 'build': 12.0, 'indexsize': 256})
            native.append({'file': f'results/{dataset}/10/{algorithm}/{index}.hdf5',
                           'algorithm': algorithm, 'parameters': param, 'qps': count - index,
                           'recall_at_10': rows[-1]['k-nn'], 'query_ef': [10, 20, 30, 40, 60, 80, 120, 200, 400, 600, 800][index % 11]})
    if glass_metadata:
        (tmp_path / 'raw-point-metadata.json').write_text(json.dumps(native))
    if duplicate_csv_row:
        rows[-1] = rows[-2].copy()
    csv_path = tmp_path / 'results.csv'
    with csv_path.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    manifest = tmp_path / 'manifest.json'
    metadata = {'dataset': dataset, 'source_rev': 'a' * 40}
    if matched_subset:
        metadata['matched_hms_degrees'] = [48, 64]
    manifest.write_text(json.dumps(metadata))
    output = tmp_path / 'summary.json'
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('ann-live-export-20261024.py')),
                             str(csv_path), str(manifest), str(output)], capture_output=True, text=True)
    return result, output


@pytest.mark.parametrize('delta', [-1, 0, 1])
def test_export_requires_complete_exact_grid_and_accepts_run_manifest(tmp_path, delta):
    result, output = export(tmp_path, count_delta=delta)
    assert (result.returncode == 0) == (delta == 0), result.stderr
    if delta == 0:
        data = json.loads(output.read_text())
        assert list(data['standings']) == ['glove-100-angular']
        assert len(data['points']) == 192
        assert data['standings']['glove-100-angular']['0.9']['hms']['qps'] == 108
        assert data['standings']['glove-100-angular']['0.99']['hms'] is None


def test_export_rejects_duplicate_configuration_receipts(tmp_path):
    result, output = export(tmp_path, duplicate=True)
    assert result.returncode != 0 and not output.exists()


def test_export_rejects_impossible_recall(tmp_path):
    result, output = export(tmp_path, invalid_recall=True)
    assert result.returncode != 0 and not output.exists()


def test_preempted_points_are_retained_but_excluded_from_matched_comparisons(tmp_path):
    result, output = export(tmp_path, matched_subset=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text())
    assert len(data['points']) == 192
    assert sum(not point['same_container_comparison'] for point in data['points']) == 36
    assert data['standings']['glove-100-angular']['0.9']['hms']['qps'] == 72


def test_official_glass_names_can_repeat_when_raw_query_files_are_unique(tmp_path):
    result, output = export(tmp_path, glass_metadata=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text())
    assert len(data['points']) == 306
    point = data['standings']['fashion-mnist-784-euclidean']['0.9']['competitor']
    assert point['query_ef'] == 10
    assert point['raw_result'].endswith('/glass/0.hdf5')


def test_csv_duplication_cannot_replace_a_different_raw_query_result(tmp_path):
    result, output = export(tmp_path, glass_metadata=True, duplicate_csv_row=True)
    assert result.returncode != 0 and not output.exists()
