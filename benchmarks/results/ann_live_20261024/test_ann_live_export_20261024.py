import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest


def export(tmp_path, count_delta=0, duplicate=False, invalid_recall=False):
    rows = []
    for algorithm, count in [('hms', 108 + count_delta), ('scann', 84)]:
        for index in range(count):
            rows.append({'dataset': 'glove-100-angular', 'algorithm': algorithm,
                         'parameters': str(index if not duplicate or index else 1),
                         'k-nn': 1.01 if invalid_recall else 0.90 + index / 2000,
                         'qps': count - index, 'build': 12.0, 'indexsize': 256})
    csv_path = tmp_path / 'results.csv'
    with csv_path.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'dataset': 'glove-100-angular', 'source_rev': 'a' * 40}))
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
