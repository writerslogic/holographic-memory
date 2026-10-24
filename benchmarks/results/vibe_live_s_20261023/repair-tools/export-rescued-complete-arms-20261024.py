"""Export complete validated arms with the unchanged official VIBE metric function."""
import datetime
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import shutil
import sys

for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS']:
    os.environ[key] = '1'
import h5py
import polars as pl

BASE = Path('/Volumes/A/.win-vibe-live/benchmarks/results/vibe_live_s_20261023')
ROOT = Path('/Volumes/A/hms-coordination/public-bench/vibe-results/workspace-disabled-20261024T1435Z')
SOURCE = Path('/Volumes/A/hms-coordination/public-bench/vibe-official-7ac76c1')
REV = '12c80decb094d57ee13ba187a01bf1de8788ca2f'
sys.path.insert(0, str(SOURCE))
from export_results import export_all_results
spec = importlib.util.spec_from_file_location('vibe_rescue_runner', BASE.parents[1] / 'vibe/modal_vibe.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def main():
    inventory = json.loads((ROOT / 'volume-inventory.json').read_text())['entries']
    paths = {e['path'] for e in inventory if e['type'] == '1'}
    raw_root = ROOT / 'volume/vibe/run-s-20261023'
    snapshot = json.loads((BASE / 'snapshot-20261024T1440Z-interrupted.json').read_text())
    receipts = []
    for row in snapshot['rows']:
        ds = row['dataset']
        if row['hms_state'] != 'done':
            continue
        records = [json.loads(p.read_text()) for p in (raw_root / 'ledger' / ds).glob('*.json')]
        completed = [r for r in records if r.get('algorithm') and r['state'] == 'done']
        assert any(r['algorithm'] == 'hms' for r in completed)
        out = ROOT / 'local-official-export' / ds
        assert not out.exists(), f'Export already exists: {ds}'
        merged = out / 'results' / ds / '100'
        merged.mkdir(parents=True)
        expected = 0
        validations = {}
        native = {}
        for record in completed:
            assert record['hms_rev'] == REV
            validation = record['validation']
            assert validation['done'] and validation['expected'] == validation['valid'] == validation['markers']
            assert validation['test_count'] == 1000
            if record['algorithm'] == 'hms':
                assert validation['expected'] == 180
            src = raw_root / 'results' / ds / record['job'] / 'results' / ds / '100' / record['algorithm']
            groups = 0
            hashes = {}
            for path in src.glob('*.hdf5'):
                with h5py.File(path, 'r') as file:
                    for name, group in file.items():
                        marker = str(path.relative_to(ROOT / 'volume')) + '.complete/' + hashlib.sha256(name.encode()).hexdigest()
                        assert marker in paths
                        assert group.attrs['dataset'] == ds and group.attrs['count'] == 100 and group.attrs['run_count'] == 5
                        assert group.attrs['algo'] == record['algorithm']
                        for key in ['times', 'neighbors', 'distances']:
                            assert group[key].shape[0] == validation['test_count']
                        if record['algorithm'] == 'hms' and ds in (runner.OOD | {'coco-nomic-768-normalized'}):
                            learn = re.search(r'learn_queries=(\d+)', group.attrs['name'])
                            assert learn and int(learn.group(1)) > 0
                        groups += 1
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            assert groups == validation['expected'], (ds, record['job'], groups, validation)
            expected += groups
            validations[record['algorithm']] = validation
            native[record['algorithm']] = hashes
            shutil.copytree(src, merged / record['algorithm'], ignore=shutil.ignore_patterns('*.complete'))
        truth = ROOT / 'truth-distance-extracts' / ds
        export_all_results(str(out / 'results'), str(truth), 1, str(out / 'summary.parquet'), str(out / 'detail'))
        frame = pl.read_parquet(out / 'summary.parquet')
        assert len(frame) == expected, (ds, len(frame), expected)
        best = {}
        for algorithm in sorted(frame['algorithm'].unique().to_list()):
            best[algorithm] = {}
            for target in [0.9, 0.95, 0.99]:
                points = frame.filter((pl.col('algorithm') == algorithm) & (pl.col('recall') >= target)).sort('qps', descending=True)
                best[algorithm][f'{target:.2f}'] = None if points.is_empty() else {key: points[0, key] for key in ['qps', 'recall', 'params', 'build_time', 'index_size']}
        old_path = BASE / f'{ds}-standing.json'
        if old_path.exists():
            old = json.loads(old_path.read_text())['best_qps_at_recall']
            for algo, targets in old.items():
                for target, point in targets.items():
                    actual = best[algo][target]
                    assert (point is None) == (actual is None)
                    if point:
                        for key in ['qps', 'recall', 'build_time', 'index_size']:
                            assert abs(point[key] - actual[key]) < 1e-9, (ds, algo, target, key)
        cohorts = {r['algorithm']: runner.measurement_containers(r) for r in completed}
        standing = {'dataset': ds, 'hms_commit': REV, 'vibe_commit': runner.VIBE_REV, 'k': 100, 'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'export_scope': 'Complete validated arms only; remaining arms interrupted. Official export_all_results function unchanged; raw copies gain recalls cache, immutable rescue originals retained.',
                    'measurement_containers': cohorts, 'best_qps_at_recall': best}
        receipt = {'dataset': ds, 'validation': validations, 'raw_original_sha256': native, 'measurement_containers': cohorts,
                   'official_exporter_sha256': hashlib.sha256((SOURCE / 'export_results.py').read_bytes()).hexdigest(),
                   'summary_sha256': hashlib.sha256((out / 'summary.parquet').read_bytes()).hexdigest(), 'truth_receipt': str(truth / 'receipt.json'),
                   'no_fit_query_function_launch': True, 'groups_exported': len(frame), 'same_as_prior_official_points': old_path.exists()}
        (out / 'standing.json').write_text(json.dumps(standing, indent=2)+'\n')
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        target = BASE / 'interrupted-complete-arm-exports'
        target.mkdir(exist_ok=True)
        for name, suffix in [('standing.json', 'standing.json'), ('receipt.json', 'receipt.json'), ('summary.parquet', 'summary.parquet')]:
            shutil.copy2(out / name, target / f'{ds}-{suffix}')
        shutil.copy2(truth / 'receipt.json', target / f'{ds}-truth-receipt.json')
        receipts.append(receipt)
        print(json.dumps({'dataset': ds, 'groups_exported': len(frame), 'algorithms': list(best)}), flush=True)
    (ROOT / 'local-official-export-summary.json').write_text(json.dumps(receipts, indent=2)+'\n')

if __name__ == '__main__':
    main()
