CONFIG = {'dataset': 'landmark-nomic-768-normalized', 'container_id': 'ta-01M5PDWQVEXHTNJSV6WPESQA3R', 'hms_rev': '12c80decb094d57ee13ba187a01bf1de8788ca2f', 'thread_env': {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'TBB_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'VECLIB_NUM_THREADS': '1', 'BLIS_NUM_THREADS': '1', 'GOTO_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1', 'JULIA_NUM_THREADS': '1', 'RAYON_NUM_THREADS': '1', 'OMP_DYNAMIC': 'false', 'MKL_DYNAMIC': 'false', 'HDF5_USE_FILE_LOCKING': 'FALSE'}, 'jobtool': '\nimport glob, json, os, re, sys\nimport h5py\nfrom vibe.definitions import get_definitions\nfrom vibe.main import parse_dataset_string\nfrom vibe.results import build_result_filepath, _completion_marker_filepath\n\nmode, ds, algo, defs_dir = sys.argv[1:5]\ndim, dist, pt = parse_dataset_string(ds)\ndefs = [d for d in get_definitions(dimension=dim, point_type=pt, distance_metric=dist, count=100, base_dir=defs_dir)\n        if d.algorithm == algo and not d.disabled]\nexpected = [build_result_filepath(ds, 100, d, q) for d in defs for q in (d.query_argument_groups or [[]])]\nroot = os.path.join("results", ds, "100", algo)\n\ndef opens(path):\n    try:\n        with h5py.File(path, "r") as f:\n            return list(f.keys())\n    except Exception:\n        return None\n\nif mode == "clean":\n    removed = []\n    for path in glob.glob(os.path.join(root, "*.hdf5")):\n        groups = opens(path)\n        if groups is None:\n            os.remove(path)\n            removed.append(path)\n            continue\n        stale = [g for g in groups if not os.path.exists(_completion_marker_filepath(path, g))]\n        if stale:\n            with h5py.File(path, "a") as f:\n                for g in stale:\n                    del f[g]\n            removed += [f"{path}:{g}" for g in stale]\n    for marker_dir in glob.glob(os.path.join(root, "*.hdf5.complete")):\n        path = marker_dir[: -len(".complete")]\n        groups = set(opens(path) or []) if os.path.exists(path) else set()\n        for m in os.listdir(marker_dir):\n            body = open(os.path.join(marker_dir, m)).read()\n            if body not in groups:\n                os.remove(os.path.join(marker_dir, m))\n                removed.append(os.path.join(marker_dir, m))\n    print(json.dumps({"removed": removed}))\nelse:\n    with h5py.File(os.path.join("data", ds + ".hdf5"), "r") as f:\n        n_test = int(f["test"].shape[0])\n        has_learn = "learn" in f and "learn_neighbors" in f\n    # An OOD-capable HMS entry must have trained on the learn split; learn_queries=0 means VIBE called fit().\n    need_learn = {id(d) for d in defs if algo.startswith("hms") and has_learn and d.ood}\n    valid, problems = 0, []\n    for (path, sp), d in zip(expected, [d for d in defs for q in (d.query_argument_groups or [[]])]):\n        try:\n            if sp and not os.path.exists(_completion_marker_filepath(path, sp)):\n                raise ValueError("no completion marker")\n            with h5py.File(path, "r") as f:\n                g = f[sp] if sp else f[list(f.keys())[0]]\n                for key in ("times", "neighbors", "distances"):\n                    if g[key].shape[0] != n_test:\n                        raise ValueError(f"{key} has {g[key].shape[0]} rows, dataset has {n_test} queries")\n                if id(d) in need_learn:\n                    m = re.search(r"learn_queries=(\\d+)", str(g.attrs.get("name", "")))\n                    if not m or int(m.group(1)) == 0:\n                        raise ValueError(f"OOD dataset but the run did not train on learn queries: {g.attrs.get(\'name\')!r}")\n            valid += 1\n        except Exception as e:\n            problems.append(f"{os.path.basename(path)}:{sp}: {type(e).__name__}: {e}"[:300])\n    markers = len(glob.glob(os.path.join(root, "*.hdf5.complete", "*")))\n    print(json.dumps({"expected": len(expected), "valid": valid, "test_count": n_test, "markers": markers,\n                      "done": bool(expected) and valid == len(expected), "problems": problems[:20]}))\n'}
"""Repair migrated HMS builds before the original official exporter is invoked."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import yaml

ds, cid = CONFIG['dataset'], CONFIG['container_id']
assert ds in ('landmark-nomic-768-normalized', 'yahoo-minilm-384-normalized')
assert os.environ.get('MODAL_TASK_ID') == cid
root = Path('/vol/vibe/run-s-20261023')
job = 'hms@12c80decb094'
ledger_path = root / 'ledger' / ds / f'{job}.json'
record = json.loads(ledger_path.read_text())
if record.get('provenance_repair', {}).get('state') == 'done':
    raise SystemExit(0)
assert record['hms_rev'] == CONFIG['hms_rev'] and record['container_id'] == cid and record['state'] == 'done'
for path in ledger_path.parent.glob('*.json'):
    other = json.loads(path.read_text())
    assert other['state'] in ('done', 'failed'), f'Another arm remains active: {path}'
spent = sum(json.loads(path.read_text()).get('usd', 0) for path in (root / 'spend').glob('*.json'))
assert spent < 300, f'No provenance repair may start above the $300 soft cap: ${spent:.2f}'
start = datetime.datetime.fromisoformat(record['start'].replace('Z', '+00:00')).timestamp()
local_job = Path('/work/jobs') / job
raw = local_job / 'results' / ds / '100/hms'
volume_raw = root / 'results' / ds / job / 'results' / ds / '100/hms'
old_files = [path for path in raw.glob('*.hdf5') if any(
    marker.stat().st_mtime < start for marker in path.with_name(path.name + '.complete').iterdir())]
assert old_files, 'No migrated groups remain; do not repeat completed work'
degrees = []
for path in old_files:
    match = re.fullmatch(r'normalized_(32|48|64|96)_500_8_true\.hdf5', path.name)
    assert match, path.name
    degrees.append(int(match.group(1)))
assert len(degrees) <= 3, 'Repair must stay within the parent exporter four-hour timeout'
archive = root / 'diagnostic-before-provenance-repair' / ds
archive.mkdir(parents=True)
(archive / 'ledger.json').write_text(json.dumps(record, indent=2) + '\n')
original_hashes = {}
for path in old_files:
    original_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copy2(path, archive / path.name)
    shutil.copytree(path.with_name(path.name + '.complete'), archive / (path.name + '.complete'))
(archive / 'raw-results.sha256.json').write_text(json.dumps(original_hashes, indent=2) + '\n')


def commit():
    subprocess.run(['/usr/local/bin/python', '-c',
                    "import modal; modal.Volume.from_name('hms-bench-run').commit()"], check=True, timeout=180)


repair = {'state': 'running', 'container_id': cid, 'degrees': degrees, 'original_sha256': original_hashes,
          'started': time.time(), 'source_rev': CONFIG['hms_rev'], 'mode': 'pre-export while original function remains active'}
receipt_path = archive / 'repair.json'
record['state'] = 'running'
record['heartbeat'] = time.time()
record['provenance_repair'] = repair | {'receipt': str(receipt_path)}
ledger_path.write_text(json.dumps(record, indent=2) + '\n')
receipt_path.write_text(json.dumps(repair, indent=2) + '\n')
commit()
for path in old_files:
    path.unlink()
    shutil.rmtree(path.with_name(path.name + '.complete'))
    (volume_raw / path.name).unlink()
    shutil.rmtree(volume_raw / (path.name + '.complete'))
commit()

defs = Path('/work/provenance-repair-defs')
(defs / 'hms').mkdir(parents=True)
spec = yaml.safe_load(Path('/opt/vibe/vibe/algorithms/hms/config.yml').read_text())
spec['float']['normalized'][0]['run_groups']['base']['args']['degree'] = degrees
(defs / 'hms/config.yml').write_text(yaml.safe_dump(spec))
env = {**os.environ, 'PATH': '/opt/env/micromamba/bin:' + os.environ['PATH'],
       'PYTHONPATH': '/work/shim', **CONFIG['thread_env']}
args = ['/opt/env/micromamba/bin/python', 'run.py', '--dataset', ds, '--algorithm', 'hms',
        '--local', '--parallelism', '1', '--definitions', str(defs)]
repair['command'] = args
try:
    with (root / 'logs' / ds / 'provenance-repair.log').open('w') as log:
        subprocess.run(args, cwd=local_job, env=env, stdout=log, stderr=subprocess.STDOUT,
                       timeout=3600 * len(degrees), check=True)
    tool = subprocess.run(['/opt/env/micromamba/bin/python', '-c', CONFIG['jobtool'], 'check', ds, 'hms',
                           '/opt/vibe/vibe/algorithms'], cwd=local_job, env=env, capture_output=True,
                          text=True, check=True, timeout=600)
    validation = json.loads(tool.stdout.strip().splitlines()[-1])
    assert validation['done'] and validation['valid'] == validation['expected'] == 180, validation
    for path in raw.glob('*.hdf5'):
        shutil.copy2(path, volume_raw / path.name)
        shutil.copytree(path.with_name(path.name + '.complete'), volume_raw / (path.name + '.complete'), dirs_exist_ok=True)
    merged_raw = Path('/work/merged/results') / ds / '100/hms'
    shutil.rmtree(merged_raw)
    shutil.copytree(raw, merged_raw)
    repair |= {'state': 'done', 'validation': validation,
               'raw_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in raw.glob('*.hdf5')}}
    record['state'] = 'done'
    record['validation'] = validation
    record['end'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
except Exception as exc:
    repair |= {'state': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
    record['state'] = 'pending'
    raise
finally:
    repair['ended'] = time.time()
    repair['elapsed_s'] = repair['ended'] - repair['started']
    repair['cost_note'] = 'Included in the original function heartbeat cost ledger; do not add it again'
    record['provenance_repair'] = repair | {'receipt': str(receipt_path)}
    receipt_path.write_text(json.dumps(repair, indent=2) + '\n')
    ledger_path.write_text(json.dumps(record, indent=2) + '\n')
    commit()
    print(json.dumps(repair), flush=True)
