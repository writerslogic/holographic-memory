"""Restore the SIFT single-container comparison after preemption changed CPUs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import h5py
import modal

ROOT = Path('/vol/ann-live-20261024/sift-128-euclidean')
SOURCE = '12c80decb094d57ee13ba187a01bf1de8788ca2f'
CONTAINER = 'ta-01M5PDWGY5JF6G29DRNPS32RMR'
VOL = modal.Volume.from_name('hms-ann-live-20261024')
assert os.environ.get('MODAL_TASK_ID') == CONTAINER
wait_until = time.monotonic() + 10 * 3600
while not (ROOT / 'receipt.json').exists():
    if time.monotonic() >= wait_until:
        raise TimeoutError('Original SIFT grid did not finish within repair wait bound')
    time.sleep(60)

manifest = json.loads((ROOT / 'manifest.json').read_text())
receipt = json.loads((ROOT / 'receipt.json').read_text())
assert manifest['source_rev'] == SOURCE
assert all(receipt['outcomes'].get(job) == 'done' for job in ('hms-d32', 'hms-d48', 'hms-d64', 'faiss-ivfpqfs'))
assert (time.time() - manifest['started']) / 3600 * 0.6344 < 6.5, 'Insufficient remaining original cap'
claim = ROOT / 'checkpoints/same-container-d32-repair.claim'
claim.mkdir()
archive = ROOT / 'diagnostic-preempted-d32'
archive.mkdir()
raw = ROOT / 'results/sift-128-euclidean/10/hms'
originals = []
for path in sorted(raw.glob('*.hdf5')):
    with h5py.File(path) as result:
        if 'degree=32,' in str(result.attrs.get('name', '')):
            assert result['neighbors'].shape == (10000, 10)
            originals.append(path)
assert len(originals) == 36
hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in originals}
(archive / 'raw-results.sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
shutil.copy2(ROOT / 'logs/hms-d32.log', archive / 'hms-d32.log')
for path in originals:
    path.rename(archive / path.name)
(ROOT / 'checkpoints/hms-d32.done').rename(archive / 'hms-d32.done')
VOL.commit()

args = ['taskset', '-c', '0', 'python', '-u', 'run.py', '--local', '--dataset', 'sift-128-euclidean',
        '--definitions', str(ROOT / 'defs/hms-d32'), '--runs', '5', '--algorithm', 'hms']
env = {**os.environ, **{k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                      'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'TBB_NUM_THREADS')}}
started = time.time()
attempts = []
with (ROOT / 'logs/hms-d32-repair.log').open('a') as log:
    for attempt in range(1, 4):
        log.write(json.dumps({'command': args, 'attempt': attempt, 'started': time.time(), 'timeout_s': 3600}) + '\n')
        log.flush()
        result = subprocess.run(args, cwd='/work/ann', env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        attempts.append({'attempt': attempt, 'returncode': result.returncode, 'ended': time.time()})
        VOL.commit()
        count = 0
        for path in raw.glob('*.hdf5'):
            with h5py.File(path) as data:
                count += 'degree=32,' in str(data.attrs.get('name', '')) and data['neighbors'].shape == (10000, 10)
        if count == 36 and result.returncode == 0:
            break
        if time.time() - started > 3600:
            break
elapsed = time.time() - started
outcome = {'source_rev': SOURCE, 'container_id': CONTAINER, 'command': args, 'attempts': attempts,
           'started': started, 'ended': time.time(), 'elapsed_s': elapsed,
           'estimated_usd': elapsed / 3600 * 0.6344, 'valid_points': count, 'original_raw_sha256': hashes,
           'original_receipt': receipt, 'state': 'done' if count == 36 and result.returncode == 0 else 'failed',
           'purpose': 'Rerun only completed degree32 points whose original container had a different CPU; original points/logs preserved diagnostically'}
(ROOT / 'same-container-d32-repair.json').write_text(json.dumps(outcome, indent=2) + '\n')
if outcome['state'] == 'done':
    (ROOT / 'checkpoints/hms-d32.done').touch()
VOL.commit()
print(json.dumps(outcome, indent=2), flush=True)
assert outcome['state'] == 'done'
