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
archive_base = root / 'diagnostic-before-provenance-repair' / ds
archive = archive_base / f'attempt-{time.time_ns()}'
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
          'started': time.time(), 'source_rev': CONFIG['hms_rev'], 'archive': str(archive),
          'mode': 'pre-export while original function remains active'}
receipt_path = archive_base / 'repair.json'
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
(defs / 'hms').mkdir(parents=True, exist_ok=True)
spec = yaml.safe_load(Path('/opt/vibe/vibe/algorithms/hms/config.yml').read_text())
spec['float']['normalized'][0]['run_groups']['base']['args']['degree'] = degrees
(defs / 'hms/config.yml').write_text(yaml.safe_dump(spec))
env = {**os.environ, 'PATH': '/opt/env/micromamba/bin:' + os.environ['PATH'],
       'PYTHONPATH': '/work/shim', **CONFIG['thread_env']}
args = ['/opt/env/micromamba/bin/python', 'run.py', '--dataset', ds, '--algorithm', 'hms',
        '--local', '--parallelism', '1', '--definitions', str(defs)]
repair['command'] = args
try:
    with (archive / 'repair.log').open('w') as log:
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
    (archive / 'repair.json').write_text(json.dumps(repair, indent=2) + '\n')
    ledger_path.write_text(json.dumps(record, indent=2) + '\n')
    commit()
    print(json.dumps(repair), flush=True)
