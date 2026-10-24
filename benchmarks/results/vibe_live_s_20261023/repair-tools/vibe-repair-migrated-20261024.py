"""Run inside a completed VIBE dataset container, with CONFIG supplied by the owner."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import modal
import yaml

ds, cid = CONFIG['dataset'], CONFIG['container_id']
assert ds in ('landmark-nomic-768-normalized', 'yahoo-minilm-384-normalized')
assert os.environ.get('MODAL_TASK_ID') == cid
root = Path('/vol/vibe/run-s-20261023')
job = 'hms@12c80decb094'
ledger_path = root / 'ledger' / ds / f'{job}.json'
record = json.loads(ledger_path.read_text())
assert record['hms_rev'] == CONFIG['hms_rev'] and record['container_id'] == cid and record['state'] == 'done'
for p in ledger_path.parent.glob('*.json'):
    r = json.loads(p.read_text())
    assert r['state'] in ('done', 'failed'), f'Another arm remains active: {p}'
standing_path = root / 'standings' / f'{ds}@12c80decb094.json'
standing = json.loads(standing_path.read_text())
assert standing['container'] == cid
start = datetime.datetime.fromisoformat(record['start'].replace('Z', '+00:00')).timestamp()
local_job = Path('/work/jobs') / job
raw = local_job / 'results' / ds / '100/hms'
volume_raw = root / 'results' / ds / job / 'results' / ds / '100/hms'
old_files = [p for p in raw.glob('*.hdf5') if any(
    m.stat().st_mtime < start for m in p.with_name(p.name + '.complete').iterdir())]
assert old_files, 'No migrated groups remain; do not repeat completed work'
degrees = []
for p in old_files:
    match = re.fullmatch(r'normalized_(32|48|64|96)_500_8_true\.hdf5', p.name)
    assert match, p.name
    degrees.append(int(match.group(1)))
archive = root / 'diagnostic-before-provenance-repair' / ds
archive.mkdir(parents=True)
archive.joinpath('ledger.json').write_text(json.dumps(record, indent=2) + '\n')
shutil.copy2(standing_path, archive / 'standing.json')
shutil.copy2(standing_path.with_suffix('.summary.parquet'), archive / 'summary.parquet')
original_hashes = {}
for p in old_files:
    original_hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    shutil.copy2(p, archive / p.name)
    shutil.copytree(p.with_name(p.name + '.complete'), archive / (p.name + '.complete'))
archive.joinpath('raw-results.sha256.json').write_text(json.dumps(original_hashes, indent=2) + '\n')
v = modal.Volume.from_name('hms-bench-run')
v.commit()
for p in old_files:
    p.unlink()
    shutil.rmtree(p.with_name(p.name + '.complete'))
    (volume_raw / p.name).unlink()
    shutil.rmtree(volume_raw / (p.name + '.complete'))
v.commit()

defs = Path('/work/provenance-repair-defs')
(defs / 'hms').mkdir(parents=True)
spec = yaml.safe_load(Path('/opt/vibe/vibe/algorithms/hms/config.yml').read_text())
spec['float']['normalized'][0]['run_groups']['base']['args']['degree'] = degrees
(defs / 'hms/config.yml').write_text(yaml.safe_dump(spec))
env = {**os.environ, 'PATH': '/opt/env/micromamba/bin:' + os.environ['PATH'],
       'PYTHONPATH': '/work/shim', **CONFIG['thread_env']}
args = ['/opt/env/micromamba/bin/python', 'run.py', '--dataset', ds, '--algorithm', 'hms',
        '--local', '--parallelism', '1', '--definitions', str(defs)]
repair = {'state': 'running', 'container_id': cid, 'degrees': degrees, 'command': args,
          'original_sha256': original_hashes, 'started': time.time(), 'source_rev': CONFIG['hms_rev']}
receipt_path = archive / 'repair.json'
receipt_path.write_text(json.dumps(repair, indent=2) + '\n')
v.commit()
try:
    with (root / 'logs' / ds / 'provenance-repair.log').open('w') as log:
        result = subprocess.run(args, cwd=local_job, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=3600 * len(degrees), check=True)
    tool = subprocess.run(['/opt/env/micromamba/bin/python', '-c', CONFIG['jobtool'], 'check', ds, 'hms',
                           '/opt/vibe/vibe/algorithms'], cwd=local_job, env=env, capture_output=True,
                          text=True, check=True, timeout=600)
    validation = json.loads(tool.stdout.strip().splitlines()[-1])
    assert validation['done'] and validation['valid'] == validation['expected'] == 180, validation
    for p in raw.glob('*.hdf5'):
        shutil.copy2(p, volume_raw / p.name)
        shutil.copytree(p.with_name(p.name + '.complete'), volume_raw / (p.name + '.complete'), dirs_exist_ok=True)
    merged = Path('/work/merged')
    shutil.rmtree(merged / 'results' / ds / '100/hms')
    shutil.copytree(raw, merged / 'results' / ds / '100/hms')
    with (root / 'logs' / ds / 'provenance-repair-export.log').open('w') as log:
        subprocess.run(['/opt/env/micromamba/envs/plot/bin/python', '/opt/vibe/export_results.py',
                        '--results', str(merged / 'results'), '--data', str(merged / 'data'),
                        '--output', str(merged / 'repaired-out'), '--skip-pca-mahalanobis'],
                       cwd='/opt/vibe', stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
    summary = merged / 'repaired-out/summary.parquet'
    exported = subprocess.run(['/opt/env/micromamba/envs/plot/bin/python', '-c', CONFIG['standing'], str(summary)],
                              capture_output=True, text=True, check=True)
    standing['best_qps_at_recall'] = json.loads(exported.stdout)
    standing['generated'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    standing['provenance_repair'] = str(receipt_path)
    shutil.copy2(summary, standing_path.with_suffix('.summary.parquet'))
    standing_path.write_text(json.dumps(standing, indent=2) + '\n')
    repair |= {'state': 'done', 'validation': validation,
               'raw_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.glob('*.hdf5')}}
    record['provenance_repair'] = repair | {'receipt': str(receipt_path)}
    ledger_path.write_text(json.dumps(record, indent=2) + '\n')
except Exception as exc:
    repair |= {'state': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
    raise
finally:
    repair['ended'] = time.time()
    repair['elapsed_s'] = repair['ended'] - repair['started']
    repair['estimated_cpu_memory_usd'] = repair['elapsed_s'] / 3600 * (2 * 0.04716 + CONFIG['mem_gib'] * 0.007992)
    receipt_path.write_text(json.dumps(repair, indent=2) + '\n')
    if repair['state'] == 'done':
        record['provenance_repair'] = repair | {'receipt': str(receipt_path)}
        ledger_path.write_text(json.dumps(record, indent=2) + '\n')
    v.commit()
    print(json.dumps(repair), flush=True)
