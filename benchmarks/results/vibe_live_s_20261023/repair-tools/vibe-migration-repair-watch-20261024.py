"""Wait for completed dataset arms, then repair migrated builds in their final container."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import modal

BASE = Path('/Volumes/A/.win-vibe-live')
ARTIFACT = Path('/Volumes/A/hms-coordination/public-bench/vibe-results')
spec = importlib.util.spec_from_file_location('vibe_owner', BASE / 'benchmarks/vibe/modal_vibe.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
remote_source = Path('/Volumes/A/hms-coordination/public-bench/vibe-repair-migrated-20261024.py').read_text()
v = modal.Volume.from_name('hms-bench-run')
root = '/vibe/run-s-20261023'
source = '12c80decb094d57ee13ba187a01bf1de8788ca2f'
plan = json.loads(b''.join(v.read_file(root + '/plan.json')))


def repair(dataset):
    started = time.monotonic()
    state = {'dataset': dataset, 'state': 'waiting', 'started': time.time()}
    out = ARTIFACT / f'{dataset}-migration-watch.json'
    out.write_text(json.dumps(state, indent=2) + '\n')
    while time.monotonic() - started < 12 * 3600:
        try:
            record = json.loads(b''.join(v.read_file(root + f'/ledger/{dataset}/hms@12c80decb094.json')))
            if record.get('provenance_repair', {}).get('state') == 'done':
                state |= {'state': 'already_repaired', 'record': record['provenance_repair']}
                break
            standing = json.loads(b''.join(v.read_file(root + f'/standings/{dataset}@12c80decb094.json')))
            records = [json.loads(b''.join(v.read_file(e.path))) for e in v.iterdir(root + f'/ledger/{dataset}')
                       if e.path.endswith('.json')]
            ready = record['state'] == 'done' and standing['container'] == record['container_id'] and all(
                r['state'] in ('done', 'failed') for r in records)
            if ready:
                config = {'dataset': dataset, 'container_id': record['container_id'], 'hms_rev': source,
                          'thread_env': runner.THREAD_ENV, 'mem_gib': plan['datasets'][dataset]['mem_gib'],
                          'jobtool': runner.JOBTOOL, 'standing': runner.STANDING}
                payload = 'CONFIG = ' + repr(config) + '\n' + remote_source
                command = ['modal', 'container', 'exec', record['container_id'], '--',
                           '/opt/env/micromamba/bin/python', '-u', '-c', f'exec(bytes.fromhex({payload.encode().hex()!r}))']
                state |= {'state': 'running', 'container_id': record['container_id'], 'launched': time.time()}
                out.write_text(json.dumps(state, indent=2) + '\n')
                with (ARTIFACT / f'{dataset}-migration-repair.log').open('w') as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=5 * 3600)
                proof = json.loads(b''.join(v.read_file(root + f'/diagnostic-before-provenance-repair/{dataset}/repair.json')))
                state |= {'state': 'finished' if completed.returncode == 0 and proof['state'] == 'done' else 'failed',
                          'returncode': completed.returncode, 'ended': time.time(), 'receipt': proof}
                break
        except (FileNotFoundError, modal.exception.NotFoundError):
            pass
        time.sleep(30)
    else:
        state |= {'state': 'wait_timeout', 'ended': time.time()}
    out.write_text(json.dumps(state, indent=2) + '\n')
    return state


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    for state in pool.map(repair, ['landmark-nomic-768-normalized', 'yahoo-minilm-384-normalized']):
        print(json.dumps(state), flush=True)
