"""Read-only rescue of reachable VIBE files, with bounded per-file retries."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import threading
import time
import modal

p = argparse.ArgumentParser()
p.add_argument('--data', action='store_true')
p.add_argument('--priority', action='store_true')
a = p.parse_args()
root = Path('/Volumes/A/hms-coordination/public-bench/vibe-results/workspace-disabled-20261024T1435Z')
inventory = json.loads((root / 'volume-inventory.json').read_text())['entries']
entries = [e for e in inventory if e['type'] == '1' and (('/data/' in e['path']) == a.data)]
if a.priority:
    entries = [e for e in entries if e['size'] >= 16]
entries.sort(key=lambda e: (e['size'], e['path']))
volume = modal.Volume.from_name('hms-bench-run')
lock = threading.Lock()
journal = root / ('dataset-downloads.jsonl' if a.data else 'priority-downloads.jsonl' if a.priority else 'artifact-downloads.jsonl')


def backup(entry):
    parts = PurePosixPath(entry['path']).parts
    assert parts[:2] == ('vibe', 'run-s-20261023') and '..' not in parts
    target = root / 'volume' / Path(*parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 3):
        temp = target.with_name(target.name + '.partial')
        try:
            digest = hashlib.sha256()
            if target.exists() and target.stat().st_size == entry['size']:
                with target.open('rb') as f:
                    for chunk in iter(lambda: f.read(8 << 20), b''):
                        digest.update(chunk)
                action = 'retained-existing'
            else:
                size = 0
                with temp.open('wb') as f:
                    for chunk in volume.read_file(entry['path']):
                        size += len(chunk)
                        digest.update(chunk)
                        f.write(chunk)
                assert size == entry['size'], (entry['path'], size, entry['size'])
                temp.replace(target)
                action = 'downloaded'
            record = entry | {'local': str(target), 'sha256': digest.hexdigest(), 'state': 'done', 'attempt': attempt, 'action': action}
            break
        except Exception as exc:
            record = entry | {'state': 'failed', 'attempt': attempt, 'error': f'{type(exc).__name__}: {exc}'[:700]}
            if attempt == 1:
                time.sleep(2)
    record['observed_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with lock:
        with journal.open('a') as f:
            f.write(json.dumps(record) + '\n')
    return record

with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(backup, e) for e in entries]
    done = failures = total_bytes = 0
    for future in as_completed(futures):
        result = future.result()
        done += 1
        failures += result['state'] != 'done'
        total_bytes += result['size'] if result['state'] == 'done' else 0
        if done % 250 == 0 or done == len(entries):
            print(json.dumps({'completed': done, 'expected': len(entries), 'failures': failures, 'bytes': total_bytes}), flush=True)
print(json.dumps({'completed': done, 'expected': len(entries), 'failures': failures, 'bytes': total_bytes, 'data': a.data}), flush=True)
raise SystemExit(bool(failures))
