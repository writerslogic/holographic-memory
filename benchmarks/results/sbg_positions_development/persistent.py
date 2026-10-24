import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
sample = json.loads((root / 'sample.json').read_text())
commands = ((10, False), (100, True), (0, True))
processes = {}
errors = []
try:
    for variant in ('base', 'candidate'):
        index = root / f'index-confirm-{variant}'
        if not index.exists():
            subprocess.run(['cp', '-a', '--reflink=auto', str(Path.home() / 'sbg-hms/dev/idx600'), str(index)], check=True)
        error = (root / f'persistent-{variant}.stderr').open('w')
        errors.append(error)
        process = subprocess.Popen(['nice', '-n', '19', 'taskset', '-c', '12,13', str(root / f'driver-{variant}/target/release/posting-query-kernel'), str(index)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error, text=True, env=dict(os.environ, RAYON_NUM_THREADS='2'))
        processes[variant] = process
        assert process.stdout.readline().strip() == 'ready', (variant, process.poll())
        os.sched_setaffinity(process.pid, {0})
    def query(variant, request):
        process = processes[variant]
        process.stdin.write(json.dumps(request) + '\n')
        process.stdin.flush()
        return json.loads(process.stdout.readline())
    requests = []
    with (Path.home() / 'hms-timing.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        start = dict(timestamp=time.time(), load=os.getloadavg(), machine='Intel Core Ultra 9 185H', parent_sha='7db61aa01b19d4ac6e4b5fec15526025a30fd17c', candidate_sha='7e216c6a5afd5a0f518c619fb95b1bb990e08aab', query_threads=1, query_cpu=0, source_sha256={variant: hashlib.sha256((root / variant / 'src/core/documents/postings.rs').read_bytes()).hexdigest() for variant in processes}, rss_kib={variant: {line.split(':')[0]: int(line.split()[1]) for line in Path(f'/proc/{process.pid}/status').read_text().splitlines() if line.startswith(('VmRSS:', 'VmHWM:'))} for variant, process in processes.items()})
        for k, count in commands:
            for row in sample:
                request = dict(query=row['query'], k=k, count=count, repeats=1, verify=False)
                before = query('base', request)
                after = query('candidate', request)
                assert (before['hash'], before['count']) == (after['hash'], after['count'])
                request['repeats'] = max(5, min(500000, math.ceil(50000000 / max(1, before['ns']))))
                requests.append((row['group'], request, before['hash'], before['count']))
        runs = []
        for run in range(5):
            receipt = dict(run=run, load=os.getloadavg(), timestamp=time.time(), results=[])
            for group, request, expected_hash, expected_count in requests:
                variants = ('base', 'candidate') if run % 2 == 0 else ('candidate', 'base')
                for variant in variants:
                    result = query(variant, request)
                    assert (result['hash'], result['count']) == (expected_hash, expected_count)
                    receipt['results'].append(dict(group=group, variant=variant, **request, matches=result['count'], hash=result['hash'], ns=result['ns']))
            receipt['end_load'] = os.getloadavg()
            runs.append(receipt)
            (root / 'persistent-results.json').write_text(json.dumps(dict(start=start, runs=runs), indent=2))
            print(run, receipt['load'], receipt['end_load'], flush=True)
finally:
    for process in processes.values():
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        assert process.wait() == 0
    for error in errors:
        error.close()
