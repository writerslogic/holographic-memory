import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path.home() / 'hms-sbg-positions'
queries = [json.loads(line) for line in (Path.home() / 'sbg-hms/sbg/queries.txt').read_text().splitlines()]
groups = ('term', 'union', 'intersection', 'intersection_union', 'negated', 'phrase', 'two-phase-critic')
sample = []
for group in groups:
    sample.extend(dict(row, group=group) for row in queries if group in row['tags'])
    start = len(sample) - sum(group in row['tags'] for row in queries)
    sample[start:] = sample[start:start + 16]
commands = ((10, False), (100, True), (0, True))
(root / 'paired-results').mkdir(exist_ok=True)
(root / 'sample.json').write_text(json.dumps(sample, indent=2))
def rss_kib(pid):
    return {line.split(':')[0]: int(line.split()[1]) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith(('VmRSS:', 'VmHWM:'))}

for run, variant in enumerate(('base', 'candidate', 'candidate', 'base', 'base', 'candidate')):
    with (root / f'paired-results/{run}-{variant}.stderr').open('w') as error:
        process = subprocess.Popen(
            ['nice', '-n', '19', 'taskset', '-c', '0-3', str(root / f'driver-{variant}/target/release/posting-query-kernel'), str(root / 'index600')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error, text=True,
            env=dict(os.environ, RAYON_NUM_THREADS='4'))
        assert process.stdout.readline().strip() == 'ready', (variant, process.poll())
        os.sched_setaffinity(process.pid, {0})
        with (Path.home() / 'hms-timing.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            receipt = dict(machine='Intel Core Ultra 9 185H', parent_sha='7db61aa01b19d4ac6e4b5fec15526025a30fd17c', candidate_sha='7e216c6a5afd5a0f518c619fb95b1bb990e08aab', source_sha256=hashlib.sha256((root / variant / 'src/core/documents/postings.rs').read_bytes()).hexdigest(), variant=variant, run=run, timestamp=time.time(), load=os.getloadavg(), query_threads=1, query_cpu=0, rss_kib=rss_kib(process.pid), results=[])
            for k, count in commands:
                for row in sample:
                    request = dict(query=row['query'], k=k, count=count, repeats=100, verify=False)
                    process.stdin.write(json.dumps(request) + '\n')
                    process.stdin.flush()
                    result = json.loads(process.stdout.readline())
                    receipt['results'].append(dict(group=row['group'], **request, matches=result['count'], hash=result['hash'], ns=result['ns']))
            receipt['end_load'] = os.getloadavg()
            (root / f'paired-results/{run}-{variant}.json').write_text(json.dumps(receipt, indent=2))
        process.stdin.close()
        assert process.wait() == 0
        print(run, variant, receipt['load'], receipt['end_load'], flush=True)
