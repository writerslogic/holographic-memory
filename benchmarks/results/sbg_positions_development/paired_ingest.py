import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path.home() / 'hms-sbg-positions'
(root / 'ingest-results').mkdir(exist_ok=True)
for run, variant in enumerate(('base', 'candidate', 'candidate', 'base', 'base', 'candidate')):
    index = root / f'ingest-{run}-{variant}'
    assert not index.exists(), 'refusing to overwrite index'
    with (root / f'ingest-results/{run}-{variant}.stderr').open('w') as error:
        process = subprocess.Popen(['nice', '-n', '19', 'taskset', '-c', '0-3', str(root / f'driver-{variant}/target/release/posting-query-kernel'), str(index), str(Path.home() / 'sbg-hms/sbg/corpus.json')], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error, text=True, env=dict(os.environ, RAYON_NUM_THREADS='4'))
        assert process.stdout.readline().strip() == 'ready', (variant, process.poll())
        with (Path.home() / 'hms-timing.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            receipt = dict(machine='Intel Core Ultra 9 185H', parent_sha='7db61aa01b19d4ac6e4b5fec15526025a30fd17c', candidate_sha='7e216c6a5afd5a0f518c619fb95b1bb990e08aab', source_sha256=hashlib.sha256((root / variant / 'src/core/documents/postings.rs').read_bytes()).hexdigest(), variant=variant, run=run, timestamp=time.time(), load=os.getloadavg(), threads=4, cpus='0-3')
            process.stdin.write('go\n')
            process.stdin.flush()
            receipt.update(json.loads(process.stdout.readline()))
            receipt['end_load'] = os.getloadavg()
            (root / f'ingest-results/{run}-{variant}.json').write_text(json.dumps(receipt, indent=2))
        process.stdin.close()
        assert process.wait() == 0
        print(run, variant, receipt['ingest_ns'], receipt['load'], receipt['end_load'], flush=True)
