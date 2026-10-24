import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import threading
import time

root = Path(__file__).resolve().parent
data = Path(sys.argv[1]).resolve()
index = data / 'index-full'
queries_path = data / 'queries.txt'
reference_path = data / 'COUNT-tantivy-0.26.json'
queries = [json.loads(line) for line in queries_path.read_text().splitlines()]
reference = json.loads(reference_path.read_text())['results']['COUNT']['tantivy-0.26']
reference = {row['query']: row['count'] for row in reference}
assert len(queries) == 962
assert sys.maxsize > 2**32
binary = Path(sys.argv[2]).resolve()
started = time.time()
peak_rss_kib = [0]
samples = []
finished = threading.Event()
with (root / 'full-count-native.stderr').open('w') as error, (root / 'full-count-native.jsonl').open('w') as output:
    process = subprocess.Popen(['nice', '-n', '19', str(binary), str(index)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error, text=True, env=dict(os.environ, RAYON_NUM_THREADS='4'))
    def monitor():
        while not finished.is_set():
            result = subprocess.run(['ps', '-p', str(process.pid), '-o', 'rss='], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                rss = int(result.stdout.strip())
                peak_rss_kib[0] = max(peak_rss_kib[0], rss)
                samples.append(dict(elapsed_s=time.time() - started, rss_kib=rss))
                (root / 'full-count-native-progress.json').write_text(json.dumps(dict(pid=process.pid, samples=samples), indent=2))
            finished.wait(15)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        assert process.stdout.readline().strip() == 'ready', process.poll()
        opened = time.time()
        checked = 0
        for row in queries:
            request = dict(query=row['query'], k=0, count=True, repeats=0, verify=False)
            process.stdin.write(json.dumps(request) + '\n')
            process.stdin.flush()
            result = json.loads(process.stdout.readline())
            assert result['count'] == reference[row['query']], (row, result, reference[row['query']])
            output.write(json.dumps(dict(query=row['query'], matches=result['count'], hash=result['hash'])) + '\n')
            output.flush()
            checked += 1
        process.stdin.close()
        assert process.wait() == 0
    finally:
        finished.set()
        thread.join()
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.wait()
# macOS reports ru_maxrss in bytes. This correctness run establishes no latency claim.
peak_rss_kib[0] = max(peak_rss_kib[0], resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024)
summary = dict(mode='full-count', checked=checked, mismatches=0, machine='Apple M4', architecture=platform.machine(), slot_bits=64, threads=4, nice=19, source_sha='7e216c6a5afd5a0f518c619fb95b1bb990e08aab', correctness_only=True, open_elapsed_s=opened-started, elapsed_s=time.time()-started, peak_rss_kib=peak_rss_kib[0], query_sha256=hashlib.sha256(queries_path.read_bytes()).hexdigest(), reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
(root / 'full-count-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary), flush=True)
