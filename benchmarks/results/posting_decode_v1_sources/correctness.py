# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import subprocess
import sys

root = Path.home() / 'hms-sbg-codec'
full = len(sys.argv) > 1 and sys.argv[1] == 'full'
queries = [json.loads(line) for line in (Path.home() / 'sbg-hms/sbg/queries.txt').read_text().splitlines()]
index = root / ('index-full' if full else 'index600')
if full and not index.exists():
    subprocess.run(['cp', '-a', '--reflink=auto', str(Path.home() / 'sbg-hms/sbg/engines/hms-before/idx'), str(index)], check=True)
reference = json.loads((Path.home() / 'sbg-hms/res/COUNT-tantivy-0.26.json').read_text())['results']['COUNT']['tantivy-0.26']
reference = {row['query']: row['count'] for row in reference}
mode = 'full-count' if full else 'exhaustive600'
with (root / f'{mode}.stderr').open('w') as error, (root / f'{mode}.jsonl').open('w') as output:
    process = subprocess.Popen(['systemd-run', '--user', '--scope', '-q', '-p', 'MemoryMax=14G', '-p', 'MemorySwapMax=0', 'nice', '-n', '19', 'taskset', '-c', '12-15', str(root / 'driver-candidate/target/release/posting-query-kernel'), str(index)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error, text=True, env=dict(os.environ, RAYON_NUM_THREADS='4'))
    assert process.stdout.readline().strip() == 'ready', process.poll()
    commands = ((0, True),) if full else ((10, False), (100, True), (0, True))
    checked = 0
    for k, count in commands:
        for row in queries:
            request = dict(query=row['query'], k=k, count=count, repeats=0, verify=not full)
            process.stdin.write(json.dumps(request) + '\n')
            process.stdin.flush()
            result = json.loads(process.stdout.readline())
            if full:
                assert result['count'] == reference[row['query']], (row, result, reference[row['query']])
            output.write(json.dumps(dict(**request, matches=result['count'], hash=result['hash'])) + '\n')
            output.flush()
            checked += 1
    process.stdin.close()
    assert process.wait() == 0
print(json.dumps(dict(mode=mode, checked=checked, mismatches=0)), flush=True)
