# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy==2.2.6", "lz4==4.4.5"]
# ///
"""Run the frozen checker on closed shards; bind every read to final hashes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

import check_sparse_composition as checker


def verify(attempt):
    freeze = json.loads((attempt / 'freeze.json').read_text())
    deadline = (attempt / 'freeze.json').stat().st_mtime + freeze['deadline_seconds']
    original_open = gzip.open
    observed = {}
    dimensions = freeze['dimensions']

    def closed_open(filename, mode='rb', *args, **kwargs):
        path = Path(filename)
        if 'r' in mode and path.parent == attempt and path.name in [f'd{d}.jsonl.gz' for d in dimensions] + ['operators.jsonl.gz']:
            if path.name == 'operators.jsonl.gz':
                successor = attempt / f'd{dimensions[0]}.jsonl.gz'
            else:
                i = dimensions.index(int(path.name.split('.')[0][1:]))
                successor = attempt / (f'd{dimensions[i+1]}.jsonl.gz' if i+1 < len(dimensions) else 'execution.json')
            while not successor.exists():
                if (attempt / 'execution.json').exists():
                    assert json.loads((attempt / 'execution.json').read_text())['complete'], 'collection failed'
                if time.time() >= deadline:
                    raise TimeoutError('closed-shard deadline')
                time.sleep(5)
            observed[path.name] = checker.sha(path)
        return original_open(filename, mode, *args, **kwargs)

    assert not (attempt / 'independent-check.json').exists(), 'refuse duplicate verification'
    assert checker.sha(Path(checker.__file__)) == freeze['sources']['benchmarks/public/check_sparse_composition.py']
    gzip.open = closed_open
    try:
        checker.main(attempt)
    finally:
        gzip.open = original_open
    execution = json.loads((attempt / 'execution.json').read_text())
    assert execution['complete']
    assert set(observed) == set(execution['outputs'])
    for name, expected in execution['outputs'].items():
        assert observed[name] == expected == checker.sha(attempt / name), name
    receipt = {'complete': True, 'closed_shard_sha256': observed,
               'frozen_checker_sha256': checker.sha(Path(checker.__file__)),
               'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'performance_timings_eligible': False,
               'scheduling': 'one producer plus one checker of already closed immutable dimension files'}
    (attempt / 'validation-lineage.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('attempt', type=Path)
    verify(parser.parse_args().attempt.resolve())
