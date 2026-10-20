# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Four separate corruptions must fail the independent packet validator."""
import argparse
import copy
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, default=Path('benchmarks/results/evidence_budget_dev.json'))
    p.add_argument('--reproduction', type=Path, default=Path('target/conversation-geometry-validation/operand-repeat.json'))
    p.add_argument('--binary', type=Path, required=True)
    args = p.parse_args()
    work = Path('target/conversation-geometry-validation')
    report = json.loads(args.report.read_text())
    cases = {}
    for kind in ['packet', 'budget', 'oracle', 'membership']:
        mutated = copy.deepcopy(report)
        row = next(r for r in mutated['rows'] if r['scored'])
        if kind == 'packet':
            row['arms']['operand']['packet_sha256'] = '0' * 64
        elif kind == 'budget':
            row['budget'] += 1
        elif kind == 'oracle':
            row['oracle']['unanchored_source_budget'] = not row['oracle']['unanchored_source_budget']
        else:
            row['arms']['operand']['selected'][0] = 'unobserved-turn'
        path = work / f'operand-corrupted-{kind}.json'
        path.write_text(json.dumps(mutated, allow_nan=False) + '\n')
        command = ['uv', 'run', '--no-project', 'python', 'benchmarks/public/check_operand_report.py', str(path),
                   '--reproduction', str(args.reproduction), '--binary', str(args.binary), '--output', str(work / f'operand-invalid-{kind}-summary.json')]
        with (work / f'operand-reject-{kind}.log').open('w') as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        assert proc.returncode != 0, f'{kind} tampering accepted'
        message = (work / f'operand-reject-{kind}.log').read_text()
        expected = {'packet': 'corrupted packet hash', 'budget': "row['budget']", 'oracle': "row['oracle']", 'membership': 'unobserved source membership'}[kind]
        assert expected in message, f'{kind} rejected for an unrelated reason'
        cases[kind] = {'exit_code': proc.returncode, 'rejected': True, 'reason_checked': expected, 'log': str(work / f'operand-reject-{kind}.log')}
    (work / 'operand-tamper-checks.json').write_text(json.dumps(cases, indent=2) + '\n')
    print('4 tamper checks rejected')


if __name__ == '__main__':
    main()
