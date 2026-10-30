# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXEMPT = {'dco.yml', 'campaign-policy.yml'}
JOB = re.compile(r'^  ([A-Za-z_][A-Za-z0-9_-]*):\s*$')
GUARD = '    if: ${{ false }}'


def violations(root):
    workflows = root / '.github/workflows'
    ledger = json.loads((root / 'docs/research/technical-audit-20261029/campaign.json').read_text())
    errors = []
    files = sorted([*workflows.glob('*.yml'), *workflows.glob('*.yaml')])
    names = {p.name for p in files}
    if ledger.get('ci_mode') != 'deferred':
        errors.append('campaign ledger must declare ci_mode deferred')
    for name in ledger['deferred_workflows']:
        if name not in names:
            errors.append(f'{name}: missing deferred workflow')
    for name in names - EXEMPT - set(ledger['deferred_workflows']):
        errors.append(f'{name}: workflow missing from deferred ledger')
    for path in files:
        if path.name in EXEMPT:
            continue
        lines = path.read_text().splitlines()
        try:
            start = lines.index('jobs:') + 1
        except ValueError:
            errors.append(f'{path.name}: missing jobs section')
            continue
        headers = [(i, m.group(1)) for i, line in enumerate(lines[start:], start)
                   if (m := JOB.fullmatch(line))]
        for line in lines[start:]:
            if re.match(r'^  [^\s#]', line) and not JOB.fullmatch(line):
                errors.append(f'{path.name}: unrecognized job declaration: {line.strip()}')
        if not headers:
            errors.append(f'{path.name}: no recognized jobs')
        for n, (index, name) in enumerate(headers):
            end = headers[n + 1][0] if n + 1 < len(headers) else len(lines)
            body = lines[index + 1:end]
            conditions = [line for line in body if re.match(r'^    if:', line)]
            if conditions != [GUARD]:
                errors.append(f'{path.name}:{name}: job must have exactly the false deferral guard')
    return errors


def main():
    try:
        errors = violations(ROOT)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors = [f'policy could not inspect campaign: {exc}']
    if errors:
        for error in errors:
            print(f'::error::{error}')
        return 1
    print('::notice::Campaign policy only: expensive validation is deferred; releases and deployments are frozen. This is not a quality gate.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
