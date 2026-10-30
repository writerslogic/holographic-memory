# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

from campaign_policy import violations


class CampaignPolicyTest(unittest.TestCase):
    def test_job_guards_and_missing_workflows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflows = root / '.github/workflows'
            workflows.mkdir(parents=True)
            audit = root / 'docs/research/technical-audit-20261029'
            audit.mkdir(parents=True)
            (audit / 'campaign.json').write_text(json.dumps({'ci_mode': 'deferred', 'deferred_workflows': ['ci.yml']}))
            workflow = workflows / 'ci.yml'
            workflow.write_text('jobs:\n  build:\n    if: ${{ false }}\n    runs-on: ubuntu-latest\n')
            self.assertEqual(violations(root), [])
            workflow.write_text('jobs:\n  build:\n    if: ${{ false }}\n    runs-on: ubuntu-latest\n  publish:\n    runs-on: ubuntu-latest\n')
            self.assertTrue(any('publish' in error for error in violations(root)))
            workflow.write_text('jobs:\n  build:\n    if: ${{ false }}\n    if: ${{ true }}\n')
            self.assertTrue(violations(root))
            workflow.unlink()
            self.assertTrue(any('missing' in error for error in violations(root)))


if __name__ == '__main__':
    unittest.main()
