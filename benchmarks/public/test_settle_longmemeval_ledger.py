# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0"]
# ///
"""Settlement identity, conservative errors, idempotence, and interrupted recovery."""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import settle_longmemeval_ledger as settlement
from check_longmemeval_answers import canonical, digest
from test_check_longmemeval_complete_history import make_complete_fixture
from test_check_longmemeval_full_history import make_full_fixture, update_spend


def make_work(root, high=False):
    fixture = make_complete_fixture(root) if high else make_full_fixture(root)
    work = root / fixture['work_name']
    (work / 'cache').mkdir(parents=True)
    base = root / 'reader_protocol_v1.json'
    preregistration = root / 'preregistration_2026-10.json'
    split = root / 'longmemeval_split.json'
    for path, value in ((base, fixture['base_protocol']), (preregistration, fixture['preregistration']),
                        (split, fixture['split']), (work / 'protocol.json', fixture['protocol'])):
        path.write_bytes(settlement.encoded(value))
    report, ledger = fixture['report'], fixture['ledger']
    report.update(protocol_path=str(work / 'protocol.json'), protocol_sha256=settlement.file_digest(work / 'protocol.json'),
                  artifacts=[{'path': str(path), 'sha256': settlement.file_digest(path)} for path in (base, preregistration, split)])
    for row in report['rows']:
        for stage in ('reader', 'judge'):
            call = row['systems']['matched_gpt54']['arms']['full_history'][stage]['call']
            call_id = digest(f"full-history:{work.name}:{stage}:{row['qid']}")
            key = digest(canonical({'request': call['request'], 'call_id': call_id,
                                    'protocol_sha256': settlement.file_digest(base)}))
            entry = next(entry for entry in ledger['entries'] if entry.get('call_id') == call_id)
            call['cache_key'] = key
            entry.update(cache_key=key, unix_time=1.0)
            (work / 'cache' / (key + '.json')).write_bytes(settlement.encoded(call))
    stale = ledger['entries'][2]
    stale.update(status='reserved', cost_usd=stale['reserved_usd'], usage=None)
    update_spend(fixture)
    save(work, fixture)
    return work, fixture


def save(work, fixture):
    (work / 'spend.json').write_bytes(settlement.encoded(fixture['ledger']))
    fixture['report']['spend_ledger_sha256'] = settlement.file_digest(work / 'spend.json')
    (work / 'answers.json').write_bytes(settlement.encoded(fixture['report']))


class LedgerSettlementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.work, self.fixture = make_work(self.root)

    def test_preview_is_readonly_and_apply_changes_only_verified_own_money_fields(self):
        before_ledger = (self.work / 'spend.json').read_bytes()
        before_report = (self.work / 'answers.json').read_bytes()
        result = settlement.settle(self.work)
        self.assertEqual(result['repairs'], 1)
        self.assertGreater(result['released_usd'], 0)
        self.assertEqual((self.work / 'spend.json').read_bytes(), before_ledger)
        self.assertEqual((self.work / 'answers.json').read_bytes(), before_report)
        applied = settlement.settle(self.work, apply=True)
        ledger = json.loads((self.work / 'spend.json').read_bytes())
        report = json.loads((self.work / 'answers.json').read_bytes())
        expected = copy.deepcopy(json.loads(before_report))
        for field in ('spend_ledger_sha256', 'incremental_spend_usd', 'global_recorded_spend_usd'):
            expected[field] = report[field]
        self.assertEqual(report, expected)
        self.assertEqual(ledger['entries'][0:2], json.loads(before_ledger)['entries'][0:2])
        self.assertEqual(ledger['entries'][2]['status'], 'ok')
        self.assertEqual(report['spend_ledger_sha256'], settlement.file_digest(self.work / 'spend.json'))
        audit = json.loads(Path(applied['audit_path']).read_bytes())
        self.assertEqual(audit['status'], 'committed')
        self.assertEqual(audit['before']['spend_sha256'], settlement.byte_digest(before_ledger))
        self.assertEqual(audit['before']['answers_sha256'], settlement.byte_digest(before_report))
        self.assertEqual(audit['entry_updates'][0]['before']['status'], 'reserved')
        self.assertEqual(audit['entry_updates'][0]['after']['status'], 'ok')

    def test_settlement_is_idempotent_including_audit_count(self):
        settlement.settle(self.work, apply=True)
        before = [(self.work / name).read_bytes() for name in ('spend.json', 'answers.json')]
        result = settlement.settle(self.work, apply=True)
        self.assertFalse(result['changed'])
        self.assertEqual(before, [(self.work / name).read_bytes() for name in ('spend.json', 'answers.json')])
        self.assertEqual(len(list((self.work / 'ledger_settlements').glob('*.json'))), 1)

    def test_verified_cached_cost_keeps_its_exact_float_representation(self):
        row = self.fixture['report']['rows'][0]
        call = row['systems']['matched_gpt54']['arms']['full_history']['judge']['call']
        call['cost_usd'] = math.nextafter(call['cost_usd'], math.inf)
        self.fixture['ledger']['entries'][3]['cost_usd'] = call['cost_usd']
        (self.work / 'cache' / (call['cache_key'] + '.json')).write_bytes(settlement.encoded(call))
        update_spend(self.fixture)
        save(self.work, self.fixture)
        result = settlement.settle(self.work, apply=True)
        self.assertEqual(result['repairs'], 1)
        settled = json.loads((self.work / 'spend.json').read_bytes())
        self.assertEqual(settled['entries'][3]['cost_usd'], call['cost_usd'])

    def test_interruption_after_ledger_replace_recovers_report_from_durable_audit(self):
        original = settlement.write_atomic
        report_before = (self.work / 'answers.json').read_bytes()

        def fail_report(path, content):
            if path.name == 'answers.json':
                raise OSError('simulated report replacement failure')
            original(path, content)

        with patch.object(settlement, 'write_atomic', side_effect=fail_report):
            with self.assertRaisesRegex(OSError, 'simulated'):
                settlement.settle(self.work, apply=True)
        pending = next((self.work / 'ledger_settlements').glob('*.json'))
        self.assertEqual(json.loads(pending.read_bytes())['status'], 'prepared')
        self.assertEqual((self.work / 'answers.json').read_bytes(), report_before)
        self.assertEqual(json.loads((self.work / 'spend.json').read_bytes())['entries'][2]['status'], 'ok')
        result = settlement.settle(self.work, apply=True)
        self.assertTrue(result['recovered'])
        self.assertEqual(json.loads(pending.read_bytes())['status'], 'committed')
        self.assertFalse(settlement.settle(self.work)['changed'])

    def test_missing_cache_preserves_all_charges_and_refuses_settlement(self):
        entry = self.fixture['ledger']['entries'][2]
        (self.work / 'cache' / (entry['cache_key'] + '.json')).unlink()
        before = (self.work / 'spend.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'conservative charge preserved'):
            settlement.settle(self.work, apply=True)
        self.assertEqual((self.work / 'spend.json').read_bytes(), before)

    def test_error_result_keeps_full_conservative_charge_when_status_is_settled(self):
        row = self.fixture['report']['rows'][0]
        arm = row['systems']['matched_gpt54']['arms']['full_history']
        call = arm['reader']['call']
        entry = self.fixture['ledger']['entries'][2]
        call.update(status='error', raw_text='', response=None, usage=None, cost_usd=entry['reserved_usd'],
                    cost_is_conservative_reservation=True)
        arm['reader'].update(status='error', raw_text='', parsed=None)
        judge_id = digest(f"full-history:{self.work.name}:judge:{row['qid']}")
        arm['judge'] = {'status': 'reader_failure', 'raw_text': '', 'score': False}
        self.fixture['ledger']['entries'] = [entry for entry in self.fixture['ledger']['entries'] if entry.get('call_id') != judge_id]
        (self.work / 'cache' / (call['cache_key'] + '.json')).write_bytes(settlement.encoded(call))
        update_spend(self.fixture)
        save(self.work, self.fixture)
        result = settlement.settle(self.work, apply=True)
        settled = json.loads((self.work / 'spend.json').read_bytes())['entries'][2]
        self.assertEqual(settled['status'], 'error')
        self.assertEqual(settled['cost_usd'], entry['reserved_usd'])
        self.assertEqual(result['released_usd'], 0.0)

    def test_high_reader_and_medium_judge_profile_settles_without_request_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            work, fixture = make_work(Path(temporary), high=True)
            before = copy.deepcopy(fixture['report']['rows'][0]['systems']['matched_gpt54']['arms']['full_history']['reader']['call'])
            settlement.settle(work, apply=True)
            after = json.loads((work / 'answers.json').read_bytes())['rows'][0]['systems']['matched_gpt54']['arms']['full_history']['reader']['call']
            self.assertEqual(before, after)
            self.assertEqual(after['request']['reasoning'], {'effort': 'high'})

    def test_cache_request_usage_or_record_inconsistency_rejects_without_writes(self):
        key = self.fixture['ledger']['entries'][2]['cache_key']
        cache = self.work / 'cache' / (key + '.json')
        cached = cache.read_bytes()
        before = (self.work / 'spend.json').read_bytes()
        for field in ('request_sha256', 'cost_usd', 'raw_text'):
            with self.subTest(field=field):
                altered = json.loads(cached)
                altered[field] = 'altered' if field != 'cost_usd' else altered[field] + 0.01
                cache.write_bytes(settlement.encoded(altered))
                with self.assertRaisesRegex(ValueError, 'cached API record differs'):
                    settlement.settle(self.work, apply=True)
                self.assertEqual((self.work / 'spend.json').read_bytes(), before)
        cache.write_bytes(cached)

    def test_duplicate_calls_and_nonreservation_conflicts_are_not_silently_reconciled(self):
        for corruption in ('duplicate', 'conflicting_settled'):
            with self.subTest(corruption=corruption):
                fixture = copy.deepcopy(self.fixture)
                entry = fixture['ledger']['entries'][2]
                if corruption == 'duplicate':
                    extra = copy.deepcopy(entry)
                    extra['cache_key'] = 'different-cache-same-call'
                    fixture['ledger']['entries'].append(extra)
                else:
                    entry['status'] = 'ok'
                update_spend(fixture)
                save(self.work, fixture)
                with self.assertRaisesRegex(ValueError, 'duplicate|conflicts'):
                    settlement.settle(self.work, apply=True)
        save(self.work, self.fixture)

    def test_incomplete_work_and_concurrently_changed_targets_are_rejected(self):
        plan = settlement.build_plan(self.work)
        report = copy.deepcopy(self.fixture['report'])
        report['complete'] = False
        (self.work / 'answers.json').write_bytes(settlement.encoded(report))
        with self.assertRaisesRegex(ValueError, 'completed dev-only'):
            settlement.settle(self.work, apply=True)
        before = (self.work / 'spend.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'target changed concurrently'):
            settlement.apply_plan(self.work, plan)
        self.assertEqual((self.work / 'spend.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
