# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0", "tokenizers==0.23.3", "openai==3.27.0", "python-dotenv==1.2.4"]
# ///
"""Synthetic source-only production adapter checks; no native models or API calls."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import longmemeval_production as production
from check_longmemeval_answers import canonical, digest, recompute_scores


def raw_question():
    return {'question_id': 'private-type-abs', 'question_type': 'knowledge-update', 'answer': 'GOLD_SENTINEL',
            'question': 'Which material did I confirm?', 'question_date': '2024/03/15 (Fri) 12:00',
            'haystack_dates': ['2024/03/14 (Thu) 09:00', '2024/03/15 (Fri) 10:00'],
            'haystack_session_ids': ['answer_label', 'noans_label'],
            'haystack_sessions': [[{'role': 'user', 'content': 'I am choosing a coating.', 'has_answer': True},
                                   {'role': 'assistant', 'content': '  Consider cerulean enamel.\n'},
                                   {'role': 'user', 'content': 'Yes, that one.'},
                                   {'role': 'assistant', 'content': 'Your cerulean enamel selection is recorded.'}],
                                  [{'role': 'assistant', 'content': 'A separate orphan source.'}]]}


def packet(row, selected=None, anchor=3):
    originals = list(row['original_sources'].values())
    selected = list(range(4)) if selected is None else selected
    sources = []
    for index in selected:
        original = originals[index]
        sources.append({'recordId': production.native_record_id(original), 'sourceId': original['source_id'],
                        'sessionId': original['session_id'], 'version': '1', 'turnIndex': original['turn_index'],
                        'role': original['role'], 'observationTime': original['native_observation_time'],
                        'text': original['text'], 'premiseRecordIds': production.native_premises(row, original)})
    original = originals[anchor]
    anchor_value = {'recordId': production.native_record_id(original), 'sourceId': original['source_id'],
                    'sessionId': original['session_id'], 'version': '1', 'turnIndex': original['turn_index'],
                    'score': 1 / 61}
    return {'sources': sources, 'anchors': [anchor_value], 'dropped': [],
            'serializedBytes': len(json.dumps(sources, ensure_ascii=False, separators=(',', ':')).encode())}


def resize(evidence):
    evidence['serializedBytes'] = len(json.dumps(evidence['sources'], ensure_ascii=False, separators=(',', ':')).encode())


class FakeBridge:
    def __init__(self, row, failure=None):
        self.row, self.failure = row, failure
        self.requests = []
        self.sources = self.sessions = 0

    def exchange(self, request):
        self.requests.append(copy.deepcopy(request))
        command = request['command']
        response = {'status': 'ok', 'command': command, 'epoch': 2, 'sourceCount': self.sources,
                    'chunkCount': self.sources, 'sessionCount': self.sessions, 'ingestAttempts': self.sessions,
                    'recallAttempts': 0, 'modelUsage': {'exactCountersAvailable': False, 'remoteApiCalls': 0,
                                                     'localComputeUsd': None}}
        if command == 'reset':
            self.sources = self.sessions = 0
            response.update(sourceCount=0, chunkCount=0, sessionCount=0)
        elif command == 'ingest':
            self.sources += len(request['conversation']['turns'])
            self.sessions += 1
            response.update(sourceCount=self.sources, chunkCount=self.sources, sessionCount=self.sessions,
                            ingestAttempts=self.sessions)
        else:
            response.update(evidence=packet(self.row), recallAttempts=1)
            if self.failure == 'status':
                response.update(status='error', error={'code': 'source_error'})
            elif self.failure == 'epoch':
                response['epoch'] = 1
            elif self.failure == 'transport':
                raise TimeoutError('synthetic timeout')
        return {'request': copy.deepcopy(request), 'request_jsonl': canonical(request) + '\n',
                'request_sha256': digest(canonical(request)), 'response': response,
                'response_jsonl': json.dumps(response, ensure_ascii=False) + '\n'}


class ProductionSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.raw = raw_question()
        self.row = production.source_row('dev-report-id', self.raw)

    def test_native_input_never_uses_gold_type_qid_or_gold_bearing_session_labels(self):
        original = copy.deepcopy(self.raw)
        altered = copy.deepcopy(self.raw)
        altered.update(question_id='different-report-id', question_type='other-type', answer='OTHER_GOLD')
        altered['haystack_session_ids'] = ['completely-different-label', 'answer-noans']
        for session in altered['haystack_sessions']:
            for turn in session:
                turn['has_answer'] = not turn.get('has_answer', False)
        a, b = production.export_history(original), production.export_history(altered)
        self.assertEqual(a, b)
        native = self.row['native_input']
        self.assertEqual(set(native), {'conversations', 'query', 'options'})
        self.assertEqual(native['options'], {'roles': ['user', 'assistant'], 'k': 64,
                                          'candidateLimit': 256, 'maxSources': 128, 'maxBytes': 65536})
        text = canonical(native)
        for forbidden in ('GOLD_SENTINEL', 'private-type-abs', 'knowledge-update', 'answer_label', 'has_answer'):
            self.assertNotIn(forbidden, text)
        self.assertEqual(native['query'], f"As of {self.raw['question_date']}, {self.raw['question']}")
        self.assertEqual(self.raw, original)

    def test_complete_confirmation_sources_and_assistant_context_are_preserved(self):
        evidence = packet(self.row)
        originals = production.native_sources(self.row, evidence)
        self.assertEqual([source['text'] for source in originals],
                         [turn['content'] for turn in self.raw['haystack_sessions'][0]])
        self.assertEqual([source['role'] for source in originals], ['user', 'assistant', 'user', 'assistant'])
        arm = production.arm_for_native(self.row, {'status': 'ok', 'exchanges': [{'response': {'evidence': evidence}}]},
                                       {'reader_prompt': 'GENERIC\n'})
        payload = json.loads(arm['reader_input'][len('GENERIC\n'):])
        self.assertEqual(set(payload), {'question', 'question_date', 'evidence'})
        self.assertEqual(payload['question'], self.raw['question'])
        self.assertEqual(len(payload['evidence']), 4)
        self.assertEqual(payload['evidence'][1]['text'], '  Consider cerulean enamel.\n')

    def test_corrupt_bytes_missing_premise_wrong_session_and_forged_links_are_rejected(self):
        for corruption in ('text', 'missing_premise', 'scope', 'links', 'record', 'duplicate'):
            with self.subTest(corruption=corruption):
                evidence = packet(self.row)
                if corruption == 'text':
                    evidence['sources'][1]['text'] = 'paraphrased'
                elif corruption == 'missing_premise':
                    evidence['sources'].pop(1)
                elif corruption == 'scope':
                    evidence['sources'][0]['sessionId'] = 'different-session'
                elif corruption == 'links':
                    evidence['sources'][-1]['premiseRecordIds'] = []
                elif corruption == 'record':
                    evidence['sources'][0]['recordId'] = 'forged-record'
                else:
                    evidence['sources'].append(copy.deepcopy(evidence['sources'][0]))
                resize(evidence)
                with self.assertRaises(ValueError):
                    production.native_sources(self.row, evidence)

    def test_source_byte_boundary_is_a_native_byte_cap_and_never_truncates(self):
        base = packet(self.row)
        size = base['serializedBytes']
        for delta in (-1, 0, 1):
            with self.subTest(delta=delta):
                options = dict(production.OPTIONS, maxBytes=size + delta)
                with patch.object(production, 'OPTIONS', options):
                    if delta < 0:
                        with self.assertRaisesRegex(ValueError, 'byte count'):
                            production.native_sources(self.row, base)
                    else:
                        result = production.native_sources(self.row, base)
                        self.assertEqual(result[1]['text'], self.raw['haystack_sessions'][0][1]['content'])

    def test_native_transport_failure_or_wrong_epoch_is_terminal_without_resampling(self):
        for failure in ('status', 'epoch', 'transport'):
            with self.subTest(failure=failure):
                path = self.work / (failure + '.json')
                bridge = FakeBridge(self.row, failure)
                result = production.question_native(bridge, self.row, path, {'path': 'init', 'sha256': 'x'})
                self.assertEqual(result['status'], 'error')
                self.assertEqual(sum(request['command'] == 'recall' for request in bridge.requests), 1)
                arm = production.arm_for_native(self.row, result, {'reader_prompt': 'GENERIC\n'})
                self.assertEqual(arm['reader']['status'], 'selection_failure')
                self.assertEqual(arm['retained_sources'], [])
                self.assertEqual(production.question_native(bridge, self.row, path, {}), result)
                self.assertEqual(sum(request['command'] == 'recall' for request in bridge.requests), 1)
                if failure == 'status':
                    self.assertEqual(result['exchanges'][-1]['response']['status'], 'error')

    def test_native_success_journals_raw_requests_epoch_and_all_local_stages(self):
        bridge = FakeBridge(self.row)
        result = production.question_native(bridge, self.row, self.work / 'ok.json', {'path': 'init', 'sha256': 'x'})
        self.assertEqual(result['status'], 'ok')
        self.assertEqual([item['request']['command'] for item in result['exchanges']], ['reset', 'ingest', 'ingest', 'recall'])
        for item in result['exchanges']:
            self.assertEqual(json.loads(item['request_jsonl']), item['request'])
            self.assertEqual(json.loads(item['response_jsonl']), item['response'])
            self.assertIsNone(item['response']['modelUsage']['localComputeUsd'])
            self.assertFalse(item['response']['modelUsage']['exactCountersAvailable'])
        self.assertGreaterEqual(result['elapsed_seconds'], 0)

    def test_all_100_native_failures_remain_in_denominator_without_acceptance_target(self):
        arm = production.arm_for_native(self.row, {'status': 'error'}, {'reader_prompt': 'GENERIC\n'})
        arm['judge'] = {'status': 'reader_failure', 'raw_text': '', 'score': False}
        arm['scores'] = recompute_scores(arm)
        rows = [{'qid': str(index), 'native_status': 'error', 'systems': {production.SYSTEM: {'arms': {production.ARM: copy.deepcopy(arm)}}}}
                for index in range(100)]
        result = production.summarize({'rows': rows})
        self.assertEqual((result['n'], result['correct'], result['answer_accuracy']), (100, 0, 0))
        self.assertNotIn('acceptance_met', result)
        self.assertNotIn('acceptance_correct', result)
        with self.assertRaises(ValueError):
            production.summarize({'rows': rows[:99]})

    def test_pending_readers_prevent_reference_loading_or_client_construction(self):
        report = {'framing_frozen': True, 'rows': [{'systems': {production.SYSTEM: {'arms': {production.ARM: {
            'reader': {'status': 'pending'}}}}}}]}
        with patch.object(production, 'judge_references') as references, patch.object(production.paid.reader, 'client') as client:
            with self.assertRaisesRegex(ValueError, 'finish all readers'):
                production.run_stage(report, self.work, 'judge')
            references.assert_not_called()
            client.assert_not_called()

    def test_provider_provenance_never_serializes_credentials_path_or_query(self):
        with patch.object(production, 'version', return_value='3.27.0'):
            result = production.provider_provenance(SimpleNamespace(base_url='https://user:secret@proxy.example/private?key=hidden'))
        self.assertEqual(result, {'scheme': 'https', 'hostname': 'proxy.example', 'sdk_version': '3.27.0'})
        self.assertNotIn('secret', canonical(result))

    def test_exact_endpoint_boundary_fails_without_clipping_or_paid_generation(self):
        for count in (32255, 32256, 32257):
            with self.subTest(input_tokens=count):
                work = self.work / str(count)
                work.mkdir()
                protocol = {'reader_prompt': 'GENERIC\n'}
                production.paid.reader.write(work / 'protocol.json', protocol)
                success = {'status': 'ok', 'exchanges': [{'response': {'evidence': packet(self.row)}}]}
                rows = []
                for index in range(100):
                    arm = production.arm_for_native(self.row, success if index == 0 else {'status': 'error'}, protocol)
                    rows.append({'qid': f'dev{index}', 'native_status': 'ok' if index == 0 else 'error',
                                 'systems': {production.SYSTEM: {'arms': {production.ARM: arm}}}})
                report = {'framing_frozen': False, 'rows': rows, 'artifacts': []}
                original = copy.deepcopy(rows[0]['systems'][production.SYSTEM]['arms'][production.ARM]['source_groups'])

                def exact(client, prompt, path):
                    request = {'model': 'gpt-6.1-sol', 'input': prompt}
                    key = digest(canonical({'namespace': 'sol61-input-count-v1', 'request': request}))
                    record = {'cache_key': key, 'request': request, 'request_sha256': digest(canonical(request)),
                              'status': 'ok', 'response': {'input_tokens': count}}
                    production.paid.reader.write(path / 'input_counts' / (key + '.json'), record)
                    return record

                with patch.object(production, 'validate_inputs'), patch.object(production, 'save'), patch.object(production, 'progress'), \
                        patch.object(production, 'provider_provenance', return_value={'scheme': 'https', 'hostname': 'synthetic'}), \
                        patch.object(production.paid, 'refresh_ledger'), \
                        patch.object(production.paid.reader, 'client', return_value=object()), \
                        patch.object(production.paid, 'exact_input_count', side_effect=exact) as endpoint, \
                        patch.object(production.paid, 'call_with_transport_retries') as generation:
                    production.count_inputs(report, work)
                    endpoint.assert_called_once()
                    generation.assert_not_called()
                arm = rows[0]['systems'][production.SYSTEM]['arms'][production.ARM]
                self.assertEqual(arm['source_groups'], original)
                self.assertEqual(arm['input_tokens'], count)
                self.assertEqual(arm['reader']['status'], 'selection_failure' if count > 32256 else 'pending')
                self.assertEqual(production.summarize(report)['n'], 100)
                self.assertTrue(report['framing_frozen'])

    def test_interrupted_native_query_is_not_replayed_or_assigned_zero_duration(self):
        path = self.work / 'interrupted.json'
        journal = {'schema_version': 1, 'qid': self.row['qid'], 'status': 'running', 'initialization': {},
                   'source_input_sha256': digest(canonical(self.row['native_input'])), 'exchanges': []}
        production.paid.reader.write(path, journal)
        bridge = FakeBridge(self.row)
        result = production.question_native(bridge, self.row, path, {})
        self.assertEqual(result['error_type'], 'InterruptedNativeAttempt')
        self.assertIsNone(result['elapsed_seconds'])
        self.assertTrue(result['timing_unknown'])
        self.assertEqual(bridge.requests, [])

    def test_clean_protocol_has_one_global_profile_and_no_dev_prior_dependencies(self):
        base = {'reader_prompt': 'OFFICIAL GENERIC PROMPT', 'judge_source': {'templates': {}}}
        preregistration = {'spend_caps_usd': {'total': 250, 'reader_judge_dev': 120, 'heldout400': 0}}
        with patch.object(production, 'check_protocol') as official_check:
            protocol = production.make_protocol(base, preregistration)
        official_check.assert_called_once_with(base, preregistration)
        self.assertEqual(protocol['reader_prompt'], base['reader_prompt'])
        self.assertEqual(protocol['native_options'], production.OPTIONS)
        self.assertEqual(protocol['native_profile']['total_subquery_weight'], 1)
        for forbidden in ('prior_snapshots', 'acceptance_correct', 'local_proxy_cap', 'additional_reasoning_rules'):
            self.assertNotIn(forbidden, protocol)
        self.assertEqual(protocol['new_spend_cap_usd'], 14)

    def test_bridge_eof_allows_child_cleanup_and_closes_all_handles(self):
        marker = self.work / 'child-cleaned-up'
        script = 'import pathlib,sys\ntry:\n sys.stdin.read()\nfinally:\n pathlib.Path(sys.argv[1]).write_text("cleaned")\n'
        bridge = production.NativeBridge([sys.executable, '-u', '-c', script, str(marker)], self.work / 'child.stderr')
        self.addCleanup(bridge.close)
        with patch.object(bridge.process, 'kill', wraps=bridge.process.kill) as kill:
            bridge.close()
            kill.assert_not_called()
        self.assertEqual(marker.read_text(), 'cleaned')
        self.assertEqual(bridge.process.returncode, 0)
        self.assertTrue(bridge.process.stdin.closed)
        self.assertTrue(bridge.process.stdout.closed)
        self.assertTrue(bridge.stderr.closed)
        bridge.close()

    def test_bridge_kills_and_reaps_only_a_child_that_ignores_eof(self):
        script = ('import json,sys,time\n'
                  'for line in sys.stdin:\n'
                  ' request=json.loads(line)\n'
                  ' print(json.dumps({"command":request["command"],"status":"ok"}),flush=True)\n'
                  'time.sleep(60)\n')
        bridge = production.NativeBridge([sys.executable, '-u', '-c', script], self.work / 'stubborn.stderr')
        self.addCleanup(bridge.close)
        self.assertEqual(bridge.exchange({'command': 'initialize'})['response']['status'], 'ok')
        with patch.object(bridge.process, 'kill', wraps=bridge.process.kill) as kill:
            bridge.close()
            kill.assert_called_once_with()
        self.assertLess(bridge.process.returncode, 0)
        self.assertTrue(bridge.process.stdin.closed)
        self.assertTrue(bridge.process.stdout.closed)
        self.assertTrue(bridge.stderr.closed)


if __name__ == '__main__':
    unittest.main()
