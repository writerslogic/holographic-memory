# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0", "tokenizers==0.23.3", "openai==3.27.0", "python-dotenv==1.2.4"]
# ///
"""Independent native provenance and accounting tampering checks; no model execution."""

import copy
import math
import tempfile
import unittest
from pathlib import Path

import tiktoken

from check_longmemeval_answers import canonical, digest, file_digest, recompute_scores, sorted_sources
from check_longmemeval_production import (
    ARM, SYSTEM, PINNED, RATES, SOURCE_GATE, STAGE_PARAMS, NATIVE_OPTIONS, attempt_identity,
    check_evidence, check_native_journal, independent_summary, reader_sources, source_inputs, validate,
)
from longmemeval_answer_evidence import reader_payload
from longmemeval_production import make_protocol


DATE = '2024/01/02 (Tue) 09:00'


def raw_question():
    return {'question_id': 'arbitrary-dev', 'question_type': 'knowledge-update', 'question': 'What was reported?',
            'question_date': DATE, 'answer': 'judge-only reference', 'haystack_session_ids': ['answer-label'],
            'haystack_dates': [DATE], 'haystack_sessions': [[
                {'role': 'user', 'content': 'First user premise.'},
                {'role': 'assistant', 'content': 'An earlier assistant response.'},
                {'role': 'user', 'content': 'Second user premise.'},
                {'role': 'user', 'content': 'Nearest user premise.'},
                {'role': 'assistant', 'content': '\n  The report says "雪" and preserves every qualifier.\t'},
            ]]}


def native_fixture():
    inputs, originals, metadata = source_inputs(raw_question())
    by_index = {source['turnIndex']: source for source in originals.values()}
    source = by_index[4]
    sources = [by_index[index] for index in (0, 2, 3, 4)]
    anchor = {key: source[key] for key in ('recordId', 'sourceId', 'sessionId', 'version', 'turnIndex')}
    anchor['score'] = 1 / 61
    evidence = {'sources': copy.deepcopy(sources), 'anchors': [anchor], 'dropped': [],
                'serializedBytes': len(canonical(sources).encode())}
    return inputs, originals, metadata, evidence


def exchange(request, response):
    return {'request': copy.deepcopy(request), 'request_jsonl': canonical(request) + '\n',
            'request_sha256': digest(canonical(request)), 'response': copy.deepcopy(response),
            'response_jsonl': canonical(response) + '\n'}


def fixture_base(root):
    judge = root / 'official_judge.py'
    judge.write_text('def get_anscheck_prompt():\n    return "I will give you {}|{}|{}"\n')
    template = 'I will give you {question}|{answer}|{response}'
    protocol = {'schema_version': 1, 'created_before_first_reader_call': True, 'complete_input_counting': True,
        'token_cap': 8192, 'arms': ['first_five', 'knapsack', 'operand'], 'reader_prompt': 'Use the original sources.\n',
        'systems': {'matched_gpt54': {'reader_model': 'gpt-5.4-2026-03-05', 'judge_model': 'gpt-5.4-2026-03-05'},
                    'weak_qwen4b': {'judge_model': 'gpt-5.4-2026-03-05'}},
        'cost': {'subcap_usd': 120, 'automatic_retries': 0},
        'judge_source': {'path': str(judge), 'sha256': file_digest(judge),
            'templates': {'knowledge-update': template}, 'abstention_template': template}}
    return {'protocol': protocol, 'preregistration': {'session': 1, 'heldout_authorized': False,
            'spend_caps_usd': {'total': 250, 'reader_judge_dev': 120, 'preprocessing': 130, 'heldout400': 0}},
            'split': {'dev': [f'dev-{n:03d}' for n in range(100)],
                      'heldout': [f'heldout-{n:03d}' for n in range(400)]}}


def full_fixture(root, framed=False, paid=False):
    base = fixture_base(root)
    original_protocol, preregistration = base['protocol'], base['preregistration']
    protocol = make_protocol(original_protocol, preregistration)
    source_document = {'schema_version': 1, 'dev_only': True, 'heldout_run': False, 'source_gate': SOURCE_GATE.copy(),
                       'protocol': copy.deepcopy(protocol), 'rows': [], 'artifacts': []}
    model_root = root / 'models'
    model_artifacts, receipts = [], []
    for name, revision in PINNED.items():
        files = []
        for filename in ('config.json', 'tokenizer.json', 'model.safetensors', 'REVISION'):
            value = digest(name + filename)
            model_artifacts.append({'path': str(model_root / name / filename), 'sha256': value})
            files.append({'path': name + '/' + filename, 'sha256': value, 'sizeBytes': 17})
        receipts.append({'model': 'Qwen/' + name, 'revision': revision, 'files': files})
    identity = {'binary': {'path': str(root / 'conversation-eval'), 'sha256': 'b' * 64},
                'model_artifacts': model_artifacts, 'model_root': str(model_root),
                'argv': [str(root / 'conversation-eval'), '--models', str(model_root)]}
    profile = {'mode': 'models', 'models': receipts, 'device': 'Cpu', 'extractFacts': False, 'rewriteQueries': True,
               'embeddingQueryTask': 'conversation', 'stageParams': STAGE_PARAMS.copy(), 'embeddingBatch': 4,
               'rerankBatch': 4, 'generatorBatch': 1, 'modelLoadIncludesFileFingerprinting': True}
    def reply(command, epoch, sessions=0, sources=0, chunks=0, ingest=0, recalls=0, **extra):
        return {'command': command, 'status': 'ok', 'epoch': epoch, 'processId': 42,
                'sessionCount': sessions, 'sourceCount': sources, 'chunkCount': chunks,
                'ingestAttempts': ingest, 'recallAttempts': recalls, 'profile': copy.deepcopy(profile),
                'modelLoadSeconds': 0.0, 'resetSeconds': 0.0, 'ingestSeconds': 0.0, 'querySeconds': 0.0,
                'cumulativeTimings': {'modelLoadSeconds': 0.0, 'resetSeconds': 0.0, 'ingestSeconds': 0.0, 'querySeconds': 0.0},
                'modelUsage': {'remoteApiCalls': 0, 'localComputeUsd': None, 'exactCountersAvailable': False,
                    'reason': 'production model stages do not expose token and pair counters', 'wallTimeIncludesAllModelStages': True,
                    'maxEmbeddingTokensPerInput': 2048, 'maxQueryVariantsPerSearch': 6,
                    'maxGeneratedPlanTokensPerSearch': 1536, 'maxRerankPairsPerSearch': 20, 'maxIngestEmbeddingInputsPerAttempt': 4096},
                'transportPolicy': {'schemaVersion': 1, 'maxLineBytes': 8 * 1024 * 1024, 'maxReplyBytes': 16 * 1024 * 1024,
                    'requestTimeoutSeconds': None, 'timeoutPolicy': 'caller-enforced process termination; no automatic retry',
                    'epochScope': 'process; each successful reset creates an empty isolated history',
                    'defaultRecallOptions': NATIVE_OPTIONS.copy()}, **extra}
    initial_artifact = {'path': str(root / 'native_initializations/0000.json'), 'sha256': 'a' * 64}
    initializations = {initial_artifact['path']: {'schema_version': 1, 'status': 'ok', 'native_identity': identity,
        'exchanges': [exchange({'command': 'initialize'}, reply('initialize', 1))]}}
    protocol.update(native_identity=identity, native_initializations=[initial_artifact], prepared_native_inputs=[])
    if framed:
        protocol['prepared_inputs'] = []
        protocol['provider'] = {'scheme': 'https', 'hostname': 'example.invalid', 'sdk_version': '3.27.0'}
        protocol['input_count_provider'] = protocol['provider'].copy()
        protocol['cost_provenance'] = 'predeclared quoted-rate estimate, not a billing receipt'
    rows, raw_questions, journals, counts, caches, entries = [], {}, {}, {}, {}, []
    tokenizer = tiktoken.get_encoding('o200k_base')
    protocol_sha, work_name = 'c' * 64, 'synthetic-production'
    for n, qid in enumerate(base['split']['dev']):
        raw = raw_question()
        raw.update(question_id=qid, question=f'What report was returned for request {n}?')
        raw_questions[qid] = raw
        conversations, originals, metadata = source_inputs(raw)
        table = {source['sourceId']: {'source_id': source['sourceId'], 'session_id': source['sessionId'],
            **metadata[source['sourceId']], 'role': source['role'], 'text': source['text'],
            'native_observation_time': source['observationTime']} for source in originals.values()}
        native_input = {'conversations': conversations, 'query': f"As of {DATE}, {raw['question']}", 'options': NATIVE_OPTIONS.copy()}
        source_row = {'qid': qid, 'question': raw['question'], 'question_date': DATE,
                      'original_sources': table, 'native_input': native_input}
        source_document['rows'].append(source_row)
        _, _, _, evidence = native_fixture()
        epoch = n + 2
        recalled = reply('recall', epoch, 1, 5, 5, 1, 1, evidence=evidence,
                         recallOptions=NATIVE_OPTIONS.copy(), modelUsageBounds={
                         'maxNativeSearches': 8, 'maxQueryPlanCalls': 8, 'maxQueryEmbeddingInputs': 48,
                         'maxGeneratedPlanTokens': 12288, 'maxRerankPairs': 160})
        exchanges = [exchange({'command': 'reset'}, reply('reset', epoch)),
            exchange({'command': 'ingest', 'conversation': conversations[0]}, reply('ingest', epoch, 1, 5, 5, 1, 0,
                remembered={'sessionId': conversations[0]['sessionId'], 'version': '1', 'sourceCount': 5, 'chunkCount': 5, 'created': True})),
            exchange({'command': 'recall', 'query': native_input['query'], 'options': NATIVE_OPTIONS.copy()}, recalled)]
        journals[qid] = {'schema_version': 1, 'qid': qid, 'status': 'ok', 'initialization': initial_artifact,
            'source_input_sha256': digest(canonical(native_input)), 'elapsed_seconds': 0.1, 'exchanges': exchanges}
        sources = reader_sources(evidence, metadata)
        groups = [{'group_id': source['source_id'], 'selection_rank': rank, 'sources': [source]} for rank, source in enumerate(sources)]
        row = {'qid': qid, 'qtype': raw['question_type'], 'question': raw['question'], 'question_date': DATE,
               'native_status': 'ok', 'native_journal': {'path': str(root / 'native' / (digest(qid) + '.json')), 'sha256': 'd' * 64}}
        prompt = protocol['reader_prompt'] + canonical(reader_payload(row, {'source_groups': groups}))
        local = len(tokenizer.encode(prompt))
        exact, framing = 0, []
        if framed:
            exact = local + 10
            request = {'model': 'gpt-6.1-sol', 'input': prompt}
            key = digest(canonical({'namespace': 'sol61-input-count-v1', 'request': request}))
            check = {'request': request, 'request_sha256': digest(canonical(request)), 'cache_key': key,
                     'status': 'ok', 'response': {'input_tokens': exact}}
            counts[key], framing = copy.deepcopy(check), [check]
            protocol['prepared_inputs'].append({'qid': qid, 'reader_input_sha256': digest(prompt),
                'input_tokens': exact, 'input_count_cache_key': key, 'selection_failure': False})
        arm = {'source_groups': groups, 'retained_sources': sorted_sources(groups), 'reader_input': prompt,
            'reader_input_sha256': digest(prompt), 'input_tokens': exact, 'local_proxy_input_tokens': local,
            'framing_checks': framing, 'reader_cap': {'cap': 32768, 'exact_preflight_cap': 32256,
                'input_tokens': exact, 'selection_failure': False, 'failure_reason': None, 'source_policy': ARM},
            'reader': {'status': 'pending', 'raw_text': '', 'parsed': None},
            'judge': {'status': 'pending', 'raw_text': '', 'score': False}}
        row['systems'] = {SYSTEM: {'arms': {ARM: arm}}}
        if paid:
            parsed = {'answer': 'blue', 'abstain': False, 'claims': [{'text': 'The report is blue.',
                'citations': [{'source_id': sources[-1]['source_id'], 'quote': 'report says'}]}]}
            arm['reader'].update(status='ok', raw_text=canonical(parsed), parsed=parsed)
            arm['judge'].update(status='ok', raw_text='yes', score=True)
            for stage in ('reader', 'judge'):
                input_text = prompt if stage == 'reader' else protocol['judge_source']['templates'][row['qtype']].format(
                    question=row['question'], answer=raw['answer'], response='blue')
                incoming = exact if stage == 'reader' else len(tokenizer.encode(input_text))
                request = {'model': protocol['systems'][SYSTEM][stage + '_model'], 'input': input_text,
                    'reasoning': {'effort': 'high' if stage == 'reader' else 'medium'}, 'max_output_tokens': 4096 if stage == 'reader' else 2048,
                    'store': False, 'truncation': 'disabled', 'service_tier': 'default'}
                if stage == 'reader':
                    request['prompt_cache_options'] = {'mode': 'explicit'}
                usage = {'input_tokens': incoming, 'output_tokens': 20,
                    'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0}}
                fee = (incoming * RATES[stage]['input'] + 20 * RATES[stage]['output']) / 1e6
                reserve = ((incoming + 512) * (5 if stage == 'reader' else 2.5)
                           + request['max_output_tokens'] * RATES[stage]['output']) / 1e6
                identity_key = attempt_identity(work_name, qid, stage, 0)
                key = digest(canonical({'namespace': 'sol61-verifier-call-v1', 'request': request,
                    'call_id': identity_key, 'protocol_sha256': protocol_sha}))
                output = arm[stage]
                call = {'request': request, 'request_sha256': digest(canonical(request)), 'call_id': identity_key,
                    'protocol_sha256': protocol_sha, 'cache_key': key, 'status': 'ok', 'raw_text': output['raw_text'],
                    'response': {'model': request['model'], 'status': 'completed', 'text': output['raw_text'], 'usage': usage},
                    'usage': usage, 'cost_usd': fee, 'reserved_usd': reserve, 'cost_is_conservative': False,
                    'cost_is_conservative_reservation': False, 'unexpected_cache_writes': False, 'validation_failures': []}
                output['call'], caches[key] = call, copy.deepcopy(call)
                entries.append({'cache_key': key, 'call_id': identity_key, 'category': 'reader_judge_dev',
                    'status': 'ok', 'usage': copy.deepcopy(usage), 'cost_usd': fee, 'reserved_usd': reserve})
        arm['scores'] = recompute_scores(arm, allow_incomplete=not paid)
        rows.append(row)
        protocol['prepared_native_inputs'].append({'qid': qid, 'reader_input_sha256': digest(prompt), 'native_journal': row['native_journal']})
    total = sum(entry['cost_usd'] for entry in entries)
    ledger = {'schema_version': 1, 'caps_usd': copy.deepcopy(preregistration['spend_caps_usd']), 'entries': entries,
              'by_category_usd': {'reader_judge_dev': total, 'preprocessing': 0.0, 'heldout400': 0.0}, 'total_usd': total}
    report = {'schema_version': 1, 'dev_only': True, 'heldout_run': False, 'reader_systems': [SYSTEM], 'source_gate': SOURCE_GATE.copy(),
              'framing_frozen': framed, 'complete': paid, 'rows': rows, 'global_recorded_spend_usd': total,
              'incremental_spend_usd': total, 'summary': independent_summary(rows, [row['systems'][SYSTEM]['arms'][ARM]['scores'] for row in rows])}
    return {'report': report, 'split': base['split'], 'protocol': protocol, 'base_protocol': original_protocol,
            'preregistration': preregistration, 'raw_questions': raw_questions, 'source_document': source_document,
            'native_journals': journals, 'initializations': initializations, 'ledger': ledger, 'work_name': work_name,
            'protocol_sha256': protocol_sha, 'cached_calls': caches, 'input_counts': counts, 'inputs_only': not paid}


class NativeSourceChecks(unittest.TestCase):
    def test_source_only_identity_is_invariant_to_all_labels_and_question_metadata(self):
        raw = raw_question()
        before = copy.deepcopy(raw)
        first = source_inputs(raw)
        raw.update(question_id='another-id', question_type='multi-session', question='Another question?',
                   question_date='2025/02/03 (Mon) 10:00', answer='another reference',
                   haystack_session_ids=['noans-label'], answer_session_ids=['gold'], has_answer=True)
        for turn in raw['haystack_sessions'][0]:
            turn['has_answer'] = True
        self.assertEqual(source_inputs(raw), first)
        self.assertEqual(before['haystack_sessions'][0][4]['content'], first[0][0]['turns'][4]['text'])
        self.assertEqual(first[0][0]['turns'][0]['observationTime'], 1704186000000)
        self.assertEqual(set(first[0][0]), {'sessionId', 'version', 'turns'})

    def test_observation_scope_and_whole_user_premises_are_preserved(self):
        _, originals, metadata, evidence = native_fixture()
        before = copy.deepcopy((originals, metadata, evidence))
        selected = check_evidence(evidence, originals, NATIVE_OPTIONS)
        anchor = selected[evidence['anchors'][0]['recordId']]
        self.assertEqual([selected[rid]['turnIndex'] for rid in anchor['premiseRecordIds']], [0, 2, 3])
        transformed = reader_sources(evidence, metadata)
        self.assertEqual([item['turn_index'] for item in transformed], [0, 2, 3, 4])
        self.assertEqual(transformed[-1]['text'], raw_question()['haystack_sessions'][0][-1]['content'])
        self.assertEqual((originals, metadata, evidence), before)

    def test_source_text_role_identity_time_and_position_tampering_are_rejected(self):
        for field, value in [('text', 'Truncated.'), ('role', 'user'), ('version', '2'),
                             ('turnIndex', 3), ('observationTime', 0), ('sessionId', 'foreign'),
                             ('sourceId', 'foreign'), ('recordId', 'foreign')]:
            with self.subTest(field=field):
                _, originals, _, evidence = native_fixture()
                evidence['sources'][-1][field] = value
                with self.assertRaises(ValueError):
                    check_evidence(evidence, originals, NATIVE_OPTIONS)

    def test_missing_premises_extra_sources_duplicates_and_reordering_are_rejected(self):
        for change in ('missing', 'extra', 'duplicate', 'order'):
            with self.subTest(change=change):
                _, originals, _, evidence = native_fixture()
                if change == 'missing':
                    evidence['sources'].pop(0)
                elif change == 'extra':
                    evidence['sources'].insert(1, copy.deepcopy(list(originals.values())[1]))
                elif change == 'duplicate':
                    evidence['sources'].append(copy.deepcopy(evidence['sources'][0]))
                else:
                    evidence['sources'].reverse()
                evidence['serializedBytes'] = len(canonical(evidence['sources']).encode())
                with self.assertRaises(ValueError):
                    check_evidence(evidence, originals, NATIVE_OPTIONS)

    def test_serialized_byte_and_source_count_limits_are_inclusive(self):
        _, originals, _, evidence = native_fixture()
        size, count = evidence['serializedBytes'], len(evidence['sources'])
        for delta in (-1, 0, 1):
            for field, boundary in [('maxBytes', size), ('maxSources', count)]:
                with self.subTest(field=field, delta=delta):
                    options = {**NATIVE_OPTIONS, field: boundary + delta}
                    if delta < 0:
                        with self.assertRaisesRegex(ValueError, 'exceeds source bounds'):
                            check_evidence(evidence, originals, options)
                    else:
                        check_evidence(evidence, originals, options)
        evidence['serializedBytes'] += 1
        with self.assertRaisesRegex(ValueError, 'byte accounting'):
            check_evidence(evidence, originals, NATIVE_OPTIONS)

    def test_anchor_rank_boundaries_and_nonfinite_scores_are_rejected(self):
        for rank in (63, 64, 65):
            _, originals, _, evidence = native_fixture()
            evidence['anchors'][0]['score'] = 1 / (60 + rank)
            if rank <= 64:
                check_evidence(evidence, originals, NATIVE_OPTIONS)
            else:
                with self.assertRaisesRegex(ValueError, 'RRF bounds'):
                    check_evidence(evidence, originals, NATIVE_OPTIONS)
        for score in (0, -1, math.nan, math.inf):
            _, originals, _, evidence = native_fixture()
            evidence['anchors'][0]['score'] = score
            with self.assertRaisesRegex(ValueError, 'anchor score'):
                check_evidence(evidence, originals, NATIVE_OPTIONS)

    def test_unmapped_drops_and_missing_immutable_records_fail_closed(self):
        _, originals, _, evidence = native_fixture()
        source = list(originals.values())[1]
        drop = {key: source[key] for key in ('recordId', 'sourceId', 'sessionId', 'version')}
        evidence['dropped'].append({**drop, 'reason': 'max_bytes'})
        check_evidence(evidence, originals, NATIVE_OPTIONS)
        evidence['dropped'][0]['reason'] = 'missing_source'
        with self.assertRaisesRegex(ValueError, 'lost a source'):
            check_evidence(evidence, originals, NATIVE_OPTIONS)
        evidence['dropped'][0].update(recordId='invented', reason='max_bytes')
        with self.assertRaisesRegex(ValueError, 'dropped identity'):
            check_evidence(evidence, originals, NATIVE_OPTIONS)


class ProductionPipelineChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def arm(self, fixture):
        return fixture['report']['rows'][0]['systems'][SYSTEM]['arms'][ARM]

    def test_all100_native_only_and_exact_framed_inputs_pass_without_paid_calls(self):
        for framed in (False, True):
            with self.subTest(framed=framed):
                fixture = full_fixture(self.root, framed=framed)
                before = copy.deepcopy(fixture)
                result = validate(**fixture)
                self.assertEqual(result['summary']['n'], 100)
                self.assertEqual(result['native_requests_verified'], 301)
                self.assertEqual(result['paid_calls_verified'], 0)
                self.assertEqual(result['input_count_records_verified'], 100 if framed else 0)
                self.assertNotIn('acceptance_met', result['summary'])
                self.assertEqual(fixture, before)

    def test_native_input_labels_missing_ingestion_and_repeated_attempts_are_rejected(self):
        for mutation in ('label', 'missing', 'repeat', 'epoch', 'model', 'timing', 'source_count', 'chunk_count'):
            with self.subTest(mutation=mutation):
                fixture = full_fixture(self.root)
                qid = fixture['report']['rows'][0]['qid']
                journal = fixture['native_journals'][qid]
                if mutation == 'label':
                    fixture['source_document']['rows'][0]['native_input']['qtype'] = 'knowledge-update'
                elif mutation == 'missing':
                    journal['exchanges'].pop(1)
                elif mutation == 'repeat':
                    journal['exchanges'].append(copy.deepcopy(journal['exchanges'][-1]))
                elif mutation == 'epoch':
                    value = journal['exchanges'][0]
                    value['response']['epoch'] += 1
                    value['response_jsonl'] = canonical(value['response']) + '\n'
                elif mutation == 'model':
                    value = journal['exchanges'][0]
                    value['response']['profile']['models'][0]['revision'] = 'bad'
                    value['response_jsonl'] = canonical(value['response']) + '\n'
                else:
                    value = journal['exchanges'][-1]
                    if mutation == 'timing':
                        value['response']['cumulativeTimings']['querySeconds'] = 1.0
                    else:
                        field = 'sourceCount' if mutation == 'source_count' else 'chunkCount'
                        value['response'][field] += 1
                    value['response_jsonl'] = canonical(value['response']) + '\n'
                with self.assertRaises(ValueError):
                    validate(**fixture)

    def test_reader_transform_and_raw_native_journal_cannot_hide_changes(self):
        for mutation in ('reader', 'raw', 'premise'):
            with self.subTest(mutation=mutation):
                fixture = full_fixture(self.root, framed=True)
                if mutation == 'reader':
                    self.arm(fixture)['reader_input'] += '\nreference: blue'
                else:
                    qid = fixture['report']['rows'][0]['qid']
                    value = fixture['native_journals'][qid]['exchanges'][-1]
                    if mutation == 'raw':
                        value['response_jsonl'] = '{}\n'
                    else:
                        value['response']['evidence']['sources'].pop(0)
                        value['response_jsonl'] = canonical(value['response']) + '\n'
                with self.assertRaises(ValueError):
                    validate(**fixture)

    def test_exact_count_boundary_drops_no_sources_and_keeps_all100_failures(self):
        for count in (32255, 32256, 32257):
            with self.subTest(count=count):
                fixture = full_fixture(self.root, framed=True)
                arm = self.arm(fixture)
                sources = copy.deepcopy(arm['source_groups'])
                check = arm['framing_checks'][0]
                check['response']['input_tokens'] = count
                fixture['input_counts'][check['cache_key']] = copy.deepcopy(check)
                fixture['protocol']['prepared_inputs'][0]['input_tokens'] = count
                arm['input_tokens'] = arm['reader_cap']['input_tokens'] = count
                if count > 32256:
                    arm['reader_cap'].update(selection_failure=True, failure_reason='exact_input_count_exceeds_cap')
                    arm['reader']['status'] = 'selection_failure'
                    arm['scores'] = recompute_scores(arm, allow_incomplete=True)
                    fixture['protocol']['prepared_inputs'][0]['selection_failure'] = True
                rows = fixture['report']['rows']
                fixture['report']['summary'] = independent_summary(rows, [row['systems'][SYSTEM]['arms'][ARM]['scores'] for row in rows])
                result = validate(**fixture)
                self.assertEqual(result['summary']['n'], 100)
                self.assertEqual(arm['source_groups'], sources)
                self.assertEqual(result['paid_calls_verified'], 0)

    def test_paid_outputs_and_official_judge_are_verified_without_prior_artifacts(self):
        fixture = full_fixture(self.root, framed=True, paid=True)
        result = validate(**fixture)
        self.assertEqual(result['paid_calls_verified'], 200)
        self.assertEqual(result['summary']['n'], 100)
        self.assertEqual(result['summary']['correct'], 100)
        self.assertNotIn('acceptance_met', result['summary'])
        fixture['report']['summary']['correct'] = 99
        with self.assertRaisesRegex(ValueError, 'summary changed'):
            validate(**fixture)

    def test_pending_reader_blocks_judges_and_no_failed_call_charge_can_disappear(self):
        fixture = full_fixture(self.root, framed=True, paid=True)
        self.arm(fixture)['reader']['status'] = 'pending'
        with self.assertRaises(ValueError):
            validate(**fixture)
        fixture = full_fixture(self.root, framed=True, paid=True)
        fixture['ledger']['entries'].pop(0)
        with self.assertRaises(ValueError):
            validate(**fixture)

    def test_spend_caps_and_heldout_charge_are_enforced(self):
        for category, cost in [('reader_judge_dev', 120.01), ('heldout400', 0.01), ('preprocessing', 250.01),
                               ('reader_judge_dev', 14.01)]:
            with self.subTest(category=category):
                fixture = full_fixture(self.root)
                entry = {'cache_key': 'external', 'category': category, 'cost_usd': cost}
                if cost == 14.01:
                    entry['call_id'] = attempt_identity(fixture['work_name'], fixture['report']['rows'][0]['qid'], 'reader', 0)
                    fixture['report']['incremental_spend_usd'] = cost
                fixture['ledger']['entries'].append(entry)
                fixture['ledger']['by_category_usd'][category] = cost
                fixture['ledger']['total_usd'] = fixture['report']['global_recorded_spend_usd'] = cost
                with self.assertRaisesRegex(ValueError, 'spend cap'):
                    validate(**fixture)

    def test_native_transport_failure_keeps_one_failed_question_in_the_100_denominator(self):
        fixture = full_fixture(self.root)
        row = fixture['report']['rows'][0]
        arm = self.arm(fixture)
        journal = fixture['native_journals'][row['qid']]
        journal.update(status='error', error_type='ValueError')
        journal['exchanges'][-1].update(response=None, response_jsonl=None, error_type='ValueError')
        row['native_status'] = 'error'
        arm.update(source_groups=[], retained_sources=[])
        prompt = fixture['protocol']['reader_prompt'] + canonical(reader_payload(row, {'source_groups': []}))
        arm.update(reader_input=prompt, reader_input_sha256=digest(prompt),
                   local_proxy_input_tokens=len(tiktoken.get_encoding('o200k_base').encode(prompt)))
        arm['reader_cap'].update(selection_failure=True, failure_reason='native_bridge_failed')
        arm['reader']['status'] = 'selection_failure'
        arm['scores'] = recompute_scores(arm, allow_incomplete=True)
        fixture['protocol']['prepared_native_inputs'][0]['reader_input_sha256'] = digest(prompt)
        fixture['report']['summary'] = independent_summary(fixture['report']['rows'],
            [row['systems'][SYSTEM]['arms'][ARM]['scores'] for row in fixture['report']['rows']])
        result = validate(**fixture)
        self.assertEqual(result['summary']['n'], 100)
        self.assertEqual(result['summary']['native_status'], {'error': 1, 'ok': 99})
        self.assertEqual(result['native_failed_requests'], 1)

    def test_interrupted_first_reset_accepts_no_reply_and_rejects_following_exchanges(self):
        fixture = full_fixture(self.root)
        source_row = fixture['source_document']['rows'][0]
        original = fixture['native_journals'][source_row['qid']]
        _, originals, _ = source_inputs(fixture['raw_questions'][source_row['qid']])
        journal = copy.deepcopy(original)
        journal.update(status='error', error_type='InterruptedNativeAttempt', elapsed_seconds=None, timing_unknown=True)
        journal['exchanges'] = journal['exchanges'][:1]
        journal['exchanges'][0].update(response=None, response_jsonl=None)
        before = copy.deepcopy(journal)
        self.assertEqual(check_native_journal(journal, source_row, originals,
            fixture['protocol']['native_identity'], original['initialization'], 1), (None, 1))
        self.assertEqual(journal, before)
        journal['exchanges'].append(copy.deepcopy(original['exchanges'][1]))
        journal['exchanges'][0]['error_type'] = 'ValueError'
        with self.assertRaisesRegex(ValueError, 'native failure retried'):
            check_native_journal(journal, source_row, originals,
                fixture['protocol']['native_identity'], original['initialization'], 1)
        journal = copy.deepcopy(before)
        journal['timing_unknown'] = False
        with self.assertRaisesRegex(ValueError, 'elapsed time'):
            check_native_journal(journal, source_row, originals,
                fixture['protocol']['native_identity'], original['initialization'], 1)

    def test_interrupted_final_exchange_keeps_ingestion_but_rejects_a_completed_reply(self):
        fixture = full_fixture(self.root)
        source_row = fixture['source_document']['rows'][0]
        original = fixture['native_journals'][source_row['qid']]
        _, originals, _ = source_inputs(fixture['raw_questions'][source_row['qid']])
        journal = copy.deepcopy(original)
        journal.update(status='error', error_type='InterruptedNativeAttempt', elapsed_seconds=None, timing_unknown=True)
        journal['exchanges'][-1].update(response=None, response_jsonl=None)
        self.assertEqual(check_native_journal(journal, source_row, originals,
            fixture['protocol']['native_identity'], original['initialization'], 1), (None, 2))
        journal['exchanges'][-1] = copy.deepcopy(original['exchanges'][-1])
        with self.assertRaisesRegex(ValueError, 'valid completed native response recorded as failure'):
            check_native_journal(journal, source_row, originals,
                fixture['protocol']['native_identity'], original['initialization'], 1)

    def test_failed_paid_attempt_reservation_is_preserved_before_transport_retry(self):
        fixture = full_fixture(self.root, framed=True, paid=True)
        row, arm = fixture['report']['rows'][0], self.arm(fixture)
        original = copy.deepcopy(arm['reader']['call'])
        failed = copy.deepcopy(original)
        failed.update(status='error', raw_text='', response=None, usage=None,
                      cost_usd=original['reserved_usd'], cost_is_conservative_reservation=True,
                      http_status=520, error_type='InternalServerError')
        final = arm['reader']['call']
        final['call_id'] = attempt_identity(fixture['work_name'], row['qid'], 'reader', 1)
        final['cache_key'] = digest(canonical({'namespace': 'sol61-verifier-call-v1', 'request': final['request'],
            'call_id': final['call_id'], 'protocol_sha256': fixture['protocol_sha256']}))
        arm['reader']['transport_attempts'] = [failed]
        fixture['cached_calls'].pop(original['cache_key'])
        fixture['cached_calls'][failed['cache_key']] = copy.deepcopy(failed)
        fixture['cached_calls'][final['cache_key']] = copy.deepcopy(final)
        entry = next(item for item in fixture['ledger']['entries'] if item['cache_key'] == original['cache_key'])
        entry.update(cache_key=final['cache_key'], call_id=final['call_id'])
        fixture['ledger']['entries'].append({'cache_key': failed['cache_key'], 'call_id': failed['call_id'],
            'category': 'reader_judge_dev', 'status': 'error', 'usage': None, 'cost_usd': failed['cost_usd'],
            'reserved_usd': failed['reserved_usd']})
        total = sum(item['cost_usd'] for item in fixture['ledger']['entries'])
        fixture['ledger']['by_category_usd']['reader_judge_dev'] = total
        fixture['ledger']['total_usd'] = total
        fixture['report'].update(incremental_spend_usd=total, global_recorded_spend_usd=total)
        result = validate(**fixture)
        self.assertEqual(result['paid_calls_verified'], 201)
        self.assertEqual(result['summary']['n'], 100)
        fixture['ledger']['entries'].pop()
        with self.assertRaises(ValueError):
            validate(**fixture)


if __name__ == '__main__':
    unittest.main()
