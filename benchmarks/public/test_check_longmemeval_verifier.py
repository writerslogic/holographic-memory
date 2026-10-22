# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0"]
# ///
"""Source-only verifier provenance, frozen raw prior proofs, costs, and failure denominators."""

import copy
import statistics
import tempfile
import unittest
from pathlib import Path

import tiktoken

from check_longmemeval_answers import canonical, digest, recompute_scores, sorted_sources, wilson
from check_longmemeval_verifier import (
    ARM, SYSTEM, PRIOR_NAMES, RATES, SOURCE_POLICY, TRANSPORT_RETRY_POLICY, fee, independent_summary, replay_sources, validate,
)
from longmemeval_answer_evidence import reader_payload
from test_check_longmemeval_complete_history import make_complete_fixture


WORK_NAME = 'synthetic-verifier'
PROTOCOL_SHA = '4' * 64
VERIFIER_RULES = '\nAnswer independently from original sources and preserve uncertainty.\n'


def old_call(prompt, raw, high, base, key):
    request = {'model': 'gpt-5.4-2026-03-05', 'input': prompt,
               'reasoning': {'effort': 'high' if high else 'medium'}, 'max_output_tokens': 8192 if high else 4096,
               'store': False, 'truncation': 'disabled'}
    usage = {'input_tokens': len(tiktoken.get_encoding('o200k_base').encode(prompt)), 'output_tokens': 20,
             'input_tokens_details': {'cached_tokens': 0}}
    return {'request': request, 'request_sha256': digest(canonical(request)), 'cache_key': key,
            'status': 'ok', 'raw_text': raw, 'response': {'model': request['model'], 'status': 'completed',
                'text': raw, 'usage': copy.deepcopy(usage)}, 'usage': usage,
            'cost_usd': (usage['input_tokens'] * 2.5 + 20 * 15) / 1e6}


def update_spend(fixture):
    ledger, report = fixture['ledger'], fixture['report']
    categories = {category: 0.0 for category in ledger['caps_usd'] if category != 'total'}
    bases = {digest(f"sol61-verifier:{WORK_NAME}:{stage}:{row['qid']}") for row in report['rows'] for stage in ('reader', 'judge')}
    identities = bases | {base + f':transport-retry:{attempt}' for base in bases for attempt in (1, 2)}
    for entry in ledger['entries']:
        categories[entry['category']] += entry['cost_usd']
    ledger.update(by_category_usd=categories, total_usd=sum(categories.values()))
    report['global_recorded_spend_usd'] = ledger['total_usd']
    report['incremental_spend_usd'] = sum(entry['cost_usd'] for entry in ledger['entries'] if entry.get('call_id') in identities)


def make_fixture(root):
    old = make_complete_fixture(root)
    base, rules = old['base_protocol'], [old['full_rules'], old['complete_rules'], VERIFIER_RULES]
    protocol = {'schema_version': 1, 'dev_only': True, 'heldout_run': False, 'acceptance_correct': 100,
                'reader_prompt': base['reader_prompt'] + ''.join(rules), 'additional_reasoning_rules': VERIFIER_RULES,
                'systems': {SYSTEM: {'reader_model': 'gpt-6.1-sol', 'judge_model': 'gpt-5.4-2026-03-05',
                                     'reader_reasoning_effort': 'high', 'judge_reasoning_effort': 'medium',
                                     'reader_max_output_tokens': 4096, 'judge_max_output_tokens': 2048}},
                'judge_source': copy.deepcopy(base['judge_source']), 'token_cap': 32768, 'exact_preflight_cap': 32256,
                'framing_reserve_tokens': 512, 'local_proxy_cap': 24000, 'source_policy': SOURCE_POLICY,
                'source_gate': {'metadata_ids_verified': 500, 'dev_records_decoded': 100, 'heldout_records_decoded': 0},
                'source_selection': {'prior_names': list(PRIOR_NAMES), 'prior_answers_visible': False,
                    'question_only_counterevidence': 'reserved user prior then equal-weight RRF60 fused ranking',
                    'whole_turns_only': True, 'cap_overflow': 'selection_failure; no source clipping'},
                'input_count': {'endpoint': '/v1/responses/input_tokens', 'request_fields': ['model', 'input'], 'settings_framing_reserve': 512},
                'service_tier': 'default', 'reader_prompt_cache_options': {'mode': 'explicit'}, 'rates_usd_per_million': copy.deepcopy(RATES),
                'new_spend_cap_usd': 14.0, 'automatic_transport_retries': 2,
                'transport_retry_policy': copy.deepcopy(TRANSPORT_RETRY_POLICY), 'reader_judge_concurrency': 1,
                'global_spend_caps_usd': copy.deepcopy(old['preregistration']['spend_caps_usd']),
                'limitations': ['Reader alias is not an immutable dated snapshot.']}
    snapshots, proofs = [], []
    for index, name in enumerate(PRIOR_NAMES):
        metadata = {'schema_version': 1, 'dev_only': True, 'heldout_run': False, 'name': name,
                    'source_path': f'prior/{name}.json', 'source_sha256_at_snapshot': '0' * 64,
                    'system': 'matched_gpt54', 'arm': 'operand' if index == 0 else 'full_history'}
        snapshots.append({**metadata, 'rows': []})
        proofs.append({**metadata, 'rows': []})
    rows, counts, caches = [], {}, {}
    ledger = {'schema_version': 1, 'caps_usd': copy.deepcopy(old['ledger']['caps_usd']),
              'entries': copy.deepcopy(old['ledger']['entries'][:2])}
    tokenizer = tiktoken.get_encoding('o200k_base')
    exact_counts = []
    for index, previous in enumerate(old['report']['rows']):
        qid, prior_arm = previous['qid'], previous['systems']['matched_gpt54']['arms']['full_history']
        row = {key: previous[key] for key in ('qid', 'qtype', 'question', 'question_date', 'conversation_groups', 'orphan_assistant_groups')}
        groups = copy.deepcopy(prior_arm['source_groups'])
        cited = groups[0]['sources'][0]['source_id']
        for prior_index, (snapshot, proof) in enumerate(zip(snapshots, proofs, strict=True)):
            packet = copy.deepcopy(old['baseline_rows'][index]['systems']['matched_gpt54']['arms']['operand']['source_groups'] if prior_index == 0 else groups)
            seed = {'qid': qid, 'question': row['question'], 'question_date': row['question_date'],
                    'reader_status': 'ok', 'source_groups': packet, 'cited_source_ids': [cited]}
            snapshot['rows'].append(seed)
            prefix = base['reader_prompt'] + (rules[0] if prior_index else '') + (rules[1] if prior_index == 3 else '')
            prompt = prefix + canonical(reader_payload(row, {'source_groups': packet}))
            raw = prior_arm['reader']['raw_text']
            proof['rows'].append({'qid': qid, 'reader': {'status': 'ok', 'raw_text': raw,
                'parsed': copy.deepcopy(prior_arm['reader']['parsed']),
                'call': old_call(prompt, raw, prior_index == 3, base, digest(f'{prior_index}:{qid}'))}})
        selected = [dict(copy.deepcopy(group), selection_rank=rank) for rank, group in enumerate(groups)]
        prompt = protocol['reader_prompt'] + canonical(reader_payload(row, {'source_groups': selected}))
        local = len(tokenizer.encode(prompt))
        exact = local + 17
        exact_counts.append(exact)
        count_request = {'model': 'gpt-6.1-sol', 'input': prompt}
        count_key = digest(canonical({'namespace': 'sol61-input-count-v1', 'request': count_request}))
        check = {'cache_key': count_key, 'request': count_request, 'request_sha256': digest(canonical(count_request)),
                 'status': 'ok', 'response': {'object': 'response.input_tokens', 'input_tokens': exact}}
        counts[count_key] = copy.deepcopy(check)
        diagnostics = {'mapped_citations': [{'prior_index': i, 'cited_source_id': cited, 'original_source_ids': [cited]} for i in range(4)],
                       'unmapped_citations': [], 'ambiguous_citations': [], 'dropped_cited_source_ids': [],
                       'dropped_premise_source_ids': [], 'dropped_counterevidence_source_ids': [], 'dropped_prior_citations': []}
        arm = {'source_groups': selected, 'retained_sources': sorted_sources(selected), 'reader_input': prompt,
               'reader_input_sha256': digest(prompt), 'input_tokens': exact, 'local_proxy_input_tokens': local,
               'framing_checks': [check], 'source_selection': {'input_tokens': local, 'cap': 24000, 'selection_failure': False,
                    'failure_reason': None, 'diagnostics': diagnostics,
                    'source_policy': 'complete cited original turns; preceding/opening user premises; question-ranked counterevidence'},
               'reader_cap': {'cap': 32768, 'exact_preflight_cap': 32256, 'input_tokens': exact, 'selection_failure': False,
                              'failure_reason': None, 'source_policy': 'citation_union'},
               'reader': {'status': 'ok', 'raw_text': prior_arm['reader']['raw_text'], 'parsed': copy.deepcopy(prior_arm['reader']['parsed'])},
               'judge': {'status': 'ok', 'raw_text': 'yes', 'score': True}}
        row['counterevidence_group_ids'] = [groups[i]['group_id'] for i in (1, 0, 2)]
        row['systems'] = {SYSTEM: {'arms': {ARM: arm}}}
        for stage in ('reader', 'judge'):
            request_input = prompt if stage == 'reader' else base['judge_source']['templates'][row['qtype']].format(
                question=row['question'], answer='blue', response='blue')
            request = {'model': protocol['systems'][SYSTEM][stage + '_model'], 'input': request_input,
                       'reasoning': {'effort': 'high' if stage == 'reader' else 'medium'},
                       'max_output_tokens': 4096 if stage == 'reader' else 2048, 'store': False,
                       'truncation': 'disabled', 'service_tier': 'default'}
            if stage == 'reader':
                request['prompt_cache_options'] = {'mode': 'explicit'}
            incoming = exact if stage == 'reader' else len(tokenizer.encode(request_input))
            usage = {'input_tokens': incoming, 'output_tokens': 20,
                     'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0,
                                             'cache_write_tokens_details': {'ttl_30m_tokens': 0, 'ttl_12h_tokens': 0}}}
            rate = RATES[stage]
            cost = (incoming * rate['input'] + 20 * rate['output']) / 1e6
            reserve = ((incoming + 512) * (5.0 if stage == 'reader' else 2.5)
                       + request['max_output_tokens'] * rate['output']) / 1e6
            call_id = digest(f'sol61-verifier:{WORK_NAME}:{stage}:{qid}')
            key = digest(canonical({'namespace': 'sol61-verifier-call-v1', 'request': request,
                                    'call_id': call_id, 'protocol_sha256': PROTOCOL_SHA}))
            raw = arm[stage]['raw_text']
            call = {'request': request, 'request_sha256': digest(canonical(request)), 'cache_key': key,
                    'call_id': call_id, 'protocol_sha256': PROTOCOL_SHA, 'status': 'ok', 'raw_text': raw,
                    'response': {'model': request['model'], 'status': 'completed', 'service_tier': 'default',
                                 'text': raw, 'usage': copy.deepcopy(usage)}, 'usage': usage, 'cost_usd': cost,
                    'reserved_usd': reserve, 'cost_is_conservative': False, 'unexpected_cache_writes': False,
                    'cost_is_conservative_reservation': False, 'validation_failures': []}
            arm[stage]['call'] = call
            arm[stage]['transport_attempts'] = []
            caches[key] = copy.deepcopy(call)
            ledger['entries'].append({'cache_key': key, 'call_id': call_id, 'category': 'reader_judge_dev', 'status': 'ok',
                                      'cost_usd': cost, 'usage': copy.deepcopy(usage), 'reserved_usd': reserve})
        arm['scores'] = recompute_scores(arm)
        rows.append(row)
    protocol['prepared_inputs'] = [{'qid': row['qid'], 'reader_input_sha256': row['systems'][SYSTEM]['arms'][ARM]['reader_input_sha256'],
                                   'input_tokens': row['systems'][SYSTEM]['arms'][ARM]['input_tokens'],
                                   'input_count_cache_key': row['systems'][SYSTEM]['arms'][ARM]['framing_checks'][0]['cache_key']} for row in rows]
    summary = {'n': 100, 'correct': 100, 'answer_accuracy': 1.0, 'answer_accuracy_ci95': wilson(100, 100),
               'acceptance_correct': 100, 'acceptance_met': True, 'reader_status': {'ok': 100}, 'judge_status': {'ok': 100},
               'input_tokens': {'median': statistics.median(exact_counts), 'max': max(exact_counts)},
               'abstentions': 0, 'exact_quote_claims': 100, 'total_claims': 100, 'user_occurrences': 200, 'assistant_occurrences': 400}
    report = {'schema_version': 1, 'dev_only': True, 'heldout_run': False, 'complete': True,
              'reader_systems': [SYSTEM], 'rows': rows, 'summary': summary}
    result = {'report': report, 'split': old['split'], 'protocol': protocol, 'base_protocol': base,
              'preregistration': old['preregistration'], 'raw_questions': old['raw_questions'], 'history_rows': old['history_rows'],
              'snapshots': snapshots, 'reader_proofs': proofs, 'references': old['references'], 'ledger': ledger,
              'work_name': WORK_NAME, 'protocol_sha256': PROTOCOL_SHA, 'rules': rules, 'cached_calls': caches, 'input_counts': counts}
    update_spend(result)
    return result


def refresh_summary(fixture):
    rows = fixture['report']['rows']
    scores = [recompute_scores(row['systems'][SYSTEM]['arms'][ARM], allow_incomplete=not fixture['report']['complete']) for row in rows]
    for row, score in zip(rows, scores):
        row['systems'][SYSTEM]['arms'][ARM]['scores'] = score
    fixture['report']['summary'] = independent_summary(rows, scores, fixture['report']['complete'])
    update_spend(fixture)


def transport_chain(fixture, errors, stage='reader', exhausted=False):
    row = fixture['report']['rows'][0]
    arm = row['systems'][SYSTEM]['arms'][ARM]
    original = arm[stage]['call']
    old_key = original['cache_key']
    fixture['cached_calls'].pop(old_key)
    fixture['ledger']['entries'] = [entry for entry in fixture['ledger']['entries'] if entry['cache_key'] != old_key]
    base = digest(f"sol61-verifier:{WORK_NAME}:{stage}:{row['qid']}")
    calls = []
    for attempt, error in enumerate(errors + ([] if exhausted else [None])):
        call = copy.deepcopy(original)
        if error is not None:
            call = {key: value for key, value in call.items() if key not in ('validation_failures', 'cost_is_conservative', 'unexpected_cache_writes')}
            call.update(status='error', raw_text='', response=None, usage=None, cost_usd=call['reserved_usd'],
                        cost_is_conservative_reservation=True, **error)
        call['call_id'] = base if attempt == 0 else base + f':transport-retry:{attempt}'
        call['cache_key'] = digest(canonical({'namespace': 'sol61-verifier-call-v1', 'request': call['request'],
                                            'call_id': call['call_id'], 'protocol_sha256': PROTOCOL_SHA}))
        fixture['cached_calls'][call['cache_key']] = copy.deepcopy(call)
        fixture['ledger']['entries'].append({'cache_key': call['cache_key'], 'call_id': call['call_id'],
            'category': 'reader_judge_dev', 'status': call['status'], 'cost_usd': call['cost_usd'],
            'usage': copy.deepcopy(call['usage']), 'reserved_usd': call['reserved_usd']})
        calls.append(call)
    arm[stage].update(call=calls[-1], transport_attempts=calls[:-1])
    if exhausted:
        arm[stage].update(status='error', raw_text='')
        arm[stage]['parsed' if stage == 'reader' else 'score'] = None if stage == 'reader' else False
        if stage == 'reader':
            key = arm['judge']['call']['cache_key']
            fixture['cached_calls'].pop(key)
            fixture['ledger']['entries'] = [entry for entry in fixture['ledger']['entries'] if entry['cache_key'] != key]
            arm['judge'] = {'status': 'reader_failure', 'raw_text': '', 'score': False}
    refresh_summary(fixture)
    return calls


class VerifierChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = make_fixture(self.root)

    def arm(self, fixture=None):
        return (fixture or self.fixture)['report']['rows'][0]['systems'][SYSTEM]['arms'][ARM]

    def test_complete_verifier_validates_400_raw_prior_proofs_and_100_exact_inputs(self):
        before = copy.deepcopy(self.fixture)
        result = validate(**self.fixture)
        self.assertEqual(result['prior_reader_proofs'], {'reader_rows_verified': 400, 'missing_original_call_records': 0})
        self.assertEqual(result['input_count_records_verified'], 100)
        self.assertEqual(result['paid_calls_verified'], 200)
        self.assertEqual(result['summary']['n'], 100)
        self.assertTrue(result['summary']['acceptance_met'])
        self.assertFalse(result['reader_identity']['immutable_snapshot'])
        self.assertEqual(before, self.fixture)

    def test_inputs_only_validates_unissued_frozen_packets_before_model_calls(self):
        fixture = self.fixture
        fixture['report']['complete'] = False
        for row in fixture['report']['rows']:
            arm = row['systems'][SYSTEM]['arms'][ARM]
            arm['reader'] = {'status': 'pending', 'raw_text': '', 'parsed': None}
            arm['judge'] = {'status': 'pending', 'raw_text': '', 'score': False}
        fixture['cached_calls'] = {}
        fixture['ledger']['entries'] = fixture['ledger']['entries'][:2]
        refresh_summary(fixture)
        result = validate(**fixture, inputs_only=True)
        self.assertEqual(result['paid_calls_verified'], 0)
        self.assertEqual(result['input_count_records_verified'], 100)
        self.assertFalse(result['summary']['acceptance_met'])

    def test_seed_citations_must_derive_from_strict_frozen_raw_proof(self):
        self.fixture['snapshots'][0]['rows'][0]['cited_source_ids'] = ['fabricated-source']
        with self.assertRaisesRegex(ValueError, 'derive from frozen raw reader proof'):
            validate(**self.fixture)
        fixture = make_fixture(self.root)
        fixture['reader_proofs'][0]['rows'][0]['reader']['parsed']['answer'] = 'changed decoded answer'
        with self.assertRaisesRegex(ValueError, 'strict raw JSON'):
            validate(**fixture)

    def test_prior_answer_and_claim_text_never_enter_new_model_packet(self):
        reader = self.fixture['reader_proofs'][0]['rows'][0]['reader']
        reader['parsed']['answer'] = 'OLD ANSWER TEXT ONLY'
        reader['parsed']['claims'][0]['text'] = 'OLD CLAIM TEXT ONLY'
        raw = canonical(reader['parsed'])
        reader['raw_text'] = reader['call']['raw_text'] = reader['call']['response']['text'] = raw
        result = validate(**self.fixture)
        self.assertTrue(result['source_selection_replayed'])
        self.assertNotIn('OLD ANSWER TEXT ONLY', self.arm()['reader_input'])
        self.assertNotIn('OLD CLAIM TEXT ONLY', self.arm()['reader_input'])
        self.fixture['snapshots'][0]['rows'][0]['answer'] = 'forbidden field'
        with self.assertRaisesRegex(ValueError, 'forbidden metadata'):
            validate(**self.fixture)

    def test_prior_call_must_match_original_source_payload_and_missing_call_is_explicit(self):
        call = self.fixture['reader_proofs'][0]['rows'][0]['reader']['call']
        call['request']['input'] += 'Unrecorded prior evidence.'
        call['request_sha256'] = digest(canonical(call['request']))
        with self.assertRaisesRegex(ValueError, 'API input differs'):
            validate(**self.fixture)
        fixture = make_fixture(self.root)
        del fixture['reader_proofs'][0]['rows'][0]['reader']['call']
        self.assertEqual(validate(**fixture)['prior_reader_proofs']['missing_original_call_records'], 1)

    def test_counterevidence_and_source_bytes_are_replayed_without_old_answer_queries(self):
        self.fixture['report']['rows'][0]['counterevidence_group_ids'].reverse()
        with self.assertRaisesRegex(ValueError, 'question-only counterevidence'):
            validate(**self.fixture)
        fixture = make_fixture(self.root)
        self.arm(fixture)['source_groups'][0]['sources'][0]['text'] = 'Changed source bytes.'
        with self.assertRaisesRegex(ValueError, 'source selection or original bytes'):
            validate(**fixture)

    def test_exact_count_and_paid_cache_records_cannot_be_rewritten(self):
        self.arm()['framing_checks'][0]['response']['input_tokens'] += 1
        with self.assertRaisesRegex(ValueError, 'frozen RPC cache'):
            validate(**self.fixture)
        fixture = make_fixture(self.root)
        self.arm(fixture)['reader']['call']['raw_text'] += 'changed'
        with self.assertRaises(ValueError):
            validate(**fixture)

    def test_forged_100_and_pending_stages_never_meet_acceptance(self):
        arm = self.arm()
        arm['judge'].update(raw_text='no', score=False)
        call = arm['judge']['call']
        call['raw_text'] = call['response']['text'] = 'no'
        self.fixture['cached_calls'][call['cache_key']] = copy.deepcopy(call)
        with self.assertRaisesRegex(ValueError, 'score mismatch: correct'):
            validate(**self.fixture)
        arm['scores'] = recompute_scores(arm)
        with self.assertRaisesRegex(ValueError, 'summary mismatch: correct'):
            validate(**self.fixture)
        refresh_summary(self.fixture)
        self.assertEqual(validate(**self.fixture)['summary']['answer_accuracy'], 0.99)
        self.assertFalse(self.fixture['report']['summary']['acceptance_met'])
        arm['reader']['status'] = 'pending'
        with self.assertRaisesRegex(ValueError, 'pending generation'):
            validate(**self.fixture)

    def test_frozen_model_rates_effort_cache_mode_and_14_dollar_cap_are_checked(self):
        for corruption in ('rates', 'effort', 'cache_mode', 'cap', 'duplicate'):
            with self.subTest(corruption=corruption):
                fixture = copy.deepcopy(self.fixture)
                if corruption == 'rates':
                    fixture['protocol']['rates_usd_per_million']['reader']['input'] = 1.0
                elif corruption in ('effort', 'cache_mode'):
                    request = self.arm(fixture)['reader']['call']['request']
                    request['reasoning' if corruption == 'effort' else 'prompt_cache_options'] = {'effort': 'medium'} if corruption == 'effort' else {'mode': 'implicit'}
                elif corruption == 'cap':
                    fixture['ledger']['entries'][2]['cost_usd'] = 14.1
                    update_spend(fixture)
                else:
                    extra = copy.deepcopy(fixture['ledger']['entries'][2])
                    extra['cache_key'] = 'resampled-cache'
                    fixture['ledger']['entries'].append(extra)
                    update_spend(fixture)
                with self.assertRaises(ValueError):
                    validate(**fixture)

    def test_exact_preflight_boundary_includes_failure_in_all_100_questions(self):
        for value in (32255, 32256, 32257):
            with self.subTest(exact_count=value):
                fixture = copy.deepcopy(self.fixture)
                arm = self.arm(fixture)
                check = arm['framing_checks'][0]
                check['response']['input_tokens'] = value
                fixture['input_counts'][check['cache_key']] = copy.deepcopy(check)
                arm['input_tokens'] = arm['reader_cap']['input_tokens'] = value
                fixture['protocol']['prepared_inputs'][0]['input_tokens'] = value
                if value <= 32256:
                    call = arm['reader']['call']
                    reserve = ((value + 512) * 5 + 4096 * 10) / 1e6
                    call['reserved_usd'] = reserve
                    fixture['cached_calls'][call['cache_key']] = copy.deepcopy(call)
                    next(entry for entry in fixture['ledger']['entries'] if entry['cache_key'] == call['cache_key'])['reserved_usd'] = reserve
                else:
                    for stage in ('reader', 'judge'):
                        key = arm[stage]['call']['cache_key']
                        fixture['cached_calls'].pop(key)
                        fixture['ledger']['entries'] = [entry for entry in fixture['ledger']['entries'] if entry['cache_key'] != key]
                    arm['reader_cap'].update(selection_failure=True, failure_reason='exact_input_count_exceeds_cap')
                    arm['reader'] = {'status': 'selection_failure', 'raw_text': '', 'parsed': None}
                    arm['judge'] = {'status': 'reader_failure', 'raw_text': '', 'score': False}
                refresh_summary(fixture)
                result = validate(**fixture)
                self.assertEqual(result['summary']['n'], 100)
                self.assertEqual(result['summary']['correct'], 99 if value > 32256 else 100)
                self.assertEqual(result['summary']['acceptance_met'], value <= 32256)

    def test_unexpected_cache_writes_are_charged_and_count_as_one_reader_failure(self):
        for include_classes in (True, False):
            with self.subTest(write_classes_present=include_classes):
                fixture = copy.deepcopy(self.fixture)
                arm = self.arm(fixture)
                call = arm['reader']['call']
                details = call['usage']['input_tokens_details']
                details['cache_write_tokens'] = 10
                if include_classes:
                    details['cache_write_tokens_details'] = {'ttl_30m_tokens': 10, 'ttl_12h_tokens': 0}
                else:
                    del details['cache_write_tokens_details']
                cost, conservative, _ = fee(call['usage'], 'reader')
                call.update(status='validation_failure', cost_usd=cost, cost_is_conservative=conservative,
                            unexpected_cache_writes=True, validation_failures=['unexpected_explicit_mode_cache_writes'])
                call['response']['usage'] = copy.deepcopy(call['usage'])
                arm['reader'].update(status='error', parsed=None)
                fixture['cached_calls'][call['cache_key']] = copy.deepcopy(call)
                entry = next(entry for entry in fixture['ledger']['entries'] if entry['cache_key'] == call['cache_key'])
                entry.update(status=call['status'], usage=copy.deepcopy(call['usage']), cost_usd=cost)
                judge_key = arm['judge']['call']['cache_key']
                fixture['cached_calls'].pop(judge_key)
                fixture['ledger']['entries'] = [entry for entry in fixture['ledger']['entries'] if entry['cache_key'] != judge_key]
                arm['judge'] = {'status': 'reader_failure', 'raw_text': '', 'score': False}
                refresh_summary(fixture)
                result = validate(**fixture)
                self.assertEqual(result['paid_calls_verified'], 199)
                self.assertEqual(result['summary']['correct'], 99)
                self.assertFalse(result['summary']['acceptance_met'])

    def test_transport_retries_preserve_each_failed_charge_and_same_100_outcomes(self):
        before = self.fixture['report']['incremental_spend_usd']
        for stage in ('reader', 'judge'):
            for count in (0, 1, 2):
                with self.subTest(stage=stage, retries=count):
                    fixture = copy.deepcopy(self.fixture)
                    calls = transport_chain(fixture, [{'http_status': 520, 'error_type': 'InternalServerError'}] * count, stage)
                    result = validate(**fixture)
                    self.assertEqual(result['paid_calls_verified'], 200 + count)
                    self.assertAlmostEqual(result['incremental_spend_usd'], before + count * calls[0]['reserved_usd'])
                    self.assertEqual(result['summary']['correct'], 100)
                    self.assertTrue(result['summary']['acceptance_met'])
                    self.assertTrue(all(call['request'] == calls[-1]['request'] for call in calls))
        fixture = copy.deepcopy(self.fixture)
        transport_chain(fixture, [{'http_status': 520}] * 3)
        with self.assertRaisesRegex(ValueError, 'transport retry limit'):
            validate(**fixture)

    def test_exhausted_transport_retries_still_leave_one_failed_question_of_100(self):
        for stage in ('reader', 'judge'):
            with self.subTest(stage=stage):
                fixture = copy.deepcopy(self.fixture)
                calls = transport_chain(fixture, [{'error_type': 'APITimeoutError', 'http_status': None}] * 3, stage, exhausted=True)
                result = validate(**fixture)
                self.assertEqual(result['summary']['n'], 100)
                self.assertEqual(result['summary']['correct'], 99)
                self.assertEqual(result['summary']['answer_accuracy'], 0.99)
                self.assertFalse(result['summary']['acceptance_met'])
                self.assertEqual(result['paid_calls_verified'], 201 if stage == 'reader' else 202)
                self.assertEqual(len(calls), 3)
        fixture = copy.deepcopy(self.fixture)
        transport_chain(fixture, [{'http_status': 520}], exhausted=True)
        with self.assertRaisesRegex(ValueError, 'stopped before frozen retry limit'):
            validate(**fixture)

    def test_budget_stop_after_transport_failure_charges_only_issued_attempt(self):
        failed = transport_chain(self.fixture, [{'http_status': 520}], exhausted=True)[0]
        arm = self.arm()
        final = {key: copy.deepcopy(failed[key]) for key in
                 ('request', 'request_sha256', 'protocol_sha256', 'raw_text', 'response', 'usage')}
        final.update(call_id=failed['call_id'] + ':transport-retry:1', status='budget_exhausted', cost_usd=0,
                     budget_scope='new_variant_spend_cap')
        final['cache_key'] = digest(canonical({'namespace': 'sol61-verifier-call-v1', 'request': final['request'],
                                               'call_id': final['call_id'], 'protocol_sha256': PROTOCOL_SHA}))
        arm['reader'].update(call=final, transport_attempts=[failed])
        result = validate(**self.fixture)
        self.assertEqual(result['paid_calls_verified'], 199)
        self.assertEqual(result['summary']['n'], 100)
        self.assertEqual(result['summary']['correct'], 99)
        self.assertNotIn(final['cache_key'], self.fixture['cached_calls'])

    def test_transport_retry_allowlist_has_status_boundaries_and_no_returned_outputs(self):
        for status, allowed in ((428, False), (429, True), (430, False), (499, False), (500, True), (599, True), (600, False)):
            with self.subTest(http_status=status):
                fixture = copy.deepcopy(self.fixture)
                transport_chain(fixture, [{'http_status': status, 'error_type': 'APIStatusError'}])
                if allowed:
                    self.assertEqual(validate(**fixture)['paid_calls_verified'], 201)
                else:
                    with self.assertRaisesRegex(ValueError, 'disallowed transport failure'):
                        validate(**fixture)
        for error_type in ('APIConnectionError', 'APITimeoutError'):
            fixture = copy.deepcopy(self.fixture)
            transport_chain(fixture, [{'http_status': None, 'error_type': error_type}])
            self.assertEqual(validate(**fixture)['paid_calls_verified'], 201)
        for corruption in ('completed', 'incomplete', 'validation_failure', 'invalid_json', 'usage', 'raw_text'):
            with self.subTest(corruption=corruption):
                fixture = copy.deepcopy(self.fixture)
                earlier = transport_chain(fixture, [{'http_status': 520}])[0]
                if corruption in ('completed', 'incomplete', 'validation_failure', 'invalid_json'):
                    earlier['status'] = 'ok' if corruption in ('completed', 'invalid_json') else corruption
                    earlier['response'] = {'status': 'completed' if corruption != 'incomplete' else 'incomplete', 'text': 'no'}
                    earlier['raw_text'] = 'invalid JSON' if corruption == 'invalid_json' else 'no'
                elif corruption == 'usage':
                    earlier['usage'] = {'input_tokens': 10, 'output_tokens': 1}
                else:
                    earlier['raw_text'] = 'received output'
                with self.assertRaisesRegex(ValueError, 'returned response or disallowed'):
                    validate(**fixture)

    def test_retry_request_identity_cache_and_ledger_cannot_hide_resampling_or_spend(self):
        for corruption in ('request', 'identity', 'cache', 'ledger', 'extra_attempt', 'retry_policy'):
            with self.subTest(corruption=corruption):
                fixture = copy.deepcopy(self.fixture)
                earlier, final = transport_chain(fixture, [{'http_status': 429}])
                if corruption == 'request':
                    earlier['request']['input'] += 'Select a different answer.'
                    earlier['request_sha256'] = digest(canonical(earlier['request']))
                elif corruption == 'identity':
                    final['call_id'] = digest(final['call_id'])
                    fixture['cached_calls'][final['cache_key']] = copy.deepcopy(final)
                elif corruption == 'cache':
                    fixture['cached_calls'].pop(earlier['cache_key'])
                elif corruption == 'ledger':
                    fixture['ledger']['entries'] = [entry for entry in fixture['ledger']['entries'] if entry['cache_key'] != earlier['cache_key']]
                    update_spend(fixture)
                elif corruption == 'extra_attempt':
                    extra = copy.deepcopy(next(entry for entry in fixture['ledger']['entries'] if entry['cache_key'] == final['cache_key']))
                    extra.update(cache_key='unreported-retry-cache', call_id=final['call_id'].rsplit(':', 1)[0] + ':2')
                    fixture['ledger']['entries'].append(extra)
                    update_spend(fixture)
                else:
                    fixture['protocol']['transport_retry_policy']['max_retries'] = 3
                with self.assertRaises(ValueError):
                    validate(**fixture)


class SourceReplayChecks(unittest.TestCase):
    def setUp(self):
        date = '2024/01/02 (Tue) 09:00'

        def source(session, turn, role, text):
            return {'source_id': digest(f'source:{session}:{turn}'), 'session_id': digest(f'session:{session}'),
                    'session_index': session, 'turn_index': turn, 'role': role, 'text': text, 'observation_time': date}

        self.user = source(0, 0, 'user', 'I own a blue bicycle.')
        self.assistant = source(0, 1, 'assistant', 'A suggestion about blue bicycles.')
        self.other = source(1, 0, 'assistant', 'A complete orphan assistant observation.')
        self.groups = [{'group_id': 'exchange', 'frozen_selected_id_sha256': 'alias', 'sources': [self.user, self.assistant]},
                       {'group_id': 'orphan', 'frozen_selected_id_sha256': 'orphan-alias', 'sources': [self.other]}]
        self.question = {'question': 'What do I own?', 'question_date': date}
        self.serialize = lambda groups: canonical(reader_payload(self.question, {'source_groups': groups}))

    def test_old_alias_mapping_keeps_ambiguous_physical_sources_and_premises(self):
        legacy = dict(self.user, source_id='old-archive-source', session_id='old-archive-session', session_index=9)
        second = dict(self.user, source_id=digest('another-physical-source'), session_id=digest('another-session'), session_index=2)
        self.groups[0]['sources'].append(second)
        seed = {'source_groups': [{'frozen_selected_id_sha256': 'alias', 'sources': [legacy]}],
                'cited_source_ids': [legacy['source_id']]}
        result = replay_sources(self.groups, [seed], [], self.serialize, len, cap=10000)
        expected_ids = [self.user['source_id'], second['source_id']]
        self.assertEqual(result['diagnostics']['mapped_citations'][0]['original_source_ids'], expected_ids)
        self.assertEqual(result['diagnostics']['ambiguous_citations'][0]['original_source_ids'], expected_ids)
        self.assertEqual([source['source_id'] for source in result['source_groups'][0]['sources']], expected_ids)

    def test_direct_source_conflict_and_contradictory_prior_copies_are_not_laundered(self):
        changed = dict(self.user, text='Invented content.')
        seed = {'source_groups': [{'frozen_selected_id_sha256': 'alias', 'sources': [changed]}],
                'cited_source_ids': [self.user['source_id']]}
        result = replay_sources(self.groups, [seed], [], self.serialize, len)
        self.assertEqual(result['diagnostics']['unmapped_citations'][0]['reason'], 'prior_source_conflicts_with_original')
        seed['source_groups'].append({'frozen_selected_id_sha256': 'alias', 'sources': [self.user]})
        result = replay_sources(self.groups, [seed], [], self.serialize, len)
        self.assertEqual(result['diagnostics']['unmapped_citations'][0]['reason'], 'conflicting_prior_source_identity')

    def test_oversized_citation_is_not_clipped_or_retried_and_later_whole_turn_can_fit(self):
        self.user['text'] = 'Complete long source. ' * 1000
        seeds = [{'source_groups': [self.groups[0]], 'cited_source_ids': [self.user['source_id']]},
                 {'source_groups': [self.groups[1]], 'cited_source_ids': [self.other['source_id']]}]
        cap = len(self.serialize([self.groups[1]]))
        result = replay_sources(self.groups, seeds, ['exchange', 'orphan'], self.serialize, len, cap)
        self.assertEqual([source for group in result['source_groups'] for source in group['sources']], [self.other])
        self.assertEqual(result['diagnostics']['dropped_cited_source_ids'], [self.user['source_id']])
        self.assertNotIn(self.user['source_id'], result['diagnostics']['dropped_counterevidence_source_ids'])
        self.assertEqual(result['input_tokens'], cap)
        self.assertFalse(result['selection_failure'])

    def test_cited_assistant_admits_original_preceding_user_and_rejects_old_answer_fields(self):
        seeds = [{'source_groups': [self.groups[0]], 'cited_source_ids': [self.assistant['source_id']]}]
        result = replay_sources(self.groups, seeds, [], self.serialize, len)
        self.assertEqual(result['source_groups'][0]['sources'], [self.user, self.assistant])
        seeds[0]['answer'] = 'old answer'
        with self.assertRaisesRegex(ValueError, 'forbidden fields'):
            replay_sources(self.groups, seeds, [], self.serialize, len)


class VerifierCostChecks(unittest.TestCase):
    def test_disjoint_cache_lifetimes_are_priced_once_and_unknown_classes_fail(self):
        usage = {'input_tokens': 12000, 'output_tokens': 100,
                 'input_tokens_details': {'cached_tokens': 4000, 'cache_write_tokens': 7000,
                     'cache_write_tokens_details': {'ttl_30m_tokens': 3000, 'ttl_12h_tokens': 4000}}}
        self.assertEqual(fee(usage, 'reader'), (0.0309, False, True))
        del usage['input_tokens_details']['cache_write_tokens_details']
        self.assertEqual(fee(usage, 'reader'), (0.0384, True, True))
        usage['input_tokens_details']['cache_write_tokens_details'] = {'unsupported_lifetime': 7000}
        with self.assertRaisesRegex(ValueError, 'unknown cache write class'):
            fee(usage, 'reader')
        usage['input_tokens_details']['cached_tokens'] = 6000
        with self.assertRaisesRegex(ValueError, 'disjoint cache token counts'):
            fee(usage, 'reader')


if __name__ == '__main__':
    unittest.main()
