# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0"]
# ///
"""Synthetic provenance, ranking, reader, and accounting tampering checks."""

import copy
import statistics
import tempfile
import unittest
from pathlib import Path

import tiktoken

from check_longmemeval_answers import canonical, digest, recompute_scores, sorted_sources
from check_longmemeval_role_reader import (
    ARM, SYSTEM, RETRIEVAL_POLICY, ROLE_POLICY, SOURCE_GATE, cap_groups, original_groups,
    replay_retrieval, validate,
)
from check_longmemeval_user_first import check_historical_policy
from longmemeval_answer_evidence import reader_payload
from test_check_longmemeval_full_history import make_full_fixture, update_spend


def make_role_fixture(root):
    full = make_full_fixture(root)
    protocol, report = full['protocol'], full['report']
    protocol.update(source_policy=ROLE_POLICY, retrieval=copy.deepcopy(RETRIEVAL_POLICY), source_gate=SOURCE_GATE.copy())
    raw_questions = {}
    tokenizer = tiktoken.get_encoding('o200k_base')
    counts = []
    for index, row in enumerate(report['rows']):
        qid = row['qid']
        raw = {'question_id': qid, 'question': row['question'], 'question_type': row['qtype'],
               'question_date': row['question_date'], 'answer': 'GOLD ANSWER SHOULD NOT REACH THE READER',
               'haystack_session_ids': ['orphan_session', 'empty_gap', 'session_answer'],
               'haystack_dates': ['2024/01/02 (Tue) 09:00'] * 3,
               'haystack_sessions': [[{'role': 'assistant', 'content': 'Your favorite color is blue.'}], [], [
                   {'role': 'user', 'content': 'I prefer blue.', 'has_answer': True},
                   {'role': 'assistant', 'content': 'Your preference is blue.'},
                   {'role': 'assistant', 'content': 'You also like a blue color palette.'},
                   {'role': 'user', 'content': 'I own a green bicycle.'},
                   {'role': 'assistant', 'content': 'Your green bicycle has a basket.'}]]}
        raw_questions[qid] = raw
        source = lambda session, turn: {
            'source_id': digest(canonical(['lme-source-v1', qid, session, turn])),
            'session_id': digest(canonical(['lme-session-v1', qid, session])), 'session_index': session,
            'turn_index': turn, 'observation_time': raw['haystack_dates'][session],
            'role': raw['haystack_sessions'][session][turn]['role'], 'text': raw['haystack_sessions'][session][turn]['content']}
        pool = []
        for alias, start, end in (('session_answer_1', 0, 3), ('session_noans_4', 3, 5)):
            pool.append({'group_id': digest(canonical(['lme-group-v1', qid, [[2, start]]])),
                         'frozen_selected_id_sha256': digest(alias), 'sources': [source(2, turn) for turn in range(start, end)]})
        orphan = source(0, 0)
        orphan_id = digest(canonical(['lme-assistant-group-v1', qid, [orphan['source_id']]]))
        pool.append({'group_id': orphan_id, 'frozen_selected_id_sha256': orphan_id, 'sources': [orphan]})
        selected = [dict(copy.deepcopy(pool[group]), selection_rank=rank) for rank, group in enumerate((1, 0, 2))]
        row.update(candidate_groups=3, conversation_groups=3, orphan_assistant_groups=1,
                   retrieval={'selected': [1, 0, 2], 'assistant_ranking': [2, 0], 'fused_ranking': [0, 1, 2],
                              'reserved_user_prior': [1, 0], 'rrf_k0': 60})
        row.pop('known_assistant_groups')
        arm = row['systems'][SYSTEM]['arms'][ARM]
        prompt = protocol['reader_prompt'] + canonical(reader_payload(row, {'source_groups': selected}))
        count = len(tokenizer.encode(prompt))
        counts.append(count)
        arm.update(source_groups=selected, retained_sources=sorted_sources(selected), reader_input=prompt,
                   reader_input_sha256=digest(prompt), input_tokens=count,
                   reader_cap={'cap': 130560, 'input_tokens': count, 'uncapped_input_tokens': count,
                               'dropped_group_ids': [], 'selection_failure': False, 'failure_reason': None})
        full['history_rows'][index].update(ids=['session_answer_1', 'session_noans_4'],
                                          baseline_ranking=['session_noans_4', 'session_answer_1'])
        for stage in ('reader', 'judge'):
            call = arm[stage]['call']
            if stage == 'reader':
                call['request']['input'] = prompt
            request = call['request']
            usage = {'input_tokens': len(tokenizer.encode(request['input'])) + 10, 'output_tokens': 20,
                     'input_tokens_details': {'cached_tokens': 0}}
            call_id = digest(f"full-history:{full['work_name']}:{stage}:{qid}")
            key = digest(canonical({'request': request, 'call_id': call_id, 'protocol_sha256': full['official_sha256']}))
            cost = (usage['input_tokens'] * 2.5 + usage['output_tokens'] * 15) / 1_000_000
            call.update(request_sha256=digest(canonical(request)), cache_key=key, usage=usage, cost_usd=cost)
            call['response']['usage'] = copy.deepcopy(usage)
            entry = next(entry for entry in full['ledger']['entries'] if entry.get('call_id') == call_id)
            entry.update(cache_key=key, usage=copy.deepcopy(usage), cost_usd=cost,
                         reserved_usd=((len(tokenizer.encode(request['input'])) + 512) * 2.5
                                       + request['max_output_tokens'] * 15) / 1_000_000)
    protocol['prepared_inputs'] = [{'qid': row['qid'],
                                    'reader_input_sha256': row['systems'][SYSTEM]['arms'][ARM]['reader_input_sha256'],
                                    'input_tokens': row['systems'][SYSTEM]['arms'][ARM]['input_tokens']} for row in report['rows']]
    report['summary'].update(input_tokens={'median': statistics.median(counts), 'max': max(counts)},
                             user_occurrences=200, assistant_occurrences=400)
    update_spend(full)
    result = {key: full[key] for key in ('report', 'split', 'protocol', 'base_protocol', 'preregistration',
                                        'baseline_rows', 'history_rows', 'references', 'ledger', 'work_name',
                                        'official_sha256', 'rules')}
    result['raw_questions'] = raw_questions
    return result


class RoleReaderChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = make_role_fixture(self.root)

    def arm(self, fixture=None):
        return (fixture or self.fixture)['report']['rows'][0]['systems'][SYSTEM]['arms'][ARM]

    def test_completed_role_run_replays_original_sources_ranking_calls_and_own_spend(self):
        before = copy.deepcopy(self.fixture)
        result = validate(**self.fixture)
        self.assertEqual(result['paid_calls_verified'], 200)
        self.assertEqual(result['summary']['n'], 100)
        self.assertEqual(result['summary']['user_occurrences'], 200)
        self.assertEqual(result['summary']['assistant_occurrences'], 400)
        self.assertTrue(result['summary']['acceptance_met'])
        self.assertEqual(result['source_gate']['heldout_records_decoded'], 0)
        self.assertNotEqual(result['incremental_spend_usd'],
                            result['global_recorded_spend_usd'] - self.fixture['report']['starting_spend_usd'])
        self.assertEqual(before, self.fixture)

    def test_raw_source_replay_uses_original_indices_keeps_all_roles_and_matches_known_ranking(self):
        qid = self.fixture['split']['dev'][0]
        groups, aliases, orphans = original_groups(self.fixture['raw_questions'][qid], {'session_answer_1', 'session_noans_4'})
        self.assertEqual(aliases, {'session_answer_1': 0, 'session_noans_4': 1})
        self.assertEqual(orphans, 1)
        self.assertEqual([source['role'] for source in groups[0]['sources']], ['user', 'assistant', 'assistant'])
        self.assertEqual([source['session_index'] for source in groups[0]['sources']], [2, 2, 2])
        self.assertEqual(groups[2]['sources'][0]['role'], 'assistant')
        self.assertEqual(groups[2]['sources'][0]['session_index'], 0)
        self.assertEqual(replay_retrieval(self.fixture['report']['rows'][0]['question'], groups, [1, 0]),
                         {'selected': [1, 0, 2], 'assistant_ranking': [2, 0], 'fused_ranking': [0, 1, 2],
                          'reserved_user_prior': [1, 0], 'rrf_k0': 60})

    def test_assistant_text_omission_role_loss_and_fabricated_metadata_are_rejected(self):
        for corruption in ('omitted_assistant', 'role', 'source_id', 'date'):
            with self.subTest(corruption=corruption):
                fixture = copy.deepcopy(self.fixture)
                group = self.arm(fixture)['source_groups'][1]
                if corruption == 'omitted_assistant':
                    group['sources'].pop()
                elif corruption == 'role':
                    group['sources'][1]['role'] = 'user'
                elif corruption == 'source_id':
                    group['sources'][0]['source_id'] = 'fabricated'
                else:
                    group['sources'][0]['observation_time'] = '2024/01/03 (Wed) 09:00'
                with self.assertRaisesRegex(ValueError, 'original conversation replay'):
                    validate(**fixture)

    def test_retrieval_order_and_reserved_user_prior_cannot_be_rewritten(self):
        for field in ('selected', 'assistant_ranking', 'fused_ranking', 'reserved_user_prior'):
            with self.subTest(field=field):
                fixture = copy.deepcopy(self.fixture)
                fixture['report']['rows'][0]['retrieval'][field].reverse()
                with self.assertRaisesRegex(ValueError, 'retrieval replay mismatch'):
                    validate(**fixture)
        self.fixture['protocol']['retrieval']['assistant_weight'] = 2
        with self.assertRaisesRegex(ValueError, 'ranking policy changed'):
            validate(**self.fixture)

    def test_whole_exchange_cap_boundaries_preserve_complete_pairs_and_orphan_roles(self):
        row = self.fixture['report']['rows'][0]
        selected = [{key: value for key, value in group.items() if key != 'selection_rank'} for group in self.arm()['source_groups']]
        protocol = self.fixture['protocol']
        complete = len(protocol['reader_prompt'] + canonical(reader_payload(row, {'source_groups': selected})))
        for limit in (complete - 1, complete, complete + 1):
            with self.subTest(limit=limit):
                kept, _, cap = cap_groups(row, selected, dict(protocol, local_token_cap=limit), len)
                self.assertEqual(len(kept), 2 if limit < complete else 3)
                self.assertEqual(cap['dropped_group_ids'], [selected[-1]['group_id']] if limit < complete else [])
                self.assertFalse(cap['selection_failure'])
                for group in kept:
                    self.assertEqual(group['sources'], next(original['sources'] for original in selected if original['group_id'] == group['group_id']))
        anchor = selected[:1]
        size = len(protocol['reader_prompt'] + canonical(reader_payload(row, {'source_groups': anchor})))
        kept, _, cap = cap_groups(row, anchor, dict(protocol, local_token_cap=size - 1), len)
        self.assertTrue(cap['selection_failure'])
        self.assertEqual(kept[0]['sources'], anchor[0]['sources'])

    def test_forged_100_correct_fails_raw_judge_then_independent_summary_checks(self):
        arm = self.arm()
        arm['judge'].update(raw_text='no', score=False)
        arm['judge']['call']['raw_text'] = 'no'
        arm['judge']['call']['response']['text'] = 'no'
        with self.assertRaisesRegex(ValueError, 'score mismatch: correct'):
            validate(**self.fixture)
        arm['scores'] = recompute_scores(arm)
        with self.assertRaisesRegex(ValueError, 'summary mismatch: correct'):
            validate(**self.fixture)

    def test_pending_stages_and_missing_dev_rows_cannot_claim_completion(self):
        self.arm()['reader']['status'] = 'pending'
        with self.assertRaisesRegex(ValueError, 'pending stage'):
            validate(**self.fixture)
        self.fixture['report']['rows'].pop()
        with self.assertRaisesRegex(ValueError, '100 dev rows'):
            validate(**self.fixture)

    def test_gold_in_reader_prompt_and_changed_coverage_are_rejected(self):
        self.arm()['reader_input'] += 'The gold answer is blue.'
        with self.assertRaisesRegex(ValueError, 'reader prompt differs'):
            validate(**self.fixture)
        fixture = make_role_fixture(self.root)
        fixture['report']['rows'][0]['actual_complete'] = False
        with self.assertRaisesRegex(ValueError, 'labeled coverage mismatch'):
            validate(**fixture)

    def test_official_judge_model_context_and_spend_tampering_are_rejected(self):
        for corruption in ('judge_prompt', 'model', 'context', 'spend', 'duplicate_call'):
            with self.subTest(corruption=corruption):
                fixture = copy.deepcopy(self.fixture)
                arm = self.arm(fixture)
                if corruption in ('judge_prompt', 'model'):
                    call = arm['judge']['call']
                    call['request']['input' if corruption == 'judge_prompt' else 'model'] = 'wrong value'
                    call['request_sha256'] = digest(canonical(call['request']))
                elif corruption == 'context':
                    call = arm['reader']['call']
                    call['usage']['input_tokens'] = 131073
                    call['response']['usage'] = copy.deepcopy(call['usage'])
                    call['cost_usd'] = (131073 * 2.5 + call['usage']['output_tokens'] * 15) / 1_000_000
                elif corruption == 'spend':
                    fixture['ledger']['entries'][2]['cost_usd'] += 0.01
                    update_spend(fixture)
                else:
                    extra = copy.deepcopy(fixture['ledger']['entries'][2])
                    extra['cache_key'] = 'resampled-cache-identity'
                    fixture['ledger']['entries'].append(extra)
                    update_spend(fixture)
                with self.assertRaises(ValueError):
                    validate(**fixture)

    def test_source_gate_and_frozen_inputs_cannot_authorize_heldout_decoding(self):
        self.fixture['protocol']['source_gate']['heldout_records_decoded'] = 1
        with self.assertRaisesRegex(ValueError, 'dev-only source gate'):
            validate(**self.fixture)
        fixture = make_role_fixture(self.root)
        fixture['protocol']['prepared_inputs'][0]['input_tokens'] -= 1
        with self.assertRaisesRegex(ValueError, 'prepared input manifest'):
            validate(**fixture)

    def test_old_preregistered_protocol_linkage_remains_valid_after_path_cleanup(self):
        baseline = {'protocol_sha256': '0' * 64, 'rows': self.fixture['baseline_rows']}
        result = check_historical_policy(self.fixture['base_protocol'], self.fixture['preregistration'],
                                         baseline, self.fixture['references'])
        self.assertEqual(result['reader_inputs_verified'], 100)
        self.assertEqual(result['judge_requests_verified'], 100)
        self.assertNotEqual(self.fixture['official_sha256'], result['preregistered_protocol_sha256'])


if __name__ == '__main__':
    unittest.main()
