# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0", "tokenizers==0.23.3", "openai==3.27.0", "python-dotenv==1.2.4"]
# ///
"""Original role occurrences and whole-exchange reader budgeting without API calls."""

import copy
import hashlib
import json
import unittest

from longmemeval_answer_evidence import reader_payload
from longmemeval_role_reader import groups_for_question, prepare_arm
from longmemeval_full_history import serialize


def handle(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def fixture():
    raw = {
        'question_id': 'synthetic-role-dev', 'question': 'What changed in my ownership?',
        'question_date': '2024/05/01 (Wed) 12:00', 'question_type': 'PRIVATE QUESTION TYPE',
        'answer': 'PRIVATE GOLD ANSWER', 'answer_session_ids': ['PRIVATE GOLD SESSION'],
        'haystack_session_ids': ['assistant_only', 'session_answer', 'empty_gap', 'session_answer'],
        'haystack_dates': ['2024/01/01 (Mon) 09:00', '2024/02/02 (Fri) 10:00',
                           '2024/03/03 (Sun) 11:00', '2024/04/04 (Thu) 12:00'],
        'haystack_sessions': [
            [{'role': 'assistant', 'content': 'An assistant-only opening.', 'has_answer': True},
             {'role': 'assistant', 'content': 'Its complete continuation.\nWith a final line.'}],
            [{'role': 'assistant', 'content': 'An initial assistant remark.'},
             {'role': 'user', 'content': 'I own a blue bicycle.', 'has_answer': True},
             {'role': 'assistant', 'content': 'First complete bicycle reply.'},
             {'role': 'assistant', 'content': 'Second complete bicycle reply.\nNo clipping.'},
             {'role': 'user', 'content': 'I later sold the bicycle.', 'has_answer': False},
             {'role': 'assistant', 'content': 'A complete reply about the sale.'}],
            [],
            [{'role': 'assistant', 'content': 'An initial remark in a repeated session.'},
             {'role': 'user', 'content': 'I own a blue bicycle.', 'has_answer': False},
             {'role': 'assistant', 'content': 'A later observation of the complete reply.'}],
        ],
    }
    historical = {'session_noans_2', 'session_answer_5'}
    question = {'qid': raw['question_id'], 'question': raw['question'],
                'question_date': raw['question_date'], 'qtype': raw['question_type'],
                'answer': raw['answer'], 'answer_session_ids': raw['answer_session_ids']}
    protocol = {'reader_prompt': 'Answer using only complete evidence.\n', 'token_cap': 131072,
                'local_token_cap': 130560, 'framing_reserve_tokens': 512}
    return raw, historical, question, protocol


def ranked(groups):
    return [dict(copy.deepcopy(group), selection_rank=index) for index, group in enumerate(groups)]


class RoleConversationGroups(unittest.TestCase):
    def setUp(self):
        self.raw, self.historical, self.question, self.protocol = fixture()

    def test_every_original_turn_has_its_exact_raw_occurrence_identity_once(self):
        before = copy.deepcopy(self.raw)
        groups, aliases, orphan_count = groups_for_question(self.raw, self.historical)
        actual = [source for group in groups for source in group['sources']]
        expected = []
        for session_index, turns in enumerate(self.raw['haystack_sessions']):
            for turn_index, turn in enumerate(turns):
                expected.append({
                    'source_id': handle(['lme-source-v1', self.raw['question_id'], session_index, turn_index]),
                    'session_id': handle(['lme-session-v1', self.raw['question_id'], session_index]),
                    'session_index': session_index, 'turn_index': turn_index,
                    'observation_time': self.raw['haystack_dates'][session_index],
                    'role': turn['role'], 'text': turn['content']})
        self.assertEqual(len(actual), 11)
        self.assertEqual(len({source['source_id'] for source in actual}), 11)
        self.assertEqual({source['source_id']: source for source in actual},
                         {source['source_id']: source for source in expected})
        self.assertEqual(set(aliases), self.historical)
        self.assertEqual(orphan_count, 3)
        self.assertEqual(sum(source['role'] == 'user' for source in actual), 3)
        self.assertEqual(sum(source['role'] == 'assistant' for source in actual), 8)
        self.assertEqual(self.raw, before)

    def test_users_keep_all_following_assistants_and_repeated_alias_observations(self):
        groups, aliases, _ = groups_for_question(self.raw, self.historical)
        bicycle = groups[aliases['session_noans_2']]
        self.assertEqual([(source['session_index'], source['turn_index'], source['role']) for source in bicycle['sources']],
                         [(1, 1, 'user'), (1, 2, 'assistant'), (1, 3, 'assistant'), (3, 1, 'user'), (3, 2, 'assistant')])
        self.assertEqual(bicycle['sources'][2]['text'], 'Second complete bicycle reply.\nNo clipping.')
        self.assertNotEqual(bicycle['sources'][0]['source_id'], bicycle['sources'][3]['source_id'])
        self.assertNotEqual(bicycle['sources'][0]['session_id'], bicycle['sources'][3]['session_id'])
        self.assertEqual(bicycle['sources'][0]['session_index'], 1)
        self.assertEqual(bicycle['sources'][3]['session_index'], 3)
        sale = groups[aliases['session_answer_5']]
        self.assertEqual([(source['turn_index'], source['role']) for source in sale['sources']],
                         [(4, 'user'), (5, 'assistant')])

    def test_orphan_assistants_remain_assistants_without_fabricated_user_turns(self):
        groups, aliases, orphan_count = groups_for_question(self.raw, self.historical)
        orphans = [group for index, group in enumerate(groups) if index not in aliases.values()]
        self.assertEqual(len(orphans), orphan_count)
        self.assertEqual([[source['session_index'] for source in group['sources']] for group in orphans], [[0, 0], [1], [3]])
        self.assertTrue(all(source['role'] == 'assistant' for group in orphans for source in group['sources']))
        arm = prepare_arm(self.question, orphans, self.protocol, len)
        self.assertEqual(len(arm['retained_sources']), 4)
        self.assertTrue(all(source['role'] == 'assistant' for source in arm['retained_sources']))

    def test_model_payload_excludes_gold_flags_aliases_question_type_and_raw_question_id(self):
        groups, _, _ = groups_for_question(self.raw, self.historical)
        before = copy.deepcopy(groups)
        arm = prepare_arm(self.question, groups, self.protocol, len)
        payload = json.loads(arm['reader_input'][len(self.protocol['reader_prompt']):])
        self.assertEqual(set(payload), {'question', 'question_date', 'evidence'})
        self.assertEqual(payload, reader_payload(self.question, {'source_groups': arm['source_groups']}))
        for source in payload['evidence']:
            self.assertEqual(set(source), {'source_id', 'session_id', 'observation_time', 'role', 'text'})
        for forbidden in ('has_answer', 'PRIVATE GOLD', 'PRIVATE QUESTION TYPE', self.raw['question_id'],
                          'session_noans_2', 'session_answer_5', 'answer_session_ids'):
            self.assertNotIn(forbidden, arm['reader_input'])
        self.assertEqual(groups, before)

    def test_frozen_membership_role_and_content_inconsistencies_are_rejected(self):
        for corruption in ('missing_alias', 'extra_alias', 'ambiguous_alias', 'role', 'content', 'session_lengths'):
            with self.subTest(corruption=corruption):
                raw = copy.deepcopy(self.raw)
                historical = self.historical.copy()
                if corruption == 'missing_alias':
                    historical.remove('session_noans_2')
                elif corruption == 'extra_alias':
                    historical.add('nonexistent_1')
                elif corruption == 'ambiguous_alias':
                    historical.add('session_answer_2')
                elif corruption == 'role':
                    raw['haystack_sessions'][0][0]['role'] = 'system'
                elif corruption == 'content':
                    raw['haystack_sessions'][0][0]['content'] = None
                else:
                    raw['haystack_dates'].pop()
                with self.assertRaises(ValueError):
                    groups_for_question(raw, historical)


class RoleReaderBudget(unittest.TestCase):
    def setUp(self):
        raw, historical, self.question, self.protocol = fixture()
        groups, aliases, _ = groups_for_question(raw, historical)
        orphan = next(group for group in groups if all(source['role'] == 'assistant' for source in group['sources']))
        self.selected = [groups[aliases['session_noans_2']], orphan, groups[aliases['session_answer_5']]]

    def test_complete_input_limit_minus_one_at_limit_plus_one_drops_whole_last_exchange(self):
        unchanged = copy.deepcopy(self.selected)
        complete = len(serialize(self.question, ranked(self.selected), self.protocol))
        for cap in (complete - 1, complete, complete + 1):
            with self.subTest(cap=cap):
                protocol = dict(self.protocol, local_token_cap=cap)
                arm = prepare_arm(self.question, self.selected, protocol, len)
                dropped = cap < complete
                expected = ranked(self.selected[:-1] if dropped else self.selected)
                self.assertEqual(arm['source_groups'], expected)
                self.assertEqual(arm['reader_cap']['dropped_group_ids'],
                                 [self.selected[-1]['group_id']] if dropped else [])
                self.assertEqual(arm['reader_cap']['uncapped_input_tokens'], complete)
                self.assertFalse(arm['reader_cap']['selection_failure'])
                self.assertLessEqual(arm['input_tokens'], cap)
                self.assertEqual(arm['input_tokens'], len(arm['reader_input']))
                for retained in arm['source_groups']:
                    original = next(group for group in self.selected if group['group_id'] == retained['group_id'])
                    self.assertEqual(retained['sources'], original['sources'])
                self.assertEqual(self.selected, unchanged)

    def test_anchor_overflow_keeps_all_roles_and_bytes_and_records_selection_failure(self):
        selected = self.selected[:1]
        original = copy.deepcopy(selected)
        complete = len(serialize(self.question, ranked(selected), self.protocol))
        for cap in (complete - 1, complete, complete + 1):
            with self.subTest(cap=cap):
                arm = prepare_arm(self.question, selected, dict(self.protocol, local_token_cap=cap), len)
                failed = cap < complete
                self.assertEqual(arm['source_groups'], ranked(selected))
                self.assertEqual(arm['reader_cap']['selection_failure'], failed)
                self.assertEqual(arm['reader']['status'], 'selection_failure' if failed else 'pending')
                self.assertEqual(arm['judge']['status'], 'pending')
                self.assertFalse(arm['scores']['correct'])
                self.assertEqual(arm['input_tokens'], complete)
                self.assertEqual(sum(source['role'] == 'user' for source in arm['retained_sources']), 2)
                self.assertEqual(sum(source['role'] == 'assistant' for source in arm['retained_sources']), 3)
                self.assertEqual(selected, original)

    def test_candidate_count_at_63_64_65_and_empty_selection(self):
        groups = []
        for index in range(65):
            source = {'source_id': handle(['source', index]), 'session_id': handle(['session', index]),
                      'session_index': index, 'turn_index': 0, 'observation_time': '2024/01/01 (Mon) 09:00',
                      'role': 'assistant', 'text': f'A complete assistant observation {index}.'}
            groups.append({'group_id': handle(['group', index]), 'frozen_selected_id_sha256': handle(['alias', index]),
                           'sources': [source]})
        for size in (63, 64):
            with self.subTest(size=size):
                arm = prepare_arm(self.question, groups[:size], self.protocol, len)
                self.assertEqual(len(arm['source_groups']), size)
                self.assertEqual(len(arm['retained_sources']), size)
                self.assertFalse(arm['reader_cap']['selection_failure'])
        for selected in ([], groups):
            with self.assertRaisesRegex(ValueError, 'invalid retrieved source count'):
                prepare_arm(self.question, selected, self.protocol, len)

    def test_prepared_sources_are_independent_of_selected_input_objects(self):
        original = copy.deepcopy(self.selected)
        arm = prepare_arm(self.question, self.selected, self.protocol, len)
        self.assertEqual(arm['source_groups'], ranked(self.selected))
        arm['source_groups'][0]['sources'][0]['text'] = 'Changed only in the returned arm.'
        self.assertEqual(self.selected, original)


if __name__ == '__main__':
    unittest.main()
