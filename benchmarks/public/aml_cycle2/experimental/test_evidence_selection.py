# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Claim preservation, complete witnesses, original-ID bounds and exchange optima."""

import copy
from itertools import combinations, product
import random
import unittest
from unittest.mock import patch

from .evidence_selection import assemble


class RequirementExchangeTests(unittest.TestCase):
    def test_three_premise_witness_can_replace_irrelevant_baseline_ids(self):
        baseline = [f'old{index}' for index in range(10)]
        new = ['order', 'arrival', 'event-binding']
        candidates = baseline + new
        claims = [[['old0']], [['old1']], [new]]
        result = assemble(baseline, candidates, claims, protected_claims=[0, 1])
        self.assertEqual(result['ids'], baseline[:7] + new)
        self.assertEqual(result['covered_claims'], [0, 1, 2])
        self.assertEqual(result['chosen_witnesses'][2], new)

    def test_redundant_declared_premise_can_force_an_unnecessary_source(self):
        baseline = ['dated', 'noise']
        candidates = baseline + ['followup']
        declared_pair = assemble(baseline, candidates, [[['dated', 'followup']]], budget=2)
        sufficient_singleton = assemble(baseline, candidates, [[['dated']]], budget=2)
        self.assertEqual(declared_pair['ids'], ['dated', 'followup'])
        self.assertEqual(sufficient_singleton['ids'], baseline)
        self.assertEqual(declared_pair['covered_claims'], [0])
        self.assertEqual(sufficient_singleton['covered_claims'], [0])

    def test_protected_or_replacement_requires_a_whole_alternative(self):
        claims = [[['old'], ['left', 'right']], [['left', 'right']]]
        complete = assemble(['old', 'noise'], ['old', 'noise', 'left', 'right'],
                            claims, protected_claims=[0], budget=2)
        self.assertEqual(complete['ids'], ['left', 'right'])
        self.assertEqual(complete['covered_claims'], [0, 1])
        self.assertEqual(complete['chosen_witnesses'][0], ['left', 'right'])
        partial = assemble(['old', 'noise'], ['old', 'noise', 'left', 'right'],
                           claims, protected_claims=[0], budget=1)
        self.assertEqual(partial['ids'], ['old'])
        self.assertEqual(partial['covered_claims'], [0])

    def test_protection_is_a_constraint_even_when_abandoning_it_completes_more_claims(self):
        claims = [[['a', 'b']], [['x']], [['y']]]
        candidates = ['a', 'b', 'x', 'y']
        self.assertEqual(assemble(['a', 'b'], candidates, claims, budget=2)['ids'], ['x', 'y'])
        result = assemble(['a', 'b'], candidates, claims, protected_claims=[0], budget=2)
        self.assertEqual(result['ids'], ['a', 'b'])
        self.assertEqual(result['covered_claims'], [0])

    def test_eleven_premise_witness_is_not_shortened_or_counted_as_support(self):
        candidates = [f'source{index}' for index in range(11)]
        result = assemble(candidates[:10], candidates, [[candidates]])
        self.assertEqual(result['ids'], candidates[:10])
        self.assertEqual(result['covered_claims'], [])
        with self.assertRaisesRegex(ValueError, 'whole baseline witness'):
            assemble(candidates[:10], candidates, [[candidates]], protected_claims=[0])

    def test_nine_ten_and_eleven_output_boundaries_preserve_full_witnesses(self):
        candidates = [f'source{index}' for index in range(10)]
        for budget in (9, 10):
            result = assemble(candidates, candidates, [[candidates]], budget=budget)
            self.assertEqual(len(result['ids']), budget)
            self.assertEqual(result['covered_claims'], [0] if budget == 10 else [])
        with self.assertRaisesRegex(ValueError, 'budget'):
            assemble(candidates, candidates, [[candidates]], budget=11)
        with self.assertRaisesRegex(ValueError, 'satisfy'):
            assemble(candidates, candidates, [[candidates]], protected_claims=[0], budget=9)

    def test_unprotected_empty_claims_and_zero_budget_do_not_fabricate_support(self):
        self.assertEqual(assemble([], [], [[], []], budget=0),
                         {'ids': [], 'covered_claims': [], 'chosen_witnesses': {}})
        self.assertEqual(assemble(['a'], ['a'], [[['a']]], budget=0)['covered_claims'], [])
        with self.assertRaises(ValueError):
            assemble([], [], [[[]]])
        with self.assertRaisesRegex(ValueError, 'whole baseline witness'):
            assemble([], [], [[]], protected_claims=[0])

    def test_baseline_retention_then_candidate_order_break_ties(self):
        result = assemble(['old0', 'old1'], ['new', 'old0', 'old1'],
                          [[['new'], ['old0']]], budget=2)
        self.assertEqual(result['ids'], ['old0', 'old1'])
        result = assemble(['base'], ['base', 'a', 'b'], [[['b'], ['a']]], budget=2)
        self.assertEqual(result['ids'], ['base', 'a'])
        self.assertEqual(result['chosen_witnesses'], {0: ['a']})

    def test_unused_slots_fill_only_from_original_baseline(self):
        self.assertEqual(assemble(['base'], ['base', 'unrelated'], [])['ids'], ['base'])
        self.assertEqual(assemble([], ['unrelated'], [])['ids'], [])

    def test_shared_and_duplicate_premises_are_single_original_ids(self):
        result = assemble(['a'], ['a', 'b', 'c'],
                          [[['a', 'a'], ['a']], [['a', 'b']], [['b', 'c']]],
                          protected_claims=[0], budget=3)
        self.assertEqual(result['ids'], ['a', 'b', 'c'])
        self.assertEqual(result['covered_claims'], [0, 1, 2])
        self.assertEqual(result['chosen_witnesses'][0], ['a'])

    def test_protected_claim_indices_and_baseline_support_are_validated(self):
        for protected in ([True], [-1], [1], [0, 0], ['0'], {0}, None):
            with self.subTest(protected=protected), self.assertRaises(ValueError):
                assemble(['a'], ['a'], [[['a']]], protected_claims=protected)
        with self.assertRaisesRegex(ValueError, 'whole baseline witness'):
            assemble(['a'], ['a', 'b'], [[['b']]], protected_claims=[0])

    def test_ineligible_or_malformed_ids_fail_instead_of_repairing_witnesses(self):
        for owner in ('', ' ', '\x00', 1, '\ud800', 'x' * 257, 'é' * 129):
            with self.subTest(owner=repr(owner)), self.assertRaises(ValueError):
                assemble([], [owner], [])
        for owners in (['a', 'a'], 'a', None):
            with self.subTest(owners=owners), self.assertRaises(ValueError):
                assemble([], owners, [])
        with self.assertRaises(ValueError):
            assemble(['a', 'a'], ['a'], [])
        with self.assertRaisesRegex(ValueError, 'eligible'):
            assemble(['missing'], ['a'], [])
        with self.assertRaisesRegex(ValueError, 'ineligible'):
            assemble(['a'], ['a'], [[['a', 'missing']]])
        for owner in ('x' * 255, 'x' * 256, 'é' * 128):
            self.assertEqual(assemble([owner], [owner], [])['ids'], [owner])

    def test_candidates_above_125_can_complete_a_witness(self):
        candidates = [str(index) for index in range(140)]
        result = assemble(candidates[:10], candidates, [[['0']], [['138', '139']]],
                          protected_claims=[0])
        self.assertIn('138', result['ids'])
        self.assertIn('139', result['ids'])
        self.assertEqual(result['covered_claims'], [0, 1])

    def test_candidate_claim_and_witness_boundaries(self):
        for size in (2047, 2048, 2049):
            candidates = [str(index) for index in range(size)]
            if size <= 2048:
                self.assertEqual(assemble(['0'], candidates, [])['ids'], ['0'])
            else:
                with self.assertRaises(ValueError):
                    assemble(['0'], candidates, [])
        for count in (7, 8, 9):
            if count <= 8:
                self.assertEqual(assemble(['a'], ['a'], [[['a']]] * count)['covered_claims'],
                                 list(range(count)))
            else:
                with self.assertRaises(ValueError):
                    assemble(['a'], ['a'], [[['a']]] * count)
        for count in (63, 64, 65):
            if count <= 64:
                self.assertEqual(assemble(['a'], ['a'], [[['a']] * count])['covered_claims'], [0])
            else:
                with self.assertRaises(ValueError):
                    assemble(['a'], ['a'], [[['a']] * count])
        for claims in (None, [None], [[None]]):
            with self.assertRaises(ValueError):
                assemble([], [], claims)

    def test_exact_state_bound_and_fail_loudly_above_it(self):
        for count in (5, 6):
            ids = [f'claim{claim}w{witness}' for claim in range(count) for witness in range(7)]
            claims = [[[f'claim{claim}w{witness}'] for witness in range(7)]
                      for claim in range(count)]
            if count == 5:
                result = assemble([], ids, claims)
                self.assertEqual(result['covered_claims'], list(range(count)))
                self.assertEqual(result['ids'], [f'claim{claim}w0' for claim in range(count)])
            else:
                with self.assertRaisesRegex(RuntimeError, 'state bound'):
                    assemble([], ids, claims)

    def test_transition_bound_counts_attempts_and_fails_without_partial_output(self):
        with patch(f'{assemble.__module__}.MAX_TRANSITIONS', 2):
            self.assertEqual(assemble([], ['a', 'b'], [[['a'], ['b']]])['ids'], ['a'])
        with patch(f'{assemble.__module__}.MAX_TRANSITIONS', 1):
            with self.assertRaisesRegex(RuntimeError, 'transition bound'):
                assemble([], ['a', 'b'], [[['a'], ['b']]])

    def test_original_inputs_and_semantic_links_are_not_mutated(self):
        arguments = (['a'], ['a', 'b'], [[['a'], ['a', 'b']], [['b']]], [0])
        before = copy.deepcopy(arguments)
        assemble(*arguments)
        self.assertEqual(arguments, before)

    def test_small_search_matches_exhaustive_claim_retention_and_order_optimum(self):
        ids = ['a', 'b', 'c', 'x', 'y', 'z']
        baseline = ids[:3]
        randomizer = random.Random(140)
        for case in range(64):
            claims = [[randomizer.sample(ids, randomizer.randrange(1, 4))
                       for _ in range(randomizer.randrange(4))] for _ in range(3)]
            supported = [index for index, witnesses in enumerate(claims)
                         if any(set(witness) <= set(baseline) for witness in witnesses)]
            protected = randomizer.sample(supported, randomizer.randrange(len(supported) + 1))
            choices = []
            for positions in combinations(range(len(ids)), 3):
                selected = {ids[index] for index in positions}
                covered = [index for index, witnesses in enumerate(claims)
                           if any(set(witness) <= selected for witness in witnesses)]
                if not set(protected) <= set(covered):
                    continue
                objective = (-len(covered), -len(selected & set(baseline)), positions)
                choices.append((objective, positions, covered))
            _, positions, covered = min(choices)
            with self.subTest(case=case, protected=protected):
                result = assemble(baseline, ids, claims, protected, budget=3)
                self.assertEqual(result['ids'], [ids[index] for index in positions])
                self.assertEqual(result['covered_claims'], covered)
                self.assertTrue(all(set(witness) <= set(result['ids'])
                                    for witness in result['chosen_witnesses'].values()))

    def test_short_baselines_match_exhaustive_admissible_original_id_subsets(self):
        ids = ['a', 'b', 'c', 'x', 'y', 'z']
        randomizer = random.Random(141)
        for case in range(32):
            baseline = ids[:case % 4]
            claims = [[randomizer.sample(ids, randomizer.randrange(1, 4))
                       for _ in range(randomizer.randrange(4))] for _ in range(3)]
            choices = []
            for count in range(4):
                for positions in combinations(range(len(ids)), count):
                    selected = {ids[index] for index in positions}
                    supported = [[set(witness) for witness in witnesses if set(witness) <= selected]
                                 for witnesses in claims]
                    covered = [index for index, witnesses in enumerate(supported) if witnesses]
                    admissible = False
                    for witnesses in product(*(supported[index] for index in covered)):
                        basis = set().union(*witnesses)
                        for owner in baseline:
                            if len(basis) >= 3:
                                break
                            basis.add(owner)
                        if basis == selected:
                            admissible = True
                            break
                    if admissible:
                        objective = (-len(covered), -len(selected & set(baseline)), positions)
                        choices.append((objective, positions, covered))
            _, positions, covered = min(choices)
            with self.subTest(case=case, baseline=baseline):
                result = assemble(baseline, ids, claims, budget=3)
                self.assertEqual(result['ids'], [ids[index] for index in positions])
                self.assertEqual(result['covered_claims'], covered)


if __name__ == '__main__':
    unittest.main()
