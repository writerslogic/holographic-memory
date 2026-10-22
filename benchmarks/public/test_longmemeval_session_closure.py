# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Session-context retrieval without identity labels or mutable input state."""

import copy
import unittest

from longmemeval_session_closure import closure_candidates


def session(*tids):
    return {"sid": "opaque-session", "date": "2023/05/30 (Tue) 09:54",
            "turns": [[tid, f"Verbatim observation for {tid}."] for tid in tids]}


class SessionClosureTests(unittest.TestCase):
    def test_closing_thanks_recovers_the_initial_project_premise(self):
        source = session("opening", "project", "detail", "thanks")
        source["turns"][1][1] = "I am working on a solo project for my class."
        source["turns"][-1][1] = "Thanks for helping me present my findings."
        self.assertEqual(closure_candidates([source], ["thanks"]), ["thanks", "opening", "project"])

    def test_top_twenty_precede_openings_and_remaining_baseline(self):
        source = session("opening", "premise", *(f"hit-{i}" for i in range(25)))
        baseline = [f"hit-{i}" for i in range(25)]
        self.assertEqual(closure_candidates([source], baseline),
                         baseline[:20] + ["opening", "premise"] + baseline[20:])

    def test_seed_sessions_follow_first_hit_order_including_repeated_observations(self):
        sessions = [session("first-a", "first-b", "shared-hit"),
                    session("second-a", "second-b", "shared-hit"),
                    session("third-a", "third-b", "third-hit")]
        self.assertEqual(closure_candidates(sessions, ["third-hit", "shared-hit"]),
                         ["third-hit", "shared-hit", "third-a", "third-b",
                          "first-a", "first-b", "second-a", "second-b"])

    def test_repeated_aliases_are_returned_once_without_losing_their_session_context(self):
        sessions = [session("shared-opening", "premise-a", "hit"),
                    session("shared-opening", "premise-b", "hit")]
        result = closure_candidates(sessions, ["hit"])
        self.assertEqual(result, ["hit", "shared-opening", "premise-a", "premise-b"])
        self.assertEqual(len(result), len(set(result)))
        self.assertEqual([sum(tid == "shared-opening" for tid, _ in source["turns"])
                          for source in sessions], [1, 1])

    def test_sessions_without_top_twenty_seeds_do_not_add_openings(self):
        sources = [session("early-open", "early-premise", *(f"hit-{i}" for i in range(20))),
                   session("late-open", "late-premise", "late-hit")]
        baseline = [f"hit-{i}" for i in range(20)] + ["late-hit"]
        self.assertEqual(closure_candidates(sources, baseline),
                         baseline[:20] + ["early-open", "early-premise", "late-hit"])

    def test_candidate_limits_at_sixty_three_sixty_four_and_sixty_five(self):
        source = session(*(f"turn-{i}" for i in range(66)))
        baseline = [f"turn-{i}" for i in range(2, 66)]
        expected = baseline[:20] + ["turn-0", "turn-1"] + baseline[20:]
        for limit in (63, 64):
            with self.subTest(limit=limit):
                self.assertEqual(closure_candidates([source], baseline, limit), expected[:limit])
        with self.assertRaisesRegex(ValueError, "limit"):
            closure_candidates([source], baseline, 65)

    def test_small_limits_preserve_the_original_anchor_and_order(self):
        for limit in (1, 2, 3):
            with self.subTest(limit=limit):
                self.assertEqual(closure_candidates([session("a", "b", "hit")], ["hit"], limit),
                                 ["hit", "a", "b"][:limit])

    def test_arbitrary_bijective_alias_renaming_preserves_selection(self):
        sources = [session("answer-opening", "noans-premise", "hit-a"),
                   session("ordinary-opening", "ordinary-premise", "hit-b")]
        baseline = ["hit-b", "hit-a"]
        aliases = {tid: f"{'answer' if index % 2 else 'noans'}-{index}-opaque"
                   for index, (tid, _) in enumerate(turn for source in sources for turn in source["turns"])}
        renamed = copy.deepcopy(sources)
        for source in renamed:
            for turn in source["turns"]:
                turn[0] = aliases[turn[0]]
        result = closure_candidates(sources, baseline)
        self.assertEqual(closure_candidates(renamed, [aliases[tid] for tid in baseline]),
                         [aliases[tid] for tid in result])

    def test_input_is_unchanged_on_success_and_rejection(self):
        sources, baseline = [session("opening", "premise", "hit")], ["hit"]
        original = copy.deepcopy((sources, baseline))
        closure_candidates(sources, baseline)
        self.assertEqual((sources, baseline), original)
        with self.assertRaises(ValueError):
            closure_candidates(sources, baseline, True)
        self.assertEqual((sources, baseline), original)

    def test_rejects_unknown_duplicate_or_malformed_baseline_candidates(self):
        source = session(*(f"turn-{i}" for i in range(65)))
        invalid = ([], ["turn-0", "turn-0"], ["unknown"], [True], [""], "turn-0",
                   [f"turn-{i}" for i in range(65)])
        for baseline in invalid:
            with self.subTest(baseline=baseline), self.assertRaises(ValueError):
                closure_candidates([source], baseline)
        for limit in (0, -1, 65, True, 1.0, "64"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                closure_candidates([source], ["turn-0"], limit)

    def test_rejects_malformed_sessions_without_interpreting_alias_contents(self):
        invalid = ([], "sessions", [None], [{}], [{"turns": []}],
                   [{"turns": [["hit", 1]]}], [{"turns": [["hit"]]}],
                   [{"turns": [["", "text"]]}], [{"turns": [[True, "text"]]}])
        for sources in invalid:
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                closure_candidates(sources, ["hit"])


if __name__ == "__main__":
    unittest.main()
