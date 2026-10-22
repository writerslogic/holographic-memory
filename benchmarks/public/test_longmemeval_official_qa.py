# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0"]
# ///
"""Official-protocol reader inputs carry no evidence labels and never cover held-out questions."""

import copy
import json
import unittest

import longmemeval_official_qa as qa


def entry():
    return {"question_id": "q", "question": "Which hostel?", "question_date": "2023/06/01 (Thu) 10:00",
            "haystack_dates": ["2023/05/02 (Tue) 09:00", "2023/04/01 (Sat) 08:00"],
            "haystack_session_ids": ["answer_s1", "s2"],
            "haystack_sessions": [
                [{"role": "user", "content": "u0", "has_answer": True},
                 {"role": "assistant", "content": "a1"},
                 {"role": "user", "content": "u2"}],
                [{"role": "user", "content": "v0"}, {"role": "assistant", "content": "b1", "has_answer": True}]]}


class OfficialPromptTest(unittest.TestCase):
    def test_rounds_are_label_free_date_sorted_and_input_is_untouched(self):
        raw = entry()
        before = copy.deepcopy(raw)
        prompt = qa.official_prompt(raw, ["answer_s1_1", "noans_s1_3", "s2_1"], topk=3)
        self.assertEqual(raw, before)
        self.assertNotIn("has_answer", prompt)
        history = prompt.split("History Chats:\n\n", 1)[1].split("\n\nCurrent Date:", 1)[0]
        self.assertLess(history.index('"v0"'), history.index('"u0"'))
        self.assertIn(json.dumps([{"role": "user", "content": "u0"}, {"role": "assistant", "content": "a1"}]), history)
        self.assertIn(json.dumps([{"role": "user", "content": "u2"}]), history)
        self.assertIn(json.dumps([{"role": "user", "content": "v0"}, {"role": "assistant", "content": "b1"}]), history)

    def test_topk_keeps_the_highest_ranked_rounds(self):
        prompt = qa.official_prompt(entry(), ["s2_1", "answer_s1_1", "noans_s1_3"], topk=1)
        self.assertIn('"v0"', prompt)
        self.assertNotIn('"u0"', prompt)

    def test_ties_ignore_the_label_bearing_id_text(self):
        ids = ["answer_s1_1", "noans_s1_3", "sharegpt_x_1", "s2_1"]
        ranks = qa._label_free_ranks({t: 1.0 for t in ids})
        relabeled = qa._label_free_ranks({t: 1.0 for t in reversed(ids)})
        self.assertEqual(ranks, relabeled)
        hashed = sorted(ids, key=lambda t: qa.hashlib.sha256(t.encode()).digest())
        self.assertEqual([t for t, _ in sorted(ranks.items(), key=lambda item: item[1])], hashed)

    def test_held_out_questions_are_refused(self):
        heldout = json.loads(qa.SPLIT.read_text())["heldout"][0]
        with self.assertRaises(ValueError):
            qa.rankings([{"question_id": heldout}], 0.0)


if __name__ == "__main__":
    unittest.main()
