# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Raw citation replay, context closure and whole-turn budget regressions."""

import copy
import json
import unittest

from longmemeval_answer_evidence import reader_payload
from longmemeval_verifier_sources import select_verifier_sources


def source(si, ti, role="user", text="Verbatim fact.", sid=None):
    return {"source_id": sid or f"raw-{si}-{ti}", "session_id": f"session-{si}",
            "session_index": si, "turn_index": ti, "observation_time": "2023/05/30 (Tue) 09:54",
            "role": role, "text": text}


def group(gid, *members, alias=None):
    return {"group_id": gid, "frozen_selected_id_sha256": alias or f"opaque-{gid}",
            "sources": list(members)}


def seed(groups, *citations):
    return {"source_groups": groups, "cited_source_ids": list(citations)}


def serialize(groups):
    payload = reader_payload({"question": "Which facts apply?", "question_date": "2023/05/30 (Tue) 10:00"},
                             {"source_groups": groups})
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def ids(result):
    return [s["source_id"] for g in result["source_groups"] for s in g["sources"]]


class VerifierSourceTests(unittest.TestCase):
    def select(self, groups, seeds=None, counter=None, cap=24000, **kwargs):
        return select_verifier_sources(groups, [] if seeds is None else seeds,
                                       [] if counter is None else counter,
                                       kwargs.get("serialize_callback", serialize),
                                       kwargs.get("count_callback", len), cap)

    def test_original_citation_retains_exact_turn_and_preceding_opening_premises(self):
        members = [source(0, 0), source(0, 2), source(0, 4), source(0, 5, "assistant"), source(0, 6)]
        original = [group("g", *members)]
        result = self.select(original, [seed(original, "raw-0-5")])
        self.assertEqual(ids(result), ["raw-0-0", "raw-0-2", "raw-0-4", "raw-0-5"])
        self.assertEqual(result["diagnostics"]["mapped_citations"][0]["original_source_ids"], ["raw-0-5"])
        self.assertFalse(result["selection_failure"])

    def test_old_geometry_ids_map_through_alias_and_exact_role_text_date(self):
        raw = source(3, 7, "assistant", "  café\nA numeric fact.  ")
        original = [group("raw-group", source(3, 6), raw, alias="opaque-shared")]
        old = source(9, 1, "assistant", raw["text"], "geometry-source")
        prior = [group("old-group", old, alias="opaque-shared")]
        result = self.select(original, [seed(prior, old["source_id"])])
        self.assertIn(raw["source_id"], ids(result))
        self.assertNotIn(old["source_id"], ids(result))
        self.assertEqual(next(s for g in result["source_groups"] for s in g["sources"]
                              if s["source_id"] == raw["source_id"]), raw)

    def test_ambiguous_alias_content_preserves_all_physical_occurrences(self):
        a, b = source(0, 1, "assistant", "Same fact."), source(1, 1, "assistant", "Same fact.")
        original = [group("both", source(0, 0), a, source(1, 0), b, alias="shared")]
        old = source(4, 9, "assistant", "Same fact.", "old")
        result = self.select(original, [seed([group("old", old, alias="shared")], "old")])
        self.assertEqual(set(ids(result)), {"raw-0-0", "raw-0-1", "raw-1-0", "raw-1-1"})
        self.assertEqual(result["diagnostics"]["ambiguous_citations"][0]["original_source_ids"],
                         ["raw-0-1", "raw-1-1"])

    def test_all_readers_citation_union_precedes_context_at_exact_budget_boundaries(self):
        a, b = source(0, 1, "assistant", "First fact."), source(1, 1, "assistant", "Second fact.")
        original = [group("a", source(0, 0, text="A" * 2000), a),
                    group("b", source(1, 0, text="B" * 2000), b)]
        seeds = [seed([original[0]], a["source_id"]), seed([original[1]], b["source_id"])]
        exact = len(serialize([group("a", a), group("b", b)]))
        for delta in (-1, 0, 1):
            with self.subTest(delta=delta):
                result = self.select(original, seeds, cap=exact + delta)
                self.assertLessEqual(result["input_tokens"], exact + delta)
                self.assertIn(a["source_id"], ids(result))
                self.assertEqual(b["source_id"] in ids(result), delta >= 0)
                self.assertNotIn("raw-0-0", ids(result))
                self.assertNotIn("raw-1-0", ids(result))
                self.assertEqual(result["diagnostics"]["dropped_cited_source_ids"],
                                 [b["source_id"]] if delta < 0 else [])

    def test_oversized_citation_reports_actual_and_prior_ids_and_keeps_later_small_fact(self):
        large = source(0, 0, "assistant", "X" * 2000)
        small = source(1, 0, "assistant", "Small fact.")
        original = [group("large", large, alias="large-alias"), group("small", small)]
        old = {**large, "source_id": "old-large", "session_id": "legacy", "session_index": 9}
        seeds = [seed([group("old", old, alias="large-alias"), original[1]], "old-large", small["source_id"])]
        cap = len(serialize([original[1]]))
        result = self.select(original, seeds, cap=cap)
        self.assertEqual(ids(result), [small["source_id"]])
        self.assertEqual(result["diagnostics"]["dropped_cited_source_ids"], [large["source_id"]])
        dropped = result["diagnostics"]["dropped_prior_citations"][0]
        self.assertEqual((dropped["cited_source_id"], dropped["dropped_original_source_ids"]),
                         ("old-large", [large["source_id"]]))

    def test_unknown_fake_and_invalid_citations_are_explicitly_unmapped(self):
        original = [group("g", source(0, 0))]
        result = self.select(original, [seed([], "unknown", "raw-0-0", None, ["bad"] )])
        self.assertEqual(ids(result), [])
        reasons = [item["reason"] for item in result["diagnostics"]["unmapped_citations"]]
        self.assertEqual(reasons.count("citation_not_in_prior_packet"), 2)
        self.assertEqual(reasons.count("invalid_citation_identity"), 2)
        self.assertTrue(result["selection_failure"])

    def test_source_metadata_must_agree_and_alias_matching_does_not_guess(self):
        original = [group("g", source(0, 0), alias="alias")]
        cases = [(source(0, 0, "assistant"), "prior_source_conflicts_with_original"),
                 (source(9, 0, text="Different text.", sid="old"), "no_original_alias_content_match"),
                 ({**source(0, 0), "turn_index": True}, "invalid_prior_source_metadata")]
        for old, reason in cases:
            with self.subTest(reason=reason):
                result = self.select(original, [seed([group("old", old, alias="alias")], old["source_id"])])
                self.assertEqual(ids(result), [])
                self.assertEqual(result["diagnostics"]["unmapped_citations"][0]["reason"], reason)

    def test_conflicting_prior_identity_is_not_silently_recovered_from_one_copy(self):
        original = [group("g", source(0, 0))]
        prior = [original[0], group("other", source(0, 0, text="Conflicting source."))]
        result = self.select(original, [seed(prior, "raw-0-0")])
        self.assertEqual(ids(result), [])
        self.assertEqual(result["diagnostics"]["unmapped_citations"][0]["reason"],
                         "conflicting_prior_source_identity")

    def test_orphan_assistant_keeps_role_and_only_existing_user_context(self):
        leading = source(0, 0, "assistant", "Leading answer.")
        original = [group("orphan", leading), group("later", source(0, 2), source(0, 4))]
        result = self.select(original, [seed([original[0]], leading["source_id"])])
        self.assertEqual(ids(result), ["raw-0-0", "raw-0-2", "raw-0-4"])
        self.assertEqual(result["source_groups"][0]["sources"][0]["role"], "assistant")

    def test_counterevidence_uses_whole_turns_when_a_large_sibling_does_not_fit(self):
        large, small = source(0, 0, text="X" * 2000), source(0, 1, "assistant", "Small raw turn.")
        original = [group("g", large, small)]
        result = self.select(original, counter=["g"], cap=len(serialize([group("g", small)])))
        self.assertEqual(result["source_groups"][0]["sources"], [small])
        self.assertEqual(result["diagnostics"]["dropped_counterevidence_source_ids"], [large["source_id"]])

    def test_duplicate_source_ids_are_deduplicated_and_inputs_remain_unchanged(self):
        shared = source(0, 0)
        original = [group("first", shared), group("second", copy.deepcopy(shared))]
        seeds = [seed(original, shared["source_id"], shared["source_id"])]
        before = copy.deepcopy((original, seeds))
        result = self.select(original, seeds, ["second", "first"])
        self.assertEqual(ids(result), [shared["source_id"]])
        self.assertEqual(result["source_groups"][0]["group_id"], "first")
        self.assertEqual((original, seeds), before)

    def test_question_only_prompt_over_cap_is_an_explicit_selection_failure(self):
        original = [group("g", source(0, 0))]
        result = self.select(original, [seed(original, "raw-0-0")], ["g"], cap=len(serialize([])) - 1)
        self.assertEqual(result["source_groups"], [])
        self.assertTrue(result["selection_failure"])
        self.assertEqual(result["failure_reason"], "question_exceeds_cap")

    def test_rejects_invalid_originals_control_arguments_and_unstripped_prior_answers(self):
        valid = [group("g", source(0, 0))]
        invalid_groups = [[group("g", {**source(0, 0), "role": "system"})],
                          [group("g", {**source(0, 0), "observation_time": "today"})],
                          [group("g", source(0, 0), {**source(0, 0), "source_id": "different"})]]
        for groups in invalid_groups:
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                self.select(groups)
        for kwargs in ({"cap": True}, {"counter": ["unknown"]}, {"counter": ["g", "g"]},
                       {"seeds": [{**seed(valid, "raw-0-0"), "answer": "old answer"}]},
                       {"count_callback": lambda _: True}, {"serialize_callback": lambda _: None}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.select(valid, **kwargs)


if __name__ == "__main__":
    unittest.main()
