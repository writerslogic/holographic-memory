# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import unittest
from pathlib import Path

import tiktoken

import longmemeval_sota_closure as F
import longmemeval_sota as S
import longmemeval_sota_context as C
import longmemeval_sota_expand as E
import longmemeval_sota_hybrid as H
import longmemeval_sota_receipts as N
import longmemeval_retrieval as R


def source(session, index, identity, role="user", text="literal observation", premises=()):
    return {"recordId": C.record_id(session, "v1", index, identity), "sourceId": identity,
            "sessionId": session, "version": "v1", "turnIndex": index, "role": role,
            "observationTime": index, "text": text, "premiseRecordIds": list(premises)}


def evidence(sources, anchors):
    return {"sources": sources, "anchors": [{**{field: sources[i][field] for field in (
        "recordId", "sourceId", "sessionId", "version", "turnIndex")}, "score": score} for i, score in anchors],
        "dropped": [], "serializedBytes": len(C.wire(sources))}


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.user = source("thread", 0, "user", text="  Café 東京 🦀\n")
        self.answer = source("thread", 1, "assistant", "assistant", "Seven.\n", [self.user["recordId"]])

    def test_exact_round_trip_and_deterministic_canonical_order(self):
        packed = C.pack([self.answer, self.user])
        self.assertEqual(C.unpack(packed), [self.user, self.answer])
        self.assertEqual(C.pack([self.user, self.answer]), packed)
        self.assertEqual(packed["turns"][1][6], [0])
        self.assertEqual(packed["sufficiency"], "unverified")

    def test_unresolved_link_is_retained_without_inventing_its_source(self):
        packed = C.pack([self.answer])
        self.assertEqual(packed["turns"][0][6], [self.user["recordId"]])
        self.assertEqual(C.unpack(packed), [self.answer])
        bad = C.pack([self.user, self.answer])
        bad["turns"][1][6] = [self.user["recordId"]]
        with self.assertRaises(ValueError):
            C.unpack(bad)

    def test_mutation_and_cross_scope_links_are_rejected(self):
        packed = C.pack([self.user, self.answer])
        for index, replacement in ((0, 9), (1, 9), (2, "wrong"), (3, "user"), (4, 99), (5, "altered"), (6, [1])):
            bad = copy.deepcopy(packed)
            bad["turns"][1][index] = replacement
            with self.assertRaises(ValueError):
                C.unpack(bad)
        cross_scope = source("elsewhere", 1, "other", premises=[self.user["recordId"]])
        with self.assertRaises(ValueError):
            C.pack([self.user, cross_scope])

    def test_byte_and_unicode_label_boundaries(self):
        size = len(C.wire(C.pack([self.user, self.answer])))
        for limit in (size - 1, size, size + 1):
            if limit < size:
                with self.assertRaises(ValueError):
                    C.pack([self.user, self.answer], limit)
            else:
                self.assertEqual(C.unpack(C.pack([self.user, self.answer], limit)), [self.user, self.answer])
        for length in (255, 256, 257):
            member = source("thread", 0, "x" * length)
            if length > 256:
                with self.assertRaises(ValueError):
                    C.pack([member])
            else:
                self.assertEqual(C.unpack(C.pack([member])), [member])
        with self.assertRaises(ValueError):
            C.pack([source("thread", 0, "bad\u0085label")])

    def test_duplicate_and_changed_identity_are_rejected(self):
        with self.assertRaises(ValueError):
            C.pack([self.user, self.user])
        bad = {**self.user, "recordId": "conversation:source:" + "0" * 64}
        with self.assertRaises(ValueError):
            C.pack([bad])

    def test_python_and_rust_share_the_committed_golden_contract(self):
        path = Path(__file__).resolve().parents[2] / "tests/fixtures/conversation_context_v1.json"
        golden = json.loads(path.read_text())
        sources = C.unpack(golden)
        self.assertEqual(C.pack(sources), golden)
        self.assertEqual(sources[0]["text"], "  Café 東京 🦀\n")


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.sources = [source("a", 0, "a", text="appointment booked"),
                        source("b", 0, "b", text="appointment cancelled"),
                        source("c", 0, "c", text="appointment requested")]
        self.seed = evidence(self.sources[:1], [(0, 1 / 61)])
        self.broad = evidence(self.sources, [(0, 1 / 61), (1, 1 / 62), (2, 1 / 63), (2, 1 / 63)])
        self.rounds = {s["sourceId"]: s["sessionId"] for s in self.sources}

    def test_expansion_is_monotone_deduplicated_and_does_not_infer_event_count(self):
        result = E.expand("How many appointments?", self.seed, self.broad, C.MAX_BYTES, self.rounds)
        self.assertEqual(C.unpack(result), self.sources)
        self.assertEqual(result["sufficiency"], "unverified")
        self.assertNotIn("count", result)
        self.assertEqual(C.unpack(E.expand("Where is my appointment?", self.seed, self.broad, C.MAX_BYTES, self.rounds)), self.seed["sources"])

    def test_round_source_and_byte_budgets_admit_complete_sources_only(self):
        full = E.expand("How many appointments?", self.seed, self.broad, C.MAX_BYTES, self.rounds)
        size = len(C.wire(full))
        for limit in (size - 1, size, size + 1):
            result = E.expand("How many appointments?", self.seed, self.broad, limit, self.rounds)
            restored = C.unpack(result)
            self.assertIn(self.sources[0], restored)
            self.assertEqual(len(restored) == 3, limit >= size)
        for cap in (1, 2, 3):
            result = E.expand("How many appointments?", self.seed, self.broad, C.MAX_BYTES, self.rounds, max_rounds=cap)
            self.assertEqual(len(C.unpack(result)), cap)
            result = E.expand("How many appointments?", self.seed, self.broad, C.MAX_BYTES, self.rounds, max_sources=cap)
            self.assertEqual(len(C.unpack(result)), cap)

    def test_temporal_revisions_and_identical_speaker_mentions_remain_distinct(self):
        self.sources[1]["text"] = self.sources[0]["text"]
        self.broad = evidence(self.sources, [(0, 1 / 61), (1, 1 / 62), (2, 1 / 63)])
        result = C.unpack(E.expand("How many appointments?", self.seed, self.broad, C.MAX_BYTES, self.rounds))
        self.assertEqual(len(result), 3)
        self.assertEqual({s["recordId"] for s in result}, {s["recordId"] for s in self.sources})

    def test_empty_candidates_never_imply_exhaustive_search(self):
        result = E.expand("How many appointments?", self.seed, evidence([], []), C.MAX_BYTES, self.rounds)
        self.assertEqual(C.unpack(result), self.seed["sources"])
        self.assertEqual(result["sufficiency"], "unverified")


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.encoder = tiktoken.get_encoding("o200k_base")

    def test_completion_preserves_seed_and_cannot_follow_a_reference_outside_seed_rounds(self):
        earlier = source("thread", 0, "earlier")
        current = source("thread", 2, "current")
        answer = source("thread", 3, "answer", "assistant", premises=[earlier["recordId"]])
        available = [earlier, current, answer]
        original = copy.deepcopy(available)
        rounds = {"earlier": "excluded", "current": "kept", "answer": "kept"}
        packed = F.complete([answer], available, rounds, C.MAX_BYTES, 100000, self.encoder)
        self.assertEqual(C.unpack(packed), [current, answer])
        self.assertEqual(packed["turns"][1][6], [earlier["recordId"]])
        self.assertEqual(available, original)
        self.assertEqual(F.complete([answer], list(reversed(available)), rounds,
                                    C.MAX_BYTES, 100000, self.encoder), packed)

    def test_byte_and_token_boundaries_preserve_seed_without_truncating_additions(self):
        seed = source("thread", 0, "seed", text="appointment booked")
        added = source("thread", 1, "added", "assistant", "appointment cancelled " * 30)
        members = [seed, added]
        rounds = {member["sourceId"]: "one" for member in members}
        full = C.pack(members)
        size = len(C.wire(full))
        tokens = len(self.encoder.encode(C.wire(full).decode(), disallowed_special=()))
        for limit in (size - 1, size, size + 1):
            restored = C.unpack(F.complete([seed], members, rounds, limit, 100000, self.encoder))
            self.assertEqual(restored, members if limit >= size else [seed])
        for limit in (tokens - 1, tokens, tokens + 1):
            restored = C.unpack(F.complete([seed], members, rounds, C.MAX_BYTES, limit, self.encoder))
            self.assertEqual(restored, members if limit >= tokens else [seed])
        seed_tokens = len(self.encoder.encode(C.wire(C.pack([seed])).decode(), disallowed_special=()))
        with self.assertRaisesRegex(ValueError, "seed exceeds token budget"):
            F.complete([seed], members, rounds, C.MAX_BYTES, seed_tokens - 1, self.encoder)

    def test_source_cap_admits_whole_sources_at_limit_minus_one_limit_and_limit_plus_one(self):
        members = [source("thread", index, f"source-{index}") for index in range(65)]
        rounds = {member["sourceId"]: "one" for member in members}
        for count in (63, 64, 65):
            if count > 64:
                with self.assertRaisesRegex(ValueError, "invalid completion bounds"):
                    F.complete(members[:count], members, rounds, C.MAX_BYTES, 100000, self.encoder)
            else:
                restored = C.unpack(F.complete(members[:count], members, rounds,
                                              C.MAX_BYTES, 100000, self.encoder))
                self.assertEqual(len(restored), 64)
                self.assertEqual(restored[:count], members[:count])

    def test_evaluation_ceiling_distinguishes_missing_sources_inside_and_outside_seed_rounds(self):
        qid = "synthetic"
        entry = {"question_id": qid, "question_type": "multi-session",
                 "haystack_session_ids": ["session"], "haystack_dates": ["2024/01/01 (Mon) 00:00"],
                 "answer_session_ids": ["session"], "haystack_sessions": [[
                     {"role": "user", "content": "appointment booked"},
                     {"role": "assistant", "content": "appointment booked", "has_answer": True},
                     {"role": "user", "content": "appointment cancelled", "has_answer": True}]]}
        groups, _, _, _ = R.rounds_for(entry)
        members, _, _, _ = S.inputs(groups)
        receipts = {qid: {"replies": {"selected": {"evidence": {"sources": members[:1]}}}}}
        result = F.study({qid: entry}, receipts)
        row = result["rows"][0]
        self.assertEqual(row["recoverable_annotated_source_ids_in_seed_rounds"], [members[1]["sourceId"]])
        self.assertEqual(row["missing_annotated_source_ids_outside_seed_rounds"], [members[2]["sourceId"]])
        self.assertEqual(result["fixed_round_ceiling"]["questions_with_recoverable_missing_annotations"], 1)
        self.assertEqual(result["fixed_round_ceiling"]["missing_annotated_sources_inside_seed_rounds"], 1)
        self.assertEqual(result["fixed_round_ceiling"]["missing_annotated_sources_outside_seed_rounds"], 1)


class ReplayTests(unittest.TestCase):
    def test_portable_receipts_detect_source_corruption(self):
        sources = [source("thread", 0, "u")]
        original = evidence(sources, [(0, 1 / 61)])
        row = {"sources": [0], "anchors": [[0, 1 / 61]], "dropped": [],
               "serialized_bytes": original["serializedBytes"], "evidence_sha256": R.digest(original)}
        self.assertEqual(N.expand_row(row, sources)["evidence"], original)
        altered = [{**sources[0], "text": "altered"}]
        with self.assertRaises(ValueError):
            N.expand_row(row, altered)
        row["sources"] = [-1]
        with self.assertRaises(ValueError):
            N.expand_row(row, sources)

    def test_unscored_reranker_candidates_are_never_given_fabricated_scores(self):
        self.assertEqual(H.rerank_order([0, 1, 2, 3], {0: .1, 2: .9}, 3), [2, 0, 1, 3])
        self.assertEqual(H.rerank_order([0, 1], {}, 2), [0, 1])


if __name__ == "__main__":
    unittest.main()
