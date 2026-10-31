import base64
import copy
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

path = Path(__file__).parents[1] / "writerslogic_warm_sdk/evidence.py"
spec = importlib.util.spec_from_file_location("warm_source_evidence", path)
evidence = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evidence
spec.loader.exec_module(evidence)


def handle(value):
    return hashlib.sha256(value.encode()).hexdigest()


def source(turn, session="s", observed="2024-01-01", role="user", text="A factual source."):
    return (handle(turn), handle(session), observed, role, text)


class EvidenceTests(unittest.TestCase):
    def test_lossless_repeated_occurrences_roles_unicode_and_distinct_dates(self):
        groups = [[source("one", text="I visited café 🐈.")],
                  [source("two", role="assistant", text="Your total was €25.")],
                  [source("one", observed="2024-02-01", text="I visited café 🐈."),
                   source("one", "other", text="I visited café 🐈.")]]
        packet = evidence.encode_source_packet(groups)
        self.assertEqual(evidence.decode_source_packet(packet), groups)
        legacy = json.dumps({"v": 2, "q": groups, "i": []}, ensure_ascii=False,
                            separators=(",", ":"))
        self.assertLess(len(packet.encode()), len(legacy.encode()))
        self.assertEqual(len(json.loads(packet)["t"]), 2)
        self.assertEqual(len(json.loads(packet)["s"]), 3)

    def test_budget_boundary_preserves_mandatory_evidence_and_skips_large_group(self):
        candidates = [[source("a")], [source("large", text="x" * 2000)],
                      [source("small", text="An earlier event.")]]
        cap = len(evidence.encode_source_packet([candidates[0], candidates[2]]).encode())
        for budget, expected in ((cap - 1, (0,)), (cap, (0, 2)), (cap + 1, (0, 2))):
            with self.subTest(budget=budget):
                chosen = evidence.complete_evidence(candidates, [0], budget)
                self.assertEqual(chosen.selected, expected)
                self.assertLessEqual(chosen.packet_bytes, budget)
                self.assertEqual(evidence.decode_source_packet(chosen.packet),
                                 [candidates[index] for index in chosen.selected])
        with self.assertRaisesRegex(ValueError, "mandatory evidence"):
            evidence.complete_evidence(candidates, [0],
                                       len(evidence.encode_source_packet([candidates[0]]).encode()) - 1)

    def test_candidate_count_boundary(self):
        for count in (63, 64):
            candidates = [[source(str(i))] for i in range(count)]
            chosen = evidence.complete_evidence(candidates, [0], 100000)
            self.assertEqual(len(chosen.selected), count)
        with self.assertRaisesRegex(ValueError, "1..=64"):
            evidence.complete_evidence([[source(str(i))] for i in range(65)], [0], 100000)

    def test_source_size_boundary_including_repeated_expansion(self):
        original = evidence.MAX_SOURCE_BYTES
        evidence.MAX_SOURCE_BYTES = 100
        try:
            for size in (99, 100):
                evidence.encode_source_packet([[source("one", text="x" * size)]])
            with self.assertRaisesRegex(ValueError, "expanded source"):
                evidence.encode_source_packet([[source("one", text="x" * 101)]])
            packet = evidence.encode_source_packet([[source("one", text="x" * 60)]])
            value = json.loads(packet)
            value["g"].append(value["g"][0])
            with self.assertRaisesRegex(ValueError, "expanded source"):
                evidence.decode_source_packet(json.dumps(value, separators=(",", ":")))
        finally:
            evidence.MAX_SOURCE_BYTES = original

    def test_bad_references_noncanonical_handles_and_inferences_fail(self):
        packet = evidence.encode_source_packet([[source("one")]])
        mutations = []
        for changed_index in (-1, 1, True, 0.0):
            value = json.loads(packet)
            value["g"][0][0][0] = changed_index
            mutations.append(value)
        value = json.loads(packet)
        value["i"] = ["invented answer"]
        mutations.append(value)
        value = json.loads(packet)
        value["t"].append(copy.deepcopy(value["t"][0]))
        mutations.append(value)
        value = json.loads(packet)
        compact = value["t"][0][0]
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        value["t"][0][0] = compact[:-1] + alphabet[alphabet.index(compact[-1]) + 1]
        self.assertEqual(base64.urlsafe_b64decode(value["t"][0][0] + "="),
                         base64.urlsafe_b64decode(compact + "="))
        mutations.append(value)
        for value in mutations:
            with self.subTest(value=value), self.assertRaises(ValueError):
                evidence.decode_source_packet(json.dumps(value, separators=(",", ":")))
        with self.assertRaisesRegex(ValueError, "duplicate packet key"):
            evidence.decode_source_packet(packet.replace('"v":3', '"v":3,"v":3'))

    def test_conflicting_identity_or_malformed_selection_fails(self):
        with self.assertRaisesRegex(ValueError, "conflicting"):
            evidence.encode_source_packet([[source("one")], [source("one", text="Different")]])
        for selected in ([True], [-1], [1], [0, 0], [[0]]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                evidence.complete_evidence([[source("one")]], selected, 1000)

    def test_handles_observation_lengths_and_utf8_boundaries(self):
        for size in (63, 64):
            evidence.encode_source_packet([[source("one", observed="x" * size)]])
        with self.assertRaisesRegex(ValueError, "observation time"):
            evidence.encode_source_packet([[source("one", observed="x" * 65)]])
        for invalid in ("0" * 63, "0" * 65, "A" * 64, 1):
            record = list(source("one"))
            record[0] = invalid
            with self.subTest(handle=invalid), self.assertRaises(ValueError):
                evidence.encode_source_packet([[record]])
        for record in (source("one", text="\ud800"), source("one", observed="\ud800")):
            with self.assertRaisesRegex(ValueError, "UTF-8"):
                evidence.encode_source_packet([[record]])
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            evidence.decode_source_packet("\ud800")

    def test_max_packet_utf8_byte_boundary(self):
        empty = source("one", text="")
        value = json.loads(evidence.encode_source_packet([[empty]]))
        overhead = len(json.dumps(value, separators=(",", ":")).encode())
        limit = evidence.MAX_PACKET_BYTES
        for delta in (-1, 0, 1):
            text = "é" + "x" * (limit - overhead - 2 + delta)
            record = source("one", text=text)
            value["t"][0][2] = text
            packet = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            self.assertEqual(len(packet.encode()), limit + delta)
            self.assertLessEqual(len(packet), limit)
            with self.subTest(packet_bytes=limit + delta):
                if delta <= 0:
                    self.assertEqual(evidence.encode_source_packet([[record]]), packet)
                    self.assertEqual(evidence.decode_source_packet(packet), [[record]])
                else:
                    with self.assertRaisesRegex(ValueError, "source packet exceeds limit"):
                        evidence.encode_source_packet([[record]])
                    with self.assertRaisesRegex(ValueError, "source packet exceeds limit"):
                        evidence.decode_source_packet(packet)

    def test_group_count_boundary(self):
        group = [source("one")]
        for count in (evidence.MAX_GROUPS - 1, evidence.MAX_GROUPS):
            groups = [group] * count
            packet = evidence.encode_source_packet(groups)
            self.assertEqual(evidence.decode_source_packet(packet), groups)
        groups = [group] * (evidence.MAX_GROUPS + 1)
        with self.assertRaisesRegex(ValueError, "source group count"):
            evidence.encode_source_packet(groups)
        value = json.loads(packet)
        value["g"].append(value["g"][0])
        with self.assertRaisesRegex(ValueError, "source group count"):
            evidence.decode_source_packet(json.dumps(value, separators=(",", ":")))

    def test_per_group_occurrence_boundary(self):
        record = source("one")
        for count in (15, 16):
            groups = [[record] * count]
            packet = evidence.encode_source_packet(groups)
            self.assertEqual(evidence.decode_source_packet(packet), groups)
        with self.assertRaisesRegex(ValueError, "1..=16 occurrences"):
            evidence.encode_source_packet([[record] * 17])
        value = json.loads(packet)
        value["g"][0].append(value["g"][0][0])
        with self.assertRaisesRegex(ValueError, "1..=16 occurrences"):
            evidence.decode_source_packet(json.dumps(value, separators=(",", ":")))

    def test_total_occurrence_boundary(self):
        record = source("one", text="")
        original = evidence.MAX_GROUPS
        evidence.MAX_GROUPS += 1
        try:
            for count in (evidence.MAX_OCCURRENCES - 1, evidence.MAX_OCCURRENCES):
                whole, remainder = divmod(count, 16)
                groups = [[record] * 16] * whole
                if remainder:
                    groups.append([record] * remainder)
                packet = evidence.encode_source_packet(groups)
                self.assertEqual(evidence.decode_source_packet(packet), groups)
            groups.append([record])
            with self.assertRaisesRegex(ValueError, "too many source occurrences"):
                evidence.encode_source_packet(groups)
            value = json.loads(packet)
            value["g"].append([[0, 0]])
            with self.assertRaisesRegex(ValueError, "too many source occurrences"):
                evidence.decode_source_packet(json.dumps(value, separators=(",", ":")))
        finally:
            evidence.MAX_GROUPS = original

    def test_malformed_fields_and_unused_tables_fail(self):
        value = json.loads(evidence.encode_source_packet([[source("one")]]))
        mutations = []
        for field in ("t", "s", "g"):
            for malformed in (None, {}, 1):
                changed = copy.deepcopy(value)
                changed[field] = malformed
                mutations.append(changed)
        for field, malformed in (("v", True), ("v", 3.0), ("i", None), ("i", {})):
            changed = copy.deepcopy(value)
            changed[field] = malformed
            mutations.append(changed)
        for table, index, malformed in (("t", 0, None), ("t", 1, []), ("t", 2, {}),
                                        ("s", 0, 1), ("s", 1, [])):
            for unused in (False, True):
                changed = copy.deepcopy(value)
                if unused:
                    changed[table].append(copy.deepcopy(changed[table][0]))
                changed[table][-1][index] = malformed
                mutations.append(changed)
        changed = copy.deepcopy(value)
        changed["s"].append(copy.deepcopy(changed["s"][0]))
        mutations.append(changed)
        for changed in mutations:
            with self.subTest(packet=changed), self.assertRaises(ValueError):
                evidence.decode_source_packet(json.dumps(changed, separators=(",", ":")))


if __name__ == "__main__":
    unittest.main()
