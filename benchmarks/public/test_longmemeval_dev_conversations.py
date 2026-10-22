# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Prove held-out record bodies never enter the JSON decoder or a record buffer."""

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import longmemeval_dev_conversations as conversations


def record(qid="dev"):
    return {"question_id": qid, "question_type": "single-session-assistant",
            "question": "What did the assistant say?", "question_date": "2023/05/30 (Tue) 09:54",
            "answer": "verbatim", "haystack_session_ids": ["session-a", "session-b"],
            "haystack_dates": ["2023/05/29 (Mon) 09:54", "2023/05/30 (Tue) 09:54"],
            "haystack_sessions": [[], [{"role": "user", "content": "  My question.\n"},
                                        {"role": "assistant", "content": "  café 漢字\n\n\t20%  "}]]}


def block(value):
    raw = json.dumps(value, ensure_ascii=False, indent=4).encode()
    return b"\n".join(b"    " + line for line in raw.split(b"\n"))


def release(*blocks):
    return b"[\n" + b",\n".join(blocks) + b"\n]"


class BoundedLinesOnly(io.BytesIO):
    def read(self, *args, **kwargs):
        raise AssertionError("whole-file or body reads are forbidden")

    def readline(self, size=-1):
        if size <= 0 or size > conversations.MAX_LINE_BYTES + 1:
            raise AssertionError("line reads must have an explicit allocation bound")
        return super().readline(size)


class DevConversationTests(unittest.TestCase):
    def parse(self, raw, dev=None, heldout=None, expected_hash=None):
        return conversations._parse_dev_conversations(
            BoundedLinesOnly(raw), {"dev"} if dev is None else dev,
            {"heldout"} if heldout is None else heldout,
            hashlib.sha256(raw).hexdigest() if expected_hash is None else expected_hash)

    def test_skips_invalid_utf8_and_invalid_json_heldout_payload_without_decoding(self):
        skipped = (b'    {\n        "question_id": "heldout",\n'
                   b'        "payload": "\xff hidden-source-with-invalid-JSON\n    }')
        raw = release(block(record()), skipped)
        decoded_arguments = []
        original_loads = json.loads

        def dev_only_loads(value, *args, **kwargs):
            self.assertNotIn(b"hidden-source", value)
            decoded_arguments.append(bytes(value))
            return original_loads(value, *args, **kwargs)

        with patch.object(conversations.json, "loads", dev_only_loads):
            actual = self.parse(raw)
        self.assertEqual(actual, {"dev": record()})
        self.assertEqual(len(decoded_arguments), 1)

    def test_preserves_full_assistant_unicode_whitespace_and_original_session_indices(self):
        raw_record = record()
        actual = self.parse(release(block(raw_record), block(record("heldout"))))["dev"]
        self.assertEqual(actual, raw_record)
        self.assertEqual(actual["haystack_sessions"][0], [])
        self.assertEqual(actual["haystack_sessions"][1][1]["content"], "  café 漢字\n\n\t20%  ")

    def test_rejects_invalid_dev_json_even_when_heldout_bodies_are_opaque(self):
        invalid = b'    {\n        "question_id": "dev",\n        "field": not-JSON\n    }'
        with self.assertRaisesRegex(ValueError, "dev record JSON"):
            self.parse(release(invalid, block(record("heldout"))))

    def test_rejects_unknown_duplicate_and_missing_question_ids(self):
        cases = [release(block(record("unknown")), block(record("heldout"))),
                 release(block(record()), block(record())), release(block(record()))]
        for raw in cases:
            with self.subTest(size=len(raw)), self.assertRaisesRegex(ValueError, "unknown|duplicate|incomplete"):
                self.parse(raw)

    def test_requires_exact_header_and_root_record_framing(self):
        valid = release(block(record()), block(record("heldout")))
        cases = [valid.replace(b"[\n", b"[\r\n", 1),
                 valid.replace(b"    {\n", b"   {\n", 1),
                 valid.replace(b'        "question_id":', b'       "question_id":', 1),
                 valid.replace(b'        "question_id": "dev",', b'        "another": 1,', 1),
                 valid.replace(b'"question_id": "dev"', b'"question_id": "d\\u0065v"', 1),
                 valid[:-1], valid + b"trailing", valid.replace(b"    },\n", b"    }\n", 1)]
        for index, raw in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.parse(raw)

    def test_rejects_release_hash_mismatch_before_returning_records(self):
        raw = release(block(record()), block(record("heldout")))
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            self.parse(raw, expected_hash="0" * 64)

    def test_validates_dev_alignment_roles_content_and_observation_dates(self):
        mutations = []
        for key, value in (("haystack_dates", []), ("haystack_sessions", "text"),
                           ("haystack_session_ids", ["session-a", 1]),
                           ("question_date", "tomorrow"), ("question_type", "")):
            changed = record()
            changed[key] = value
            mutations.append(changed)
        for turn in ({"role": "system", "content": "text"}, {"role": "assistant", "content": 20}):
            changed = record()
            changed["haystack_sessions"][1][1] = turn
            mutations.append(changed)
        changed = record()
        changed["haystack_dates"][0] = "2023/99/99 (Mon) 09:54"
        mutations.append(changed)
        for index, changed in enumerate(mutations):
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.parse(release(block(changed), block(record("heldout"))))

    def test_rejects_duplicate_dev_json_keys_and_nonstandard_constants(self):
        duplicate = block(record()).replace(b'        "answer": "verbatim",',
                                            b'        "answer": "verbatim",\n        "answer": "other",')
        nonfinite = block(record()).replace(b'"answer": "verbatim"', b'"answer": NaN')
        for changed in (duplicate, nonfinite):
            with self.subTest(size=len(changed)), self.assertRaisesRegex(ValueError, "dev record JSON"):
                self.parse(release(changed, block(record("heldout"))))

    def test_line_record_and_release_allocation_boundaries(self):
        raw_block = block(record())
        raw = release(raw_block)
        boundaries = {"MAX_LINE_BYTES": max(map(len, raw.splitlines(keepends=True))),
                      "MAX_RECORD_BYTES": len(raw_block) + 1,
                      "MAX_RELEASE_BYTES": len(raw)}
        for name, size in boundaries.items():
            for delta in (-1, 0, 1):
                with self.subTest(limit=name, delta=delta), patch.object(conversations, name, size + delta):
                    if delta < 0:
                        with self.assertRaisesRegex(ValueError, "size limit"):
                            self.parse(raw, heldout=set())
                    else:
                        self.assertEqual(self.parse(raw, heldout=set()), {"dev": record()})

    def test_applies_record_bound_to_skipped_bodies_without_allocating_them(self):
        skipped = (b'    {\n        "question_id": "heldout",\n'
                   + b'        not JSON\n' * 100 + b"    }")
        raw = release(block(record()), skipped)
        with patch.object(conversations, "MAX_RECORD_BYTES", len(block(record())) + 100):
            with self.assertRaisesRegex(ValueError, "record exceeds size limit"):
                self.parse(raw)

    def test_rejects_overlapping_membership_and_invalid_expected_hash(self):
        raw = release(block(record()))
        for kwargs in ({"heldout": {"dev"}}, {"expected_hash": "invalid"},
                       {"dev": {"has spaces"}, "heldout": set()}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.parse(raw, **kwargs)

    def test_public_loader_rejects_modified_split_before_opening_source(self):
        with tempfile.TemporaryDirectory() as directory:
            split_path = Path(directory) / "split.json"
            split_path.write_text(json.dumps({"dev": ["dev"], "heldout": ["heldout"]}))
            with self.assertRaisesRegex(ValueError, "split fingerprint mismatch"):
                conversations.load_dev_conversations(Path(directory) / "missing-source.json", split_path)


if __name__ == "__main__":
    unittest.main()
