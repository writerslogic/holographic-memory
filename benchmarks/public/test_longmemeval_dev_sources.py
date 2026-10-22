# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Source integrity and dev isolation for compressed geometry observations."""

import copy
import hashlib
import json
import struct
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path
from unittest.mock import patch

import longmemeval_dev_sources as sources


def encoded_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def array_member(packed, *, version=1, header=None):
    if header is None:
        header = {"descr": "|u1", "fortran_order": False, "shape": (len(packed),)}
    encoded_header = (repr(header) + "\n").encode("latin1")
    prefix = b"\x93NUMPY" + bytes((version, 0))
    length = struct.pack("<H" if version == 1 else "<I", len(encoded_header))
    return prefix + length + encoded_header + packed


def source_record(sid="session-a", tid="turn-a"):
    return {"sid": sid, "date": "2023/05/30 (Tue) 09:54",
            "turns": [[tid, "Verbatim café.\n  Keep spacing."]]}


class SourceDecoderTests(unittest.TestCase):
    def setUp(self):
        self.record = source_record()
        self.raw = encoded_json(self.record)
        self.packed = zlib.compress(self.raw)
        self.member = array_member(self.packed)
        self.checksum = hashlib.sha256(self.raw).hexdigest()

    def decode(self, member=None, **overrides):
        args = {"expected_hash": self.checksum, "expected_raw_size": len(self.raw),
                "expected_block_size": len(self.packed)}
        args.update(overrides)
        return sources._decode_source(self.member if member is None else member, **args)

    def test_verbatim_unicode_roundtrip_for_both_supported_header_versions(self):
        for version in (1, 2):
            with self.subTest(version=version):
                self.assertEqual(self.decode(array_member(self.packed, version=version)), self.record)

    def test_rejects_unsupported_dtype_shape_and_order(self):
        valid = {"descr": "|u1", "fortran_order": False, "shape": (len(self.packed),)}
        invalid = [{**valid, key: value} for key, value in (
            ("descr", "<f8"), ("descr", "|O"), ("fortran_order", True),
            ("fortran_order", 0), ("shape", (len(self.packed), 1)),
            ("shape", [len(self.packed)]), ("shape", (True,)), ("shape", (-1,)),
            ("shape", (0,)), ("shape", (sources.MAX_MEMBER_BYTES + 1,)))]
        invalid.extend(({**valid, "extra": 1}, {"descr": "|u1"}))
        for header in invalid:
            with self.subTest(header=header), self.assertRaises(ValueError):
                self.decode(array_member(self.packed, header=header))

    def test_rejects_truncated_headers_payloads_and_unknown_versions(self):
        for member in (b"", self.member[:7], self.member[:9], self.member[:15],
                       self.member[:-1], self.member + b"x",
                       self.member[:6] + b"\x03\x00" + self.member[8:]):
            with self.subTest(length=len(member)), self.assertRaises(ValueError):
                self.decode(member)
        invalid = b"\x93NUMPY\x01\x00" + struct.pack("<H", 4) + b"oops" + self.packed
        with self.assertRaisesRegex(ValueError, "header"):
            self.decode(invalid)

    def test_rejects_truncated_trailing_or_concatenated_compressed_streams(self):
        for packed in (self.packed[:-1], self.packed + b"junk", self.packed + self.packed, b"invalid"):
            with self.subTest(length=len(packed)), self.assertRaises(ValueError):
                self.decode(array_member(packed), expected_block_size=len(packed))

    def test_verifies_hash_and_exact_sizes(self):
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.decode(expected_hash="0" * 64)
        for delta in (-1, 1):
            with self.subTest(raw_delta=delta), self.assertRaisesRegex(ValueError, "decoded source size"):
                self.decode(expected_raw_size=len(self.raw) + delta)
            with self.subTest(block_delta=delta), self.assertRaisesRegex(ValueError, "compressed source size"):
                self.decode(expected_block_size=len(self.packed) + delta)
        for size in (True, 0, -1, sources.MAX_SOURCE_BYTES + 1):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "decoded source size"):
                self.decode(expected_raw_size=size)

    def test_member_and_decoded_allocation_boundaries(self):
        for constant, size in (("MAX_MEMBER_BYTES", len(self.member)), ("MAX_SOURCE_BYTES", len(self.raw))):
            for delta in (-1, 0, 1):
                with self.subTest(constant=constant, delta=delta), patch.object(sources, constant, size + delta):
                    if delta == -1:
                        with self.assertRaisesRegex(ValueError, "size"):
                            self.decode()
                    else:
                        self.assertEqual(self.decode(), self.record)

    def test_rejects_source_schema_types_and_dates(self):
        invalid = [{**self.record, "turns": [["tid", 4]]},
                   {**self.record, "turns": [["tid", "text", "extra"]]},
                   {**self.record, "turns": "text"}, {**self.record, "sid": 4},
                   {**self.record, "date": "tomorrow"}, {**self.record, "extra": "label"}]
        for record in invalid:
            raw = encoded_json(record)
            packed = zlib.compress(raw)
            with self.subTest(record=record), self.assertRaises(ValueError):
                self.decode(array_member(packed), expected_hash=hashlib.sha256(raw).hexdigest(),
                            expected_raw_size=len(raw), expected_block_size=len(packed))


class DevSourceLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.split = {"dev": [f"dev-{i:03d}" for i in range(100)],
                      "heldout": [f"heldout-{i:03d}" for i in range(400)]}
        self.split_path = self.work / "split.json"
        self.split_path.write_bytes(encoded_json(self.split))
        self.manifest = {"version": 1, "source_sha256": sources.SOURCE_SHA256,
                         "split_sha256": hashlib.sha256(self.split_path.read_bytes()).hexdigest(),
                         "n_turns": 200, "questions": []}
        self.members = {}
        self.records = {}
        for index, qid in enumerate(self.split["dev"]):
            records = [source_record(f"session-{index}-{j}", f"turn-{index}-{j}") for j in range(2)]
            raw = [encoded_json(record) for record in records]
            packed = [zlib.compress(block) for block in raw]
            self.records[qid] = records
            self.members[index] = {f"source.{j}.npy": array_member(block) for j, block in enumerate(packed)}
            self.members[index]["raw.npy"] = b"model arrays must never be opened"
            self.write_archive(index)
            self.manifest["questions"].append({
                "qid": qid, "qtype": "label must not escape", "scored": True,
                "sids": [record["sid"] for record in records], "counts": [1, 1],
                "tids": [record["turns"][0][0] for record in records],
                "source_hashes": [hashlib.sha256(block).hexdigest() for block in raw],
                "source_raw_sizes": [len(block) for block in raw],
                "source_block_sizes": [len(block) for block in packed]})
        self.write_manifest()

    def write_archive(self, index):
        with zipfile.ZipFile(self.work / f"q{index:03d}.npz", "w") as archive:
            for name, member in self.members[index].items():
                archive.writestr(name, member)

    def write_manifest(self):
        (self.work / "manifest.json").write_bytes(encoded_json(self.manifest))

    def load(self):
        return sources.load_dev_sources(self.work, self.split_path)

    def test_loads_exactly_dev_verbatim_sources_without_opening_model_arrays(self):
        opened = []
        original_open = zipfile.ZipFile.open

        def tracking_open(archive, name, *args, **kwargs):
            opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
            return original_open(archive, name, *args, **kwargs)

        with patch.object(zipfile.ZipFile, "open", tracking_open):
            actual = self.load()
        self.assertEqual(actual, self.records)
        self.assertEqual(len(opened), 200)
        self.assertEqual(set(opened), {"source.0.npy", "source.1.npy"})
        self.assertTrue(all(set(record) == {"sid", "date", "turns"}
                            for records in actual.values() for record in records))

    def test_rejects_nondev_duplicate_or_wrong_count_before_opening_archives(self):
        original = copy.deepcopy(self.manifest)
        for mutation in ("heldout", "duplicate", "short", "long"):
            self.manifest = copy.deepcopy(original)
            questions = self.manifest["questions"]
            if mutation == "heldout":
                questions[0]["qid"] = self.split["heldout"][0]
            elif mutation == "duplicate":
                questions[0]["qid"] = questions[1]["qid"]
            elif mutation == "short":
                questions.pop()
            else:
                questions.append(copy.deepcopy(questions[-1]))
            self.write_manifest()
            with self.subTest(mutation=mutation), patch.object(zipfile, "ZipFile") as archive:
                with self.assertRaises(ValueError):
                    self.load()
                archive.assert_not_called()

    def test_requires_pinned_release_and_matching_split_hash(self):
        for key in ("source_sha256", "split_sha256"):
            old = self.manifest[key]
            self.manifest[key] = "0" * 64
            self.write_manifest()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "release|fingerprint"):
                self.load()
            self.manifest[key] = old

    def test_checks_session_turn_count_and_turn_order(self):
        original = copy.deepcopy(self.manifest)
        for key, value in (("sids", ["wrong", "session-0-1"]), ("counts", [2, 1]),
                           ("tids", ["turn-0-1", "turn-0-0"]), ("counts", [True, 1])):
            self.manifest = copy.deepcopy(original)
            self.manifest["questions"][0][key] = value
            if key == "counts" and value == [2, 1]:
                self.manifest["questions"][0]["tids"].append("another-turn")
                self.manifest["n_turns"] += 1
            self.write_manifest()
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "order|count"):
                self.load()

    def test_validates_hash_and_size_from_manifest(self):
        original = copy.deepcopy(self.manifest)
        for key, value in (("source_hashes", "0" * 64), ("source_raw_sizes", 1),
                           ("source_block_sizes", 1)):
            self.manifest = copy.deepcopy(original)
            self.manifest["questions"][0][key][0] = value
            self.write_manifest()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "hash|size"):
                self.load()

    def test_requires_exact_source_members_and_uses_only_fixed_archive_filenames(self):
        self.members[0]["source.2.npy"] = self.members[0]["source.0.npy"]
        self.write_archive(0)
        with self.assertRaisesRegex(ValueError, "member membership"):
            self.load()
        del self.members[0]["source.2.npy"]
        self.write_archive(0)
        self.manifest["questions"][0]["path"] = "../../never-read.npz"
        self.write_manifest()
        self.assertEqual(self.load(), self.records)


if __name__ == "__main__":
    unittest.main()
