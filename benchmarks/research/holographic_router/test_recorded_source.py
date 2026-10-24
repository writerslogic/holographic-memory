# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import gzip
import hashlib
from pathlib import Path
import tempfile
import unittest

from recorded_source import CLEAN_IMPORT, ORIGINAL_IMPORT, verify_source


class RecordedSourceTests(unittest.TestCase):
    def test_only_documented_import_cleanup_can_match_archived_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hybrid_probe.py"
            original = ORIGINAL_IMPORT + b"\nvalue = 3\n"
            expected = hashlib.sha256(original).hexdigest()
            path.with_name("frozen-hybrid-probe.py.gz").write_bytes(gzip.compress(original, mtime=0))
            path.write_bytes(original)
            verify_source(path, expected)
            path.write_bytes(original.replace(ORIGINAL_IMPORT, CLEAN_IMPORT))
            verify_source(path, expected)
            verify_source(path, hashlib.sha256(path.read_bytes()).hexdigest())
            path.write_bytes(path.read_bytes().replace(b"value = 3", b"value = 4"))
            with self.assertRaises(ValueError):
                verify_source(path, expected)

    def test_wrong_archive_or_expected_hash_cannot_authorize_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hybrid_probe.py"
            original = ORIGINAL_IMPORT + b"\n"
            expected = hashlib.sha256(original).hexdigest()
            path.write_bytes(CLEAN_IMPORT + b"\n")
            archive = path.with_name("frozen-hybrid-probe.py.gz")
            archive.write_bytes(gzip.compress(original, mtime=0))
            with self.assertRaises(ValueError):
                verify_source(path, "0" * 64)
            archive.write_bytes(gzip.compress(original + b"\n", mtime=0))
            with self.assertRaises(ValueError):
                verify_source(path, expected)


if __name__ == "__main__":
    unittest.main()
