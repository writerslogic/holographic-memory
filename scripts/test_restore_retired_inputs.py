# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("restore_retired", Path(__file__).with_name("restore-retired-inputs.py"))
RESTORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESTORE)


class HistoricalInputs(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.hooks = self.root / "empty-hooks"
        self.hooks.mkdir()
        self.git("init", "-q")
        self.records = []
        for size in (31, 32, 33):
            name = f"benchmarks/results/fixture-{size}.txt"
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = b"x" * size
            path.write_bytes(data)
            self.records.append({"path": name, "bytes": size, "sha256": hashlib.sha256(data).hexdigest(), "mode": 0o644})
        self.git("add", "benchmarks")
        self.git("commit", "-qm", "fixture")
        self.revision = self.git("rev-parse", "HEAD").decode().strip()
        for record in self.records:
            (self.root / record["path"]).unlink()

    def git(self, *arguments):
        return subprocess.check_output(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                                        "-c", "commit.gpgsign=false", "-c", "core.hooksPath=" + str(self.hooks),
                                        *arguments], cwd=self.root, stderr=subprocess.DEVNULL)

    def test_exact_git_roundtrip_is_idempotent_and_preserves_existing_changes(self):
        record = self.records[0]
        self.assertTrue(RESTORE.restore(self.root, self.revision, record))
        target = self.root / record["path"]
        self.assertEqual(target.read_bytes(), b"x" * 31)
        self.assertFalse(RESTORE.restore(self.root, self.revision, record))
        target.write_bytes(b"owner edit")
        with self.assertRaises(FileExistsError):
            RESTORE.restore(self.root, self.revision, record)
        self.assertEqual(target.read_bytes(), b"owner edit")

    def test_byte_limit_boundaries(self):
        for record in self.records:
            with self.subTest(size=record["bytes"]):
                if record["bytes"] <= 32:
                    self.assertTrue(RESTORE.restore(self.root, self.revision, record, max_bytes=32))
                else:
                    with self.assertRaisesRegex(ValueError, "byte limit"):
                        RESTORE.restore(self.root, self.revision, record, max_bytes=32)
                    self.assertFalse((self.root / record["path"]).exists())

    def test_hash_size_and_path_tampering_never_publish_an_input(self):
        original = self.records[0]
        for change in ({"sha256": "0" * 64}, {"bytes": 30}, {"bytes": True},
                       {"path": "../outside.txt"}, {"path": "/outside.txt"},
                       {"path": "benchmarks/results/../../outside.txt"}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    RESTORE.restore(self.root, self.revision, dict(original, **change))
                self.assertFalse((self.root / original["path"]).exists())
        self.assertEqual(list(self.root.rglob(".hms-restore-*")), [])

    @unittest.skipIf(os.name == "nt", "symlink privileges are not assumed on Windows")
    def test_symlinks_cannot_redirect_publication(self):
        with tempfile.TemporaryDirectory() as outside:
            destination = self.root / "benchmarks/results/redirect"
            destination.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink|escapes"):
                RESTORE.restore(self.root, self.revision,
                                dict(self.records[0], path="benchmarks/results/redirect/input.txt"))
            self.assertEqual(list(Path(outside).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
