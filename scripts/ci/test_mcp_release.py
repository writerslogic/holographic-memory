# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the MCP release asset manifest."""

import hashlib
import importlib.util
import pathlib
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).with_name("mcp-release.py")
SPEC = importlib.util.spec_from_file_location("mcp_release", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ChecksumsTests(unittest.TestCase):
    def test_exact_assets_have_their_own_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            assets = pathlib.Path(directory)
            for triple in MODULE.TRIPLES:
                suffix = ".exe" if "windows" in triple else ""
                (assets / f"warm-mcp-{triple}{suffix}").write_bytes(triple.encode())
            table = MODULE.checksums(assets)
            self.assertEqual(set(table), set(MODULE.TRIPLES))
            for triple, digest in table.items():
                self.assertEqual(digest, hashlib.sha256(triple.encode()).hexdigest())

    def test_missing_or_extra_asset_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            assets = pathlib.Path(directory)
            for triple in MODULE.TRIPLES:
                suffix = ".exe" if "windows" in triple else ""
                (assets / f"warm-mcp-{triple}{suffix}").write_bytes(b"binary")
            expected = assets / f"warm-mcp-{MODULE.TRIPLES[0]}"
            expected.unlink()
            with self.assertRaisesRegex(ValueError, "missing="):
                MODULE.checksums(assets)
            expected.write_bytes(b"binary")
            (assets / "unexpected").write_bytes(b"binary")
            with self.assertRaisesRegex(ValueError, "unexpected="):
                MODULE.checksums(assets)


if __name__ == "__main__":
    unittest.main()
