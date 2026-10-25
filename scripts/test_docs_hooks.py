# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MkDocs link-rewriting hook. Run: python -m unittest scripts/test_docs_hooks.py"""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location("docs_hooks", Path(__file__).with_name("docs_hooks.py"))
hooks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hooks)


class FakeFiles:
    def __init__(self, included, excluded=()):
        self.included, self.excluded = set(included), set(excluded)

    def get_file_from_path(self, path):
        if path in self.included:
            return SimpleNamespace(inclusion=SimpleNamespace(is_excluded=lambda: False))
        if path in self.excluded:
            return SimpleNamespace(inclusion=SimpleNamespace(is_excluded=lambda: True))
        return None


class RewriteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "docs" / "concepts").mkdir(parents=True)
        (root / "benchmarks" / "results").mkdir(parents=True)
        (root / "benchmarks" / "results" / "r.json").write_text("{}")
        (root / "docs" / "LOG.md").write_text("log")
        self.config = {"docs_dir": str(root / "docs")}
        self.files = FakeFiles({"a.md", "concepts/b.md"}, {"LOG.md"})

    def tearDown(self):
        self.tmp.cleanup()

    def run_hook(self, markdown, src="concepts/b.md"):
        page = SimpleNamespace(file=SimpleNamespace(src_uri=src))
        return hooks.on_page_markdown(markdown, page, self.config, self.files)

    def test_repo_file_outside_docs_links_to_github(self):
        out = self.run_hook("[r](../../benchmarks/results/r.json#x)")
        self.assertEqual(
            out, f"[r]({hooks.REPO}/blob/main/benchmarks/results/r.json#x)"
        )

    def test_excluded_doc_links_to_github(self):
        out = self.run_hook("[l](../LOG.md)")
        self.assertEqual(out, f"[l]({hooks.REPO}/blob/main/docs/LOG.md)")

    def test_site_page_and_external_links_are_untouched(self):
        text = "[a](../a.md) [x](https://example.com/a) [f](#frag)"
        self.assertEqual(self.run_hook(text), text)

    def test_missing_target_is_left_for_strict_mode_to_report(self):
        text = "[n](../../benchmarks/results/missing.json)"
        self.assertEqual(self.run_hook(text), text)

    def test_links_inside_code_fences_are_untouched(self):
        text = "```\n[r](../../benchmarks/results/r.json)\n```"
        self.assertEqual(self.run_hook(text), text)

    def test_typescript_matrix_suffixes_are_literal_not_empty_references(self):
        for dimensions in (1, 2, 3):
            suffix = "[]" * dimensions
            text = f"`number`{suffix} and [`Hit`](Hit.md){suffix}"
            expected_suffix = r"\[]" * dimensions if dimensions > 1 else suffix
            expected = f"`number`{expected_suffix} and [`Hit`](Hit.md){expected_suffix}"
            self.assertEqual(self.run_hook(text, "api/node/Example.md"), expected)
            self.assertEqual(self.run_hook(text), text)
        code = "```ts\nconst rows: number[][] = [];\n```"
        self.assertEqual(self.run_hook(code, "api/node/Example.md"), code)


if __name__ == "__main__":
    unittest.main()
