# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from warmbench import text_worker  # noqa: E402


@pytest.mark.parametrize("system, package", [("sqlite-fts5", "sqlite3"), ("tantivy", "tantivy"), ("whoosh", "whoosh")])
def test_text_mutations_are_visible_and_committed(tmp_path, system, package):
    pytest.importorskip(package)
    docs = [{"id": "one/α", "text": "apricot orchard"}, {"id": "two", "text": "banana orchard"}]
    adapter = text_worker.SYSTEMS[system](tmp_path / "store")
    adapter.build(docs)
    adapter.open()
    adapter.replace(docs[0])
    assert [i for i, _ in adapter.search(adapter.prepare("benchmarkrevision"))] == ["one/α"]
    adapter.delete(docs[1])
    assert adapter.search(adapter.prepare("banana")) == []
    adapter.open()
    assert [i for i, _ in adapter.search(adapter.prepare("benchmarkrevision"))] == ["one/α"]
    assert adapter.search(adapter.prepare("banana")) == []
