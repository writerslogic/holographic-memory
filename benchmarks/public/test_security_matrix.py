# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests of security_matrix.py's verdict and injection logic."""
import random

import security_matrix as sm


def test_classify_maps_exit_codes():
    assert sm.classify(0, "") == "served_identical"
    assert sm.classify(3, "") == "served_different"
    assert sm.classify(2, "") == "detected"
    assert sm.classify(101, "") == "crashed"
    assert sm.classify(-11, "") == "crashed"
    assert sm.classify(-999, "") == "crashed"


def test_flip_byte_changes_exactly_one_bit(tmp_path):
    p = tmp_path / "f.bin"
    original = bytes(range(256)) * 4
    p.write_bytes(original)
    at = sm.flip_byte(p, random.Random(1))
    changed = p.read_bytes()
    assert len(changed) == len(original)
    diff = [i for i in range(len(original)) if original[i] != changed[i]]
    assert diff == [at]
    assert bin(original[at] ^ changed[at]).count("1") == 1
    empty = tmp_path / "e.bin"
    empty.write_bytes(b"")
    assert sm.flip_byte(empty, random.Random(1)) is None


def test_sqlite_tamper_detects_or_serves(tmp_path):
    db = tmp_path / "c.db"
    sm.build_sqlite(db, 50)
    rows, check = sm.sqlite_query(db)
    assert check == "ok" and rows
    out = sm.tamper_sqlite(db, 12, random.Random(2))
    assert sum(out["counts"].values()) == 12
    assert out["counts"]["crashed"] == 0


def test_store_files_skips_empty(tmp_path):
    (tmp_path / "a").write_bytes(b"x")
    (tmp_path / "b").write_bytes(b"")
    assert [p.name for p in sm.store_files(tmp_path)] == ["a"]
