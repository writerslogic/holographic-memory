# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests of durability_matrix.py: shadow restore, kill summaries, the SQLite probe."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import durability_matrix as dm  # noqa: E402


def test_restore_from_shadow_keeps_synced_content_and_drops_unsynced(tmp_path):
    store = tmp_path / "store"
    shadow = tmp_path / "shadow"
    (store / "sub").mkdir(parents=True)
    (store / "a.bin").write_bytes(b"synced-part" + b"tail-after-sync")
    (store / "sub" / "b.bin").write_bytes(b"never synced")
    shadow.mkdir()
    st = (store / "a.bin").stat()
    (shadow / f"{st.st_dev}-{st.st_ino}").write_bytes(b"synced-part")
    dm.restore_from_shadow(str(store), str(shadow))
    assert (store / "a.bin").read_bytes() == b"synced-part"
    assert not (store / "sub" / "b.bin").exists()


def test_summarize_kill_counts_losses_and_open_failures():
    rows = [
        {"open_ok": True, "committed": 3, "missing": 0, "mismatched": 0, "recovery_ms": 10.0, "bytes_at_kill": 100, "bytes_after": 90, "uncommitted_present": 1},
        {"open_ok": True, "committed": 5, "missing": 2, "mismatched": 1, "recovery_ms": 30.0, "bytes_at_kill": 200, "bytes_after": 200},
        {"open_ok": False, "committed": 1, "recovery_ms": 5.0, "bytes_at_kill": 10, "bytes_after": 10},
    ]
    s = dm.summarize_kill(rows)
    assert s["pass"] is False
    assert s["open_failures"] == 1
    assert s["committed_docs_lost"] == 3
    assert s["points_with_loss"] == 1
    assert s["uncommitted_served"] == 1
    assert s["recovery_ms_median"] == 20.0
    assert s["bytes_lost_median"] == 5
    assert len(s["failures"]) == 2
    assert dm.summarize_kill([rows[0]])["pass"] is True


def test_sqlite_probe_writes_verifies_and_rejects_bumped_header(tmp_path):
    script = tmp_path / "probe.py"
    script.write_text(dm.SQLITE_SCRIPT)
    store = tmp_path / "s"
    base = [sys.executable, "-I", str(script)]
    out = subprocess.run(base + ["write", str(store), "50", "10", "3", "0"], capture_output=True, text=True, check=True).stdout
    assert out.count("committed") == 5 and "compacted 2" in out
    v = json.loads(subprocess.run(base + ["verify", str(store), "50", "10", "0", "0", "5"], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
    assert v["open_ok"] and v["present"] == 50 and v["missing"] == 0 and v["mismatched"] == 0 and v["detected"] is False
    # A claim of six committed batches with only five written is reported as missing.
    v = json.loads(subprocess.run(base + ["verify", str(store), "60", "10", "0", "0", "6"], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
    assert v["missing"] == 10

    class A:
        docs, batch = 50, 10

    s = dm.Sqlite.__new__(dm.Sqlite)
    s.a = A()
    r = s.old_binary(str(store))
    assert r["pass"] is True
