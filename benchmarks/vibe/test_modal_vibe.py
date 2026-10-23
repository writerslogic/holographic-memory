# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Stall accounting of the Modal VIBE runner: VIBE runs one process per index config, and a config's
process exiting must not look like the job stopping."""
import importlib.util
import sys
from pathlib import Path
from unittest import mock

sys.modules.setdefault("modal", mock.MagicMock())
_spec = importlib.util.spec_from_file_location("modal_vibe", Path(__file__).with_name("modal_vibe.py"))
mv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mv)

RUNPY, WORKER, CFG1, CFG2 = (10, 1), (11, 1), (12, 1), (13, 1)


def test_progress_survives_config_process_exit():
    # Config 1 ran 1500 s and exits; config 2 starts at 0 and gains 60 s per beat. The old total
    # (sum of live CPU) dropped to 60 and needed 25 min to pass 1500, which killed healthy builds.
    before = {RUNPY: 2.0, WORKER: 1.0, CFG1: 1500.0}
    after = {RUNPY: 2.0, WORKER: 1.0, CFG2: 60.0}
    assert mv.cpu_progress(before, after) == 60.0
    assert mv.cpu_progress(after, after) == 0.0


def test_progress_counts_only_gains():
    assert mv.cpu_progress({CFG1: 10.0}, {CFG1: 10.5, CFG2: 0.0}) == 0.5


def test_descendants_follow_ppid_and_skip_unrelated():
    table = {
        (1, 0): {"ppid": 0, "pgid": 1, "cpu": 0.0},
        (5, 0): {"ppid": 1, "pgid": 5, "cpu": 0.0},  # supervisor
        RUNPY: {"ppid": 5, "pgid": 10, "cpu": 0.0},
        WORKER: {"ppid": 10, "pgid": 10, "cpu": 0.0},
        CFG1: {"ppid": 11, "pgid": 12, "cpu": 0.0},  # own session, as VIBE starts it
        (20, 3): {"ppid": 1, "pgid": 20, "cpu": 0.0},
    }
    assert mv.descendants(table, 10) == {RUNPY, WORKER, CFG1}
    assert mv.descendants(table, 99) == set()
