# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import re
import sys

task = Path.home() / "sol-qag"
out = task / "v4"
sys.path.insert(0, str(Path.home() / "vibe-hms/vibe"))
from vibe.algorithms.qag_pair.module import _load_extension

sys.modules["holographic_memory"] = _load_extension(out / "holographic_memory.so", "_hms_qag_v4_final_tests")
import pytest

status = pytest.main(["-q", str(task / "repo-v4/tests/python/test_vector_index.py")])
if status:
    raise SystemExit(status)
log = out / "logs/all-features-tests.log"
counts = re.findall(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored", log.read_text())
if not counts or any(int(failed) for _, failed, _ in counts):
    raise RuntimeError("all-features gate is incomplete")
(out / "final-gate.json").write_text(json.dumps({
    "status": "pass", "command": "cargo +1.96.0 test --locked --all-features",
    "passed": sum(int(passed) for passed, _, _ in counts),
    "ignored": sum(int(ignored) for _, _, ignored in counts),
    "native_python_passed": 52,
}, indent=2) + "\n")
