# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Every ANN system in the registry is fully wired: adapter, sweeps, pinned environment, lock, facts."""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hmsbench import envs, registry  # noqa: E402

WORKER = Path(__file__).resolve().parents[1] / "hmsbench" / "ann_worker.py"


def _adapters():
    spec = importlib.util.spec_from_file_location("ann_worker_under_test", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SYSTEMS


def test_every_ann_system_is_wired_end_to_end():
    adapters = _adapters()
    for name in registry.ANN_SYSTEMS:
        reg = registry.SYSTEMS[name]
        assert name in adapters, name
        cls = adapters[name]
        assert set(cls.SWEEPS) == set(cls.BUILDS), name
        assert all(cls.SWEEPS[b] for b in cls.BUILDS), name
        assert reg["env"] in envs.PINS, name
        assert (envs.REQUIREMENTS / f"{reg['env']}.txt").read_text().strip(), name
        assert name in registry.TIMING_BOUNDARY and name in registry.FACTS, name


def test_interpreter_override_is_per_env_and_defaults_to_the_shared_python():
    assert envs.python_for("hnswlib") == envs.PYTHON
    assert all(name in envs.PINS for name in envs.PYTHON_FOR)
    assert envs.python_for("voyager") != envs.PYTHON
