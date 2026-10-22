# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Every ANN system in the registry is fully wired: adapter, sweeps, pinned environment, lock, facts."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hmsbench import envs, registry  # noqa: E402

WORKER = Path(__file__).resolve().parents[1] / "hmsbench" / "ann_worker.py"


def _adapters():
    spec = importlib.util.spec_from_file_location("ann_worker_under_test", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SYSTEMS


@pytest.mark.parametrize("metric", ["angular", "euclidean"])
def test_ngt_mutations_preserve_dataset_ids(tmp_path, metric):
    ngtpy = pytest.importorskip("ngtpy")
    cls = _adapters()["ngt"]
    adapter = cls(16, tmp_path, 1, metric)
    rng = np.random.default_rng(7)
    train = rng.standard_normal((200, 16)).astype(np.float32)
    fresh = rng.standard_normal((1, 16)).astype(np.float32)
    if metric == "angular":
        train /= np.linalg.norm(train, axis=1, keepdims=True)
        fresh /= np.linalg.norm(fresh, axis=1, keepdims=True)
    params = cls.BUILDS["onng_default"]
    path = str(tmp_path / "onng")
    ngtpy.create(path, 16, distance_type="Cosine" if metric == "angular" else "L2")
    index = ngtpy.Index(path)
    assert index.batch_insert(train, num_threads=1) is None
    index.save()
    index.close()
    adapter.open(params, len(train))
    adapter.configure({"epsilon": 1.0})
    assert adapter.query(train[199:200])[0] == 199
    deleted = {0, 5, 199}
    adapter.delete(sorted(deleted))
    for row in train:
        assert not deleted.intersection(adapter.query(row[None]))
    adapter.replace([7], fresh)
    assert adapter.query(fresh)[0] == 7
    adapter.flush()
    adapter.index.close()
    adapter.open(params, len(train))
    assert adapter.query(fresh)[0] == 7
    adapter.replace([7], train[7:8])
    assert adapter.query(train[7:8])[0] == 7
    adapter.delete([7])
    for row in np.concatenate([train, fresh]):
        assert not (deleted | {7}).intersection(adapter.query(row[None]))
    adapter.index.close()


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
