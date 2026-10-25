# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native build/serve contract; set HMS_PUBLIC_BENCH to a built public-bench binary."""

import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmsbench import common, runner  # noqa: E402


@unittest.skipUnless(os.environ.get("HMS_PUBLIC_BENCH"), "set HMS_PUBLIC_BENCH for the native integration test")
class HmsAnnBuildContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        resources = contextlib.ExitStack()
        cls.addClassCleanup(resources.close)
        configured = os.environ.get("HMS_BENCH_LOCK")
        if configured:
            lock_path = Path(configured)
        else:
            temporary = resources.enter_context(tempfile.TemporaryDirectory(prefix="hms-ann-test-lock-"))
            lock_path = Path(temporary) / "timing.lock"
            lock_path.touch()
        handle = resources.enter_context(lock_path.open("rb"))
        fcntl.flock(handle, fcntl.LOCK_EX)
        resources.enter_context(patch.object(common, "timing_lock", contextlib.nullcontext))

    def exercise(self, metric):
        binary = Path(os.environ["HMS_PUBLIC_BENCH"]).resolve()
        self.assertTrue(binary.is_file())
        with tempfile.TemporaryDirectory(prefix="hms-ann-contract-") as temporary:
            root = Path(temporary)
            directory = root / "data"
            directory.mkdir()
            train = np.random.default_rng(24301).standard_normal((256, 8)).astype("<f4")
            queries = train[::32].copy()
            x, q = train.astype(np.float64), queries.astype(np.float64)
            if metric == "angular":
                scores = -(q @ x.T) / (np.linalg.norm(q, axis=1)[:, None] * np.linalg.norm(x, axis=1))
            else:
                scores = np.sum((q[:, None] - x[None]) ** 2, axis=2)
            truth = np.argsort(scores, axis=1, kind="stable")[:, :10].astype("<i4")
            train.tofile(directory / "train.f32")
            queries.tofile(directory / "test.f32")
            truth.tofile(directory / "neighbors.i32")
            meta = {"kind": "ann", "name": "native-contract-" + metric, "metric": metric,
                    "n_train": len(train), "n_test": len(queries), "dim": train.shape[1], "n_neighbors": 10}
            (directory / "meta.json").write_text(json.dumps(meta))
            captured = []
            native_run = common.run_measured

            def capture(command, **kwargs):
                stdout, stderr, wall, peak = native_run(command, **kwargs)
                report = json.loads(Path(command[command.index("--out") + 1]).read_text())
                captured.append({"argv": command, "report": report, "stdout": stdout,
                                 "stderr": stderr, "wall_s": wall, "peak_rss_bytes": peak})
                return stdout, stderr, wall, peak

            workdir = root / "index"
            with patch.object(runner, "hms_binary", return_value=binary), patch.object(runner, "WORK", root / "work"), \
                    patch.object(runner, "run_measured", capture):
                build = runner._hms_ann_build(directory, meta, 2, workdir)
                index = workdir / "index.vgraph"
                self.assertTrue(index.is_file())
                self.assertTrue((workdir / "index.updates").is_dir())
                sidecar = json.loads(Path(str(index) + ".json").read_text())
                self.assertEqual(set(sidecar), {"graph_hash", "build_params", "dataset", "build_secs"})
                self.assertEqual(sidecar["dataset"], meta)
                self.assertEqual(build["build_s"], sidecar["build_secs"])
                self.assertEqual(build["graph_hash"], sidecar["graph_hash"])
                self.assertEqual(build["hms_threads"], 2)
                self.assertGreater(build["peak_rss_bytes"], 0)
                self.assertGreater(build["in_memory_index_bytes"], build["residual_bytes"])
                self.assertTrue(build["memory_phases"])
                raw = captured[0]["report"]
                self.assertEqual(raw["sweep"], [])
                self.assertEqual(raw["index_file"]["saved"], str(index))
                self.assertEqual(raw["index_file"]["bytes"], index.stat().st_size)
                self.assertEqual(raw["build"]["build_secs"], build["build_s"])
                self.assertEqual(raw["memory_phases"], build["memory_phases"])
                for key in ("degree", "build_ef", "vertex_bits", "residual_bits", "quantized_search"):
                    self.assertEqual(raw["build"]["params"][key], runner.HMS_ANN_BUILDS[metric][key])
                (directory / "train.f32").rename(directory / "train.hidden")
                try:
                    served = runner._hms_ann_round(directory, meta, workdir)
                finally:
                    (directory / "train.hidden").rename(directory / "train.f32")
                self.assertEqual(served["n"], len(train))
                self.assertEqual(served["n_queries"], len(queries))
                expected = {(int(ef), int(rerank)) for ef in runner.HMS_ANN_BUILDS[metric]["ef"].split(",")
                            for rerank in runner.HMS_ANN_BUILDS[metric]["rerank"].split(",")}
                observed = {(r["params"]["ef"], r["params"]["rerank"]) for r in served["configs"]}
                self.assertEqual(observed, expected)
                self.assertGreaterEqual(max(r["recall_at_10"] for r in served["configs"]), 0.99)
                for row in served["configs"]:
                    self.assertTrue(0.0 <= row["recall_at_10"] <= 1.0)
                    self.assertEqual(len(row["repeats"]), 2)
                    for timing in row["repeats"]:
                        self.assertTrue(math.isfinite(timing["qps"]) and timing["qps"] > 0)
                self.assertTrue(served["memory_phases"])
                self.assertGreater(served["peak_rss_bytes"], 0)
            evidence = os.environ.get("HMS_ANN_TEST_EVIDENCE")
            if evidence:
                destination = Path(evidence) / metric
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copytree(directory, destination / "data", dirs_exist_ok=True)
                shutil.copytree(workdir, destination / "index", dirs_exist_ok=True)
                artifacts = [{"path": str(p.relative_to(destination)), "bytes": p.stat().st_size,
                              "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                             for parent in (destination / "data", destination / "index")
                             for p in sorted(parent.rglob("*")) if p.is_file()]
                (destination / "native-results.json").write_text(json.dumps({"binary": str(binary),
                    "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "calls": captured,
                    "build_summary": build, "serve_summary": served, "artifacts": artifacts}, indent=2) + "\n")

    def test_angular_build_persists_and_serves_without_training_rows(self):
        self.exercise("angular")

    def test_euclidean_build_persists_and_serves_without_training_rows(self):
        self.exercise("euclidean")


if __name__ == "__main__":
    unittest.main()
