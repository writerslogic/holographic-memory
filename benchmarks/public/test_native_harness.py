# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy==2.2.6", "h5py==3.14.0", "faiss-cpu==1.15.1", "hnswlib==0.8.0"]
# ///
import importlib.util
import copy
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("native_harness", Path(__file__).with_name("native_harness.py"))
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)


class NativeBoundaryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HARNESS.check_environment()
        cls.bridge = HARNESS.import_bridge(os.environ["WARM_NATIVE_BRIDGE"])
        data = Path.home() / ".cache/warm-bench/glove-100-angular/train.f32"
        cls.raw = np.fromfile(data, dtype="<f4", count=10256 * 100).reshape(-1, 100)
        cls.base = HARNESS.normalized(cls.raw[:10240])
        cls.queries = HARNESS.normalized(cls.raw[10240:], query=True)
        cls.config = {"degree": 32, "build_ef": 64, "alpha": 1.0, "seed": 0x5EED,
                      "bits": 8, "residual_bits": 8, "align": True, "reorder": False, "id_bytes": 3}
        cls.index = cls.bridge.NativeIndex.build(cls.base[:512], HARNESS.canonical(cls.config))

    def test_native_persistence_and_batch_have_identical_real_vector_results(self):
        actual = self.index.query(self.queries, 10, 128, 32, 0)
        individual = np.concatenate([self.index.query(q[None], 10, 128, 32, 0) for q in self.queries])
        np.testing.assert_array_equal(actual, individual)
        self.assertEqual(actual.dtype, np.dtype("int64"))
        self.assertEqual(actual.shape, (16, 10))
        with tempfile.TemporaryDirectory(dir=os.environ["WARM_NATIVE_TEST_DIR"]) as directory:
            path = Path(directory) / "index.bin"
            self.index.save(str(path))
            loaded = self.bridge.NativeIndex.load(str(path))
            np.testing.assert_array_equal(loaded.query(self.queries, 10, 128, 32, 0), actual)
            self.assertEqual(loaded.graph_hash, self.index.graph_hash)
            original = path.read_bytes()
            for invalid in [b"badmagic" + original[8:], original + b"x", original[:16]]:
                path.write_bytes(invalid)
                with self.assertRaises(OSError):
                    self.bridge.NativeIndex.load(str(path))

    def test_query_boundary_rejects_invalid_input_and_recovers(self):
        baseline = self.index.query(self.queries[:1], 10, 128, 32, 0)
        invalid = [np.asfortranarray(self.queries), self.queries[:, :-1], np.zeros_like(self.queries[:1])]
        nonfinite = self.queries[:1].copy()
        nonfinite[0, 0] = np.inf
        invalid.append(nonfinite)
        for query in invalid:
            with self.assertRaises(ValueError):
                self.index.query(query, 10, 128, 32, 0)
        for k in [0, 513]:
            with self.assertRaises(ValueError):
                self.index.query(self.queries[:1], k, 128, 32, 0)
        self.assertEqual(self.index.query(self.queries[:1], 512, 512, 512, 0).shape, (1, 512))
        np.testing.assert_array_equal(self.index.query(self.queries[:1], 10, 128, 32, 0), baseline)

    def test_native_competitor_boundaries_persist_without_label_changes(self):
        configurations = {
            "faiss_hnsw": ({"M": 16, "ef_construction": 200, "seed": HARNESS.SEED}, {"ef": 128}),
            "hnswlib": ({"M": 16, "ef_construction": 200, "seed": HARNESS.SEED}, {"ef": 128}),
            "faiss_ivfpq": ({"nlist": 32, "pq_m": 25, "pq_bits": 8, "train_rows": 10240, "niter": 20,
                              "seed": HARNESS.SEED, "metric": "inner_product", "by_residual": True}, {"nprobe": 32}),
        }
        for system, (build, search) in configurations.items():
            with self.subTest(system=system), tempfile.TemporaryDirectory(dir=os.environ["WARM_NATIVE_TEST_DIR"]) as directory:
                adapter = HARNESS.Adapter(system, build, self.bridge)
                adapter.fit(self.base if system == "faiss_ivfpq" else self.base[:512])
                adapter.configure(search)
                found = np.asarray(adapter.query(self.queries, 10), dtype=np.int64)
                self.assertEqual(found.shape, (16, 10))
                path = Path(directory) / "index.bin"
                adapter.save(path)
                loaded = HARNESS.Adapter(system, build, self.bridge)
                loaded.load(path, 100)
                loaded.configure(search)
                np.testing.assert_array_equal(np.asarray(loaded.query(self.queries, 10), dtype=np.int64), found)
                self.assertTrue(all(len(set(row)) == 10 for row in found))

    def test_recovery_retains_first_gated_attempt_and_rejects_changed_predictions(self):
        labels = self.index.query(self.queries, 10, 128, 32, 0)
        with tempfile.TemporaryDirectory(dir=os.environ["WARM_NATIVE_TEST_DIR"]) as directory:
            predictions = HARNESS.save_array(labels, directory)
            row = {"point_id": "real-glove-functional-check", "round": 0, "mode": "single_query_loop",
                   "query_start": 0, "query_end": 16, "attempt": 0, "lock_id": "first",
                   "predictions": predictions, "timer": {"start_ns": 10, "end_ns": 20}, "load_gate_met": True}
            scored = {**row, "recall_at_10": 0.5, "strict_id_recall_at_10": 0.5,
                      "distances": {"sha256": "functional-score"}, "hits": {"sha256": "functional-score"}}
            later = {**scored, "attempt": 1, "lock_id": "retry", "timer": {"start_ns": 40, "end_ns": 41}}
            locks = [{"lock_id": name, "status": "released", "exit_code": 0, "under_15_minutes": True,
                      "timer": {"start_ns": start, "end_ns": end}}
                     for name, start, end in [("first", 1, 30), ("retry", 35, 50)]]
            selected, attempts, preliminary = HARNESS.select_attempts([later, row, scored], locks)
            self.assertEqual(selected, [scored])
            self.assertEqual(len(attempts), 2)
            self.assertEqual(preliminary[0]["retention_reason"], "earlier_valid_attempt_retained")
            self.assertTrue(preliminary[0]["preliminary"])
            damaged = copy.deepcopy(scored)
            damaged["predictions"]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                HARNESS.select_attempts([row, damaged], locks)
            loaded = {**scored, "load_gate_met": False}
            self.assertEqual(HARNESS.select_attempts([later, loaded], locks)[0], [later])

    def test_curve_headroom_uses_two_next_depths_on_real_measured_grids(self):
        configurations = {
            "warm": (512, self.config, [{"ef": depth, "rerank": rerank, "patience": 0}
                                      for depth in [16, 32, 64, 96, 128, 192, 256] for rerank in [16, 32]]),
            "faiss_ivfpq": (10240, {"nlist": 32, "pq_m": 25, "pq_bits": 8, "train_rows": 10240, "niter": 20,
                                   "seed": HARNESS.SEED, "metric": "inner_product", "by_residual": True},
                              [{"nprobe": depth} for depth in [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]]),
        }
        for system, (count, build, settings) in configurations.items():
            with self.subTest(system=system):
                adapter = HARNESS.Adapter(system, build, self.bridge)
                if system == "warm":
                    adapter.index = self.index
                else:
                    adapter.fit(self.base[:count])
                flat = HARNESS.faiss.IndexFlatIP(100)
                flat.add(self.base[:count])
                _, truth = flat.search(self.queries, 10)
                distances = HARNESS.exact_distances(self.raw[:count], self.raw[10240:], truth)
                points = []
                for search in settings:
                    adapter.configure(search)
                    found = np.asarray(adapter.query(self.queries, 10), dtype="<i8")
                    score = HARNESS.scores(self.raw[:count], self.raw[10240:], found, truth, distances)
                    points.append({"point_id": HARNESS.fingerprint(search), "search_config": search,
                                   "recall_at_10": score["recall_at_10"]})
                grid = [{key: value for key, value in point.items() if key != "recall_at_10"} for point in points]
                target = min(point["recall_at_10"] for point in points)
                seeds = HARNESS.select_curve_points(points, grid, [target], 0, 0)
                chosen = HARNESS.select_curve_points(points, grid, [target], 0, 2)
                expected = {point["point_id"] for point in seeds}
                for seed in seeds:
                    search = seed["search_config"]
                    key = "ef" if system == "warm" else "nprobe"
                    siblings = [point for point in grid if all(point["search_config"][name] == value
                                for name, value in search.items() if name != key)]
                    larger = sorted((point for point in siblings if point["search_config"][key] > search[key]),
                                    key=lambda point: point["search_config"][key])
                    expected.update(point["point_id"] for point in larger[:2])
                self.assertEqual({point["point_id"] for point in chosen}, expected)
                self.assertGreater(len(chosen), len(seeds))
                if system == "faiss_ivfpq":
                    self.assertLess(len(chosen), len(grid))


if __name__ == "__main__":
    unittest.main()
