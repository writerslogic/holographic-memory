# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = "==3.13.9"
# dependencies = ["numpy==2.2.6"]
# ///

import contextlib
import copy
import gzip
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

import ann_failure_map as experiment


class EvidenceTampering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_out = experiment.OUT
        cls.dataset = "glove-100-angular"
        cls.source = cls.original_out / cls.dataset
        with gzip.open(cls.source / "raw.json.gz", "rt") as handle:
            cls.raw = json.load(handle)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        experiment.OUT = Path(self.temporary.name)
        self.destination = experiment.OUT / self.dataset
        self.destination.mkdir()
        for name in ("train.f32", "queries.f32", "train_original_ids.npy", "queries_original_ids.npy", "inputs.json"):
            shutil.copyfile(self.source / name, self.destination / name)

    def tearDown(self):
        experiment.OUT = self.original_out
        self.temporary.cleanup()

    def check(self, raw):
        with gzip.open(self.destination / "raw.json.gz", "wt") as handle:
            json.dump(raw, handle)
        with contextlib.redirect_stdout(io.StringIO()):
            experiment.check(self.dataset)

    def test_pristine_and_corruptions(self):
        self.check(self.raw)
        for corruption in ("missing", "duplicate", "id", "membership", "pool_prefix", "flat_truth"):
            with self.subTest(corruption=corruption):
                raw = copy.deepcopy(self.raw)
                records = raw["records"]
                if corruption == "missing":
                    records.pop()
                elif corruption == "duplicate":
                    records.append(records[0])
                elif corruption == "id":
                    records[0]["ids"][0] = 10000
                else:
                    arm = "flat" if corruption == "flat_truth" else "original16"
                    for row in records:
                        if row["arm"] == arm:
                            if corruption == "membership":
                                row["ids"][0] = next(i for i in range(10000) if i not in row["pool_ids"])
                            elif corruption == "pool_prefix":
                                row["pool_ids"].reverse()
                            else:
                                row["ids"] = list(range(10))
                            break
                with self.assertRaises(ValueError):
                    self.check(raw)

    def test_nonfinite_input_rejected_even_with_matching_hash(self):
        path = self.destination / "train.f32"
        values = np.fromfile(path, dtype="<f4")
        values[-1] = np.nan
        values.tofile(path)
        manifest_path = self.destination / "inputs.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["train.f32"] = experiment.sha(path)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "nonfinite diagnostic input"):
            self.check(self.raw)


if __name__ == "__main__":
    unittest.main()
