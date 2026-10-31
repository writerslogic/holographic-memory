# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy==2.2.6", "lz4==4.4.5"]
# ///
import copy
import json
from pathlib import Path
import subprocess
import unittest

import check_sparse_composition as checker

BINARY = Path('./.warm-target/release/examples/sparse_composition')


class WitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = {"id": "test", "text": "word0000 rare0003 word0000", "seed": 11,
                    "rare_position": 1, "rare_id": 3}
        request = {k: v for k, v in cls.meta.items() if k != "rare_id"} | {"dim": 1024}
        cls.row = json.loads(subprocess.check_output([str(BINARY)], input=json.dumps(request).encode() + b'\n'))

    def test_native_reconstruction(self):
        vectors, metrics = checker.check_row(self.row, self.meta, 1024)
        self.assertEqual(len(vectors), 23)
        self.assertEqual(len(metrics), 24)

    def test_tampering(self):
        mutations = [
            lambda r: r.update(id="other"), lambda r: r.update(seed=23),
            lambda r: r.update(dim=4096), lambda r: r.update(schema=2),
            lambda r: r.update(rare_position=2), lambda r: r["ns"].__setitem__(0, 999),
            lambda r: r["rows"].pop(), lambda r: r["rows"].append(r["rows"][0]),
            lambda r: r["rows"][0].update(indices=[1023]),
            lambda r: r["rows"][0].update(delta_varint_bytes=0),
            lambda r: r["rows"][0]["rare_intersections"].__setitem__(3, 999),
            lambda r: r.update(count_payload_bytes=1),
            lambda r: r.update(counts_lz4_hex="00000000"),
            lambda r: r.update(cb_count_payload_bytes=0),
            lambda r: r["cb_scores"].__setitem__(0, 999),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                row = copy.deepcopy(self.row)
                mutate(row)
                with self.assertRaises((AssertionError, ValueError, RuntimeError)):
                    checker.check_row(row, self.meta, 1024)

    def test_edges(self):
        for dim in (255, 256, 257):
            for i, text in enumerate(("", " ", "x", "e\u0301", "東京🧠", "a b c d")):
                meta = {"id": str(i), "text": text, "seed": 23, "rare_position": 0, "rare_id": None}
                request = {k: v for k, v in meta.items() if k != "rare_id"} | {"dim": dim}
                row = json.loads(subprocess.check_output([str(BINARY)], input=json.dumps(request).encode() + b'\n'))
                checker.check_row(row, meta, dim)


if __name__ == '__main__':
    unittest.main()
