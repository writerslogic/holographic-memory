# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check stream boundaries using a retained dev source."""

import json
import unittest
from pathlib import Path

from longmemeval_incremental_judges import completed_records


class StreamBoundaries(unittest.TestCase):
    def test_unterminated_dev_record_is_never_imported(self):
        report = json.loads(Path("benchmarks/results/longmemeval_dev_answers_v1.json").read_text())
        row = report["rows"][0]
        source = row["systems"]["weak_qwen4b"]["arms"]["first_five"]["retained_sources"][0]
        record = {"qid": row["qid"], "source": source}
        encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode()
        for size, expected in ((len(encoded) - 1, []), (len(encoded), [record]),
                               (len(encoded) + 1, [record])):
            with self.subTest(size=size):
                data = (encoded + b"{")[:size]
                self.assertEqual(completed_records(data)[1], expected)

    def test_incomplete_utf8_tail_does_not_hide_completed_dev_record(self):
        report = json.loads(Path("benchmarks/results/longmemeval_dev_answers_v1.json").read_text())
        record = {"qid": report["rows"][0]["qid"]}
        encoded = (json.dumps(record) + "\n").encode()
        self.assertEqual(completed_records(encoded + b'{"quote":"\xe2\x80')[1], [record])

    def test_malformed_completed_record_stops_the_import(self):
        with self.assertRaises(json.JSONDecodeError):
            completed_records(b"{\n")


if __name__ == "__main__":
    unittest.main()
