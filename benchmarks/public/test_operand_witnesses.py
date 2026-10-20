# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Source-backed positive and negative fixtures for the independent checker."""
import unittest

from check_operand_report import check_witness


class WitnessValidation(unittest.TestCase):
    def setUp(self):
        text = 'Alice said “Beta release is ready.” after the launch on 2024/02/28.'
        self.sources = [
            ['a' * 64, 'b' * 64, '2024/03/01', 'user', text],
            ['a' * 64, 'c' * 64, '2024/02/29', 'user', text],
        ]
        self.required = {
            'literal_quotes': ['Beta release is ready.', 'Beta release is wrong.'],
            'date_literals': ['2024-02-28', '2024-02-29', '2024-03-01'],
        }

    def test_accepts_exact_quotation_and_each_date_origin(self):
        for field, session in [
            ('identity:Alice', 'b'),
            ('quotation:Beta release is ready.', 'b'),
            ('literal-date:2024-02-28', 'b'),
            ('observation-date:2024-03-01', 'b'),
            ('observation-date:2024-02-29', 'c'),
        ]:
            with self.subTest(field=field):
                check_witness(field + ':session:' + session * 64, self.sources, self.required)

    def test_rejects_invented_quotation_and_swapped_date_origins(self):
        for field, session in [
            ('identity:Ali', 'b'),
            ('quotation:Beta release is wrong.', 'b'),
            ('quotation:', 'b'),
            ('literal-date:2024-02-29', 'b'),
            ('observation-date:2024-02-28', 'b'),
            ('observation-date:2024-02-29', 'b'),
            ('observation-date:2024-02-29', 'd'),
            ('unknown:Beta release', 'b'),
        ]:
            with self.subTest(field=field, session=session):
                with self.assertRaises(AssertionError):
                    check_witness(field + ':session:' + session * 64, self.sources, self.required)


if __name__ == '__main__':
    unittest.main()
