# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import gzip
import hashlib


ORIGINAL_IMPORT = b"from probe import PhaseSketch, coverage, generate, ordering, record, reservoir_scores, squared_l2, timed"
CLEAN_IMPORT = b"from probe import PhaseSketch, generate, ordering, record, reservoir_scores, squared_l2, timed"


def verify_source(path, expected):
    current = path.read_bytes()
    if hashlib.sha256(current).hexdigest() == expected:
        return
    if path.name == "hybrid_probe.py":
        original = gzip.decompress(path.with_name("frozen-hybrid-probe.py.gz").read_bytes())
        if (hashlib.sha256(original).hexdigest() == expected
                and original.count(ORIGINAL_IMPORT) == 1
                and current == original.replace(ORIGINAL_IMPORT, CLEAN_IMPORT)):
            return
    raise ValueError(f"source differs from its recorded hash: {path.name}")
