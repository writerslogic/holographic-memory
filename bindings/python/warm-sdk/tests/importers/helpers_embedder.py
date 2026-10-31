# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Deterministic fake text encoder for tests of text-only sources.

Use it as `embedder="tests.importers.helpers_embedder:embed"`-style: tests put this module on
`sys.modules` under a stable name or pass the dotted path of a package that is importable.
"""

import random
import zlib
from typing import List

DIM = 32


def embed(texts: List[str]) -> List[List[float]]:
    """Same text, same vector; different texts are (with overwhelming probability) far apart."""
    out = []
    for t in texts:
        rng = random.Random(zlib.crc32(t.encode("utf-8")))
        out.append([rng.uniform(-1, 1) for _ in range(DIM)])
    return out
