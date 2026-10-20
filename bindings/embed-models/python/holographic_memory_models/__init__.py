# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Model packs for holographic_memory.Embedder; installed by `pip install holographic-vsa[embed]`."""
from pathlib import Path

__all__ = ["pack_dir"]


def pack_dir() -> str:
    """Directory holding the verified `<id>-int8.hmsemb` packs (read by holographic_memory.Embedder)."""
    return str(Path(__file__).resolve().parent / "packs")
