# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Path validation for the /memories virtual filesystem.

Paths are storage keys, never real filesystem paths, but every path is still validated so
nothing outside /memories is ever addressed.
"""

from __future__ import annotations

import re
from urllib.parse import unquote

ROOT = "/memories"
MAX_PATH_BYTES = 240
MAX_SEGMENT_BYTES = 128
MAX_DEPTH = 16

_ENCODED = re.compile(r"%(2e|2f|5c|00)", re.IGNORECASE)
_FORBIDDEN = re.compile(r"[\x00-\x1f\x7f\\]")


class MemoryPathError(ValueError):
    """A path outside /memories or otherwise unacceptable."""


def normalize_path(path: object) -> str:
    """Return the canonical form of ``path``: ROOT, or ROOT/segment/..., no trailing slash."""
    if not isinstance(path, str) or not path:
        raise MemoryPathError("Path must be a non-empty string")
    # Percent-encoded separators and dots are rejected outright so a path that a later layer
    # decodes cannot turn into a traversal.
    if _ENCODED.search(path):
        raise MemoryPathError(f"Path {path} contains an encoded separator or dot")
    for candidate in (path, unquote(path)):
        if _FORBIDDEN.search(candidate):
            raise MemoryPathError(f"Path {path} contains a control character or backslash")
        if any(segment in ("..", ".") for segment in candidate.split("/")):
            raise MemoryPathError(f"Path {path} would escape /memories directory")
    if path != ROOT and not path.startswith(ROOT + "/"):
        raise MemoryPathError(f"Path must start with /memories, got: {path}")
    trimmed = path.rstrip("/") if len(path) > len(ROOT) else path
    if trimmed == ROOT:
        return ROOT
    segments = trimmed[len(ROOT) + 1 :].split("/")
    if len(segments) > MAX_DEPTH:
        raise MemoryPathError(f"Path {path} is nested more than {MAX_DEPTH} levels")
    for segment in segments:
        if not segment:
            raise MemoryPathError(f"Path {path} contains an empty segment")
        if segment.startswith("."):
            raise MemoryPathError(f"Path {path} contains a hidden segment; names must not start with a dot")
        if len(segment.encode()) > MAX_SEGMENT_BYTES:
            raise MemoryPathError(f"Path {path} has a segment longer than {MAX_SEGMENT_BYTES} bytes")
    if len(trimmed.encode()) > MAX_PATH_BYTES:
        raise MemoryPathError(f"Path {path} is longer than {MAX_PATH_BYTES} bytes")
    return trimmed


def ancestors(path: str) -> list[str]:
    """Every proper ancestor of ``path`` below ROOT, shallowest first."""
    parts = path[len(ROOT) + 1 :].split("/")
    return [f"{ROOT}/{'/'.join(parts[:i])}" for i in range(1, len(parts))]


def is_inside(path: str, directory: str) -> bool:
    return path == directory or path.startswith(directory + "/")
