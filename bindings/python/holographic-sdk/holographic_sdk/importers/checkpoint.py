# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Atomic checkpoint file for resumable imports.

A checkpoint advances only after a batch is acknowledged by the server, and every write
is idempotent (documents replace by id), so a crash at any point loses nothing and a
resume at worst re-sends the last batch.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict

from .model import ImporterError

VERSION = 2


@dataclass
class Checkpoint:
    path: str
    fingerprint: str
    position: int = 0  # source records fully committed (rejected ones included)
    cursor: Any = None  # cursor of the last committed record, for seekable sources
    complete: bool = False
    stats: Dict[str, Any] = field(default_factory=dict)
    prefix_hash: str = ""  # chained hash of committed source identities and contents, in order

    @classmethod
    def open(cls, path: str, fingerprint: str, restart: bool) -> "Checkpoint":
        """Load a matching incomplete checkpoint; start fresh otherwise.

        A checkpoint written for a different source, target or mapping is an error,
        because resuming it would mix two imports."""
        if restart:
            return cls(path, fingerprint)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, ValueError) as exc:
                raise ImporterError(f"checkpoint {path} is unreadable ({exc}); remove it or pass --restart") from exc
            if not isinstance(data, dict):
                raise ImporterError(f"checkpoint {path} must be a JSON object; pass --restart after moving it aside")
            if data.get("version") != VERSION:
                raise ImporterError(f"checkpoint {path} has an unsupported version; remove it or pass --restart")
            if data.get("fingerprint") != fingerprint:
                raise ImporterError(
                    f"checkpoint {path} belongs to a different source, target or mapping options; "
                    "use another --checkpoint path or pass --restart"
                )
            position = data.get("position")
            if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < 1 << 63:
                raise ImporterError(f"checkpoint {path} has an invalid position; pass --restart")
            prefix = data.get("prefix_hash", "")
            if not isinstance(prefix, str) or (position and re.fullmatch(r"[0-9a-f]{64}", prefix) is None):
                raise ImporterError(f"checkpoint {path} has an invalid content fingerprint; pass --restart")
            if not data.get("complete"):
                return cls(path, fingerprint, position, data.get("cursor"), False,
                           data.get("stats", {}), data.get("prefix_hash", ""))
        return cls(path, fingerprint)

    def save(self) -> None:
        data = {
            "version": VERSION,
            "fingerprint": self.fingerprint,
            "position": self.position,
            "cursor": self.cursor,
            "complete": self.complete,
            "stats": self.stats,
            "prefix_hash": self.prefix_hash,
        }
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".ckpt-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, default=str)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
            if os.name != "nt":
                directory_fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
