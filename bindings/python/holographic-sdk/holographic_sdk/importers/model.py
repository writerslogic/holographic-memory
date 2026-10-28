# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Data model shared by every importer source and the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


class ImporterError(Exception):
    """Fatal import failure with a message that is safe to print."""


class MissingDependency(ImporterError):
    """An optional client library for a source is not installed."""

    def __init__(self, source: str, package: str, extra: Optional[str] = None):
        hint = f"uv add 'holographic-memory-sdk[{extra}]'" if extra else f"uv add {package}"
        super().__init__(f"source '{source}' needs the '{package}' package ({hint})")


@dataclass
class Record:
    """One item read from a source, before mapping to an HMS document.

    `vector` is whatever the source stores (any float sequence or numpy array); `None`
    when the source holds no embedding. `cursor` is an opaque JSON value; seekable
    sources accept the cursor of the last committed record to continue after it.
    """

    id: str
    text: Optional[str] = None
    vector: Optional[Sequence[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    namespace: str = ""
    created_at: Any = None
    updated_at: Any = None
    cursor: Any = None
    error: Optional[str] = None  # set when the source could not decode this item; it is rejected, not lost


@dataclass
class HmsDoc:
    """A document as sent to the server (`id`, optional `text`, `vector`, `metadata`)."""

    id: str
    text: Optional[str]
    vector: Optional[List[float]]
    metadata: Dict[str, Any]
    tenant: Optional[str] = None
    source_key: Any = None

    def payload(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"id": self.id, "metadata": self.metadata}
        if self.text:
            out["text"] = self.text
        if self.vector is not None:
            out["vector"] = self.vector
        return out


class Reject(Exception):
    """A record that cannot be imported; carries a stable reason code."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail
