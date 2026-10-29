# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Value types shared by every adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

RESERVED_PREFIX = "_hms_"
ID_KEY = "_hms_id"
COLLECTION_KEY = "_hms_collection"


class HmsError(RuntimeError):
    """The backend refused or failed a request. `status` and `code` come from the service."""

    def __init__(self, message: str, status: Optional[int] = None, code: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class Record:
    """One stored item. `vector` is optional: the backend embeds `text` when it is absent."""

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[List[float]] = None


@dataclass(frozen=True)
class Hit:
    """A search result. `score` is cosine similarity in vector mode and the fused rank score in text mode."""

    id: str
    text: str
    metadata: Dict[str, Any]
    score: float


@dataclass(frozen=True)
class Page:
    """A bounded enumeration; `truncated` means additional matching records exist."""

    records: List[Record]
    truncated: bool
