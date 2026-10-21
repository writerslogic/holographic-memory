# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""smolagents tools: `remember`, `recall`, `forget` over Holographic Memory.

smolagents tools are synchronous (`Tool.forward`), so these use `HmsBackend`; there is no
async tool path in the framework.
"""

from __future__ import annotations

import uuid
from typing import List, Optional

from smolagents import Tool

from ..core import HmsBackend, HmsConfig, Record


class _MemoryTool(Tool):
    """Shared state; not registered as a tool itself."""

    def __init__(self, backend: HmsBackend, scope: dict, default_k: int, max_k: int):
        self.backend, self.scope, self.default_k, self.max_k = backend, scope, default_k, max_k
        super().__init__()


class RememberTool(_MemoryTool):
    name = "remember"
    description = "Store a fact or note in long-term memory. Returns the memory id."
    inputs = {
        "text": {"type": "string", "description": "The fact or note to remember."},
        "tags": {"type": "array", "description": "Optional list of tag strings.", "nullable": True},
    }
    output_type = "string"

    def forward(self, text: str, tags: Optional[List[str]] = None) -> str:
        meta = dict(self.scope)
        if tags:
            meta["tags"] = [str(t) for t in tags]
        return self.backend.remember([Record(uuid.uuid4().hex, text, meta)])[0]


class RecallTool(_MemoryTool):
    name = "recall"
    description = "Search long-term memory. Returns the best matches, one `[id] text` per line."
    inputs = {
        "query": {"type": "string", "description": "What to look for."},
        "k": {"type": "integer", "description": "How many matches to return.", "nullable": True},
    }
    output_type = "string"

    def forward(self, query: str, k: Optional[int] = None) -> str:
        k = max(1, min(k or self.default_k, self.max_k))
        hits = self.backend.recall(query, k=k, filter=self.scope or None)
        if not hits:
            return "No matching memories."
        return "\n".join(f"[{h.id}] {h.text}" for h in hits)


class ForgetTool(_MemoryTool):
    name = "forget"
    description = "Delete one memory by id."
    inputs = {"memory_id": {"type": "string", "description": "Id returned by remember or recall."}}
    output_type = "string"

    def forward(self, memory_id: str) -> str:
        record = self.backend.get([memory_id])[0]
        if record is None or any(record.metadata.get(a) != b for a, b in self.scope.items()):
            return "No such memory."
        self.backend.forget([memory_id])
        return "Deleted."


def memory_tools(
    config: Optional[HmsConfig] = None,
    *,
    backend: Optional[HmsBackend] = None,
    user_id: Optional[str] = None,
    default_k: int = 4,
    max_k: int = 20,
) -> List[Tool]:
    """The three tools, sharing one backend. `user_id` scopes `recall` and `forget` to one user."""
    backend = backend or HmsBackend(config)
    scope = {"user_id": user_id} if user_id is not None else {}
    return [cls(backend, scope, default_k, max_k) for cls in (RememberTool, RecallTool, ForgetTool)]
