# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Pydantic AI toolset exposing long-term memory as `remember`, `recall` and `forget` tools."""

from __future__ import annotations

import uuid
from typing import List, Optional

from pydantic_ai.toolsets import FunctionToolset

from ..core import AsyncWarmBackend, WarmBackend, WarmConfig, Record


def _format(hits) -> str:
    if not hits:
        return "No matching memories."
    return "\n".join(f"[{h.id}] {h.text}" for h in hits)


def memory_toolset(
    config: Optional[WarmConfig] = None,
    *,
    user_id: Optional[str] = None,
    sync: bool = False,
    default_k: int = 4,
    max_k: int = 20,
    toolset_id: Optional[str] = "writerslogic-warm",
    **toolset_kwargs,
) -> FunctionToolset:
    """Build a `FunctionToolset` to pass as `Agent(toolsets=[...])`.

    `user_id` is stored as metadata and restricts `recall` and `forget` to that user's
    memories. `sync=True` builds blocking tools on `WarmBackend` (Pydantic AI runs sync tools
    in a worker thread); the default uses `AsyncWarmBackend`. Extra keyword arguments
    (`max_retries`, `timeout`, `requires_approval`, ...) go to `FunctionToolset`.
    """
    scope = {"user_id": user_id} if user_id is not None else {}

    def build(text: str, tags: Optional[List[str]]) -> Record:
        meta = {**scope}
        if tags:
            meta["tags"] = list(tags)
        return Record(uuid.uuid4().hex, text, meta)

    def clamp(k: Optional[int]) -> int:
        return max(1, min(k or default_k, max_k))

    def owned(record: Optional[Record]) -> bool:
        return record is not None and all(record.metadata.get(a) == b for a, b in scope.items())

    toolset = FunctionToolset(id=toolset_id, **toolset_kwargs)

    if sync:
        backend = WarmBackend(config)

        def remember(text: str, tags: Optional[List[str]] = None) -> str:
            """Store a fact or note in long-term memory. Returns the memory id."""
            return backend.remember([build(text, tags)])[0]

        def recall(query: str, k: Optional[int] = None) -> str:
            """Search long-term memory and return the best matches as `[id] text` lines."""
            return _format(backend.recall(query, k=clamp(k), filter=scope or None))

        def forget(memory_id: str) -> str:
            """Delete one memory by id."""
            if not owned(backend.get([memory_id])[0]):
                return "No such memory."
            backend.forget([memory_id])
            return "Deleted."

    else:
        abackend = AsyncWarmBackend(config)

        async def remember(text: str, tags: Optional[List[str]] = None) -> str:  # type: ignore[misc]
            """Store a fact or note in long-term memory. Returns the memory id."""
            return (await abackend.remember([build(text, tags)]))[0]

        async def recall(query: str, k: Optional[int] = None) -> str:  # type: ignore[misc]
            """Search long-term memory and return the best matches as `[id] text` lines."""
            return _format(await abackend.recall(query, k=clamp(k), filter=scope or None))

        async def forget(memory_id: str) -> str:  # type: ignore[misc]
            """Delete one memory by id."""
            if not owned((await abackend.get([memory_id]))[0]):
                return "No such memory."
            await abackend.forget([memory_id])
            return "Deleted."

    for fn in (remember, recall, forget):
        toolset.add_function(fn)
    return toolset
