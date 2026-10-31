# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Long-term memory tools (`remember`, `recall`, `forget`) for OpenAI Agents SDK agents."""

from __future__ import annotations

import uuid
from typing import List, Optional

from agents import FunctionTool, function_tool

from ..core import AsyncWarmBackend, WarmBackend, WarmConfig, Record


def _format(hits) -> str:
    if not hits:
        return "No matching memories."
    return "\n".join(f"[{h.id}] {h.text}" for h in hits)


def memory_tools(
    config: Optional[WarmConfig] = None,
    *,
    user_id: Optional[str] = None,
    sync: bool = False,
    default_k: int = 4,
    max_k: int = 20,
) -> List[FunctionTool]:
    """Return `remember`, `recall` and `forget` tools.

    `user_id` is stored as metadata and restricts `recall` to that user's memories, so one
    store can serve many users. `sync=True` builds blocking tools on `WarmBackend` (the SDK
    runs them in the event loop thread); the default uses `AsyncWarmBackend`.
    """
    scope = {"user_id": user_id} if user_id is not None else {}

    def remember_record(text: str, tags: Optional[List[str]]) -> Record:
        meta = {**scope}
        if tags:
            meta["tags"] = list(tags)
        return Record(uuid.uuid4().hex, text, meta)

    def clamp(k: Optional[int]) -> int:
        return max(1, min(k or default_k, max_k))

    if sync:
        backend = WarmBackend(config)

        @function_tool
        def remember(text: str, tags: Optional[List[str]] = None) -> str:
            """Store a fact or note in long-term memory. Returns the memory id."""
            return backend.remember([remember_record(text, tags)])[0]

        @function_tool
        def recall(query: str, k: Optional[int] = None) -> str:
            """Search long-term memory and return the best matches as `[id] text` lines."""
            return _format(backend.recall(query, k=clamp(k), filter=scope or None))

        @function_tool
        def forget(memory_id: str) -> str:
            """Delete one memory by id."""
            owned = backend.get([memory_id])[0]
            if owned is None or any(owned.metadata.get(a) != b for a, b in scope.items()):
                return "No such memory."
            backend.forget([memory_id])
            return "Deleted."

        return [remember, recall, forget]

    abackend = AsyncWarmBackend(config)

    @function_tool
    async def remember(text: str, tags: Optional[List[str]] = None) -> str:  # noqa: F811
        """Store a fact or note in long-term memory. Returns the memory id."""
        return (await abackend.remember([remember_record(text, tags)]))[0]

    @function_tool
    async def recall(query: str, k: Optional[int] = None) -> str:  # noqa: F811
        """Search long-term memory and return the best matches as `[id] text` lines."""
        return _format(await abackend.recall(query, k=clamp(k), filter=scope or None))

    @function_tool
    async def forget(memory_id: str) -> str:  # noqa: F811
        """Delete one memory by id."""
        owned = (await abackend.get([memory_id]))[0]
        if owned is None or any(owned.metadata.get(a) != b for a, b in scope.items()):
            return "No such memory."
        await abackend.forget([memory_id])
        return "Deleted."

    return [remember, recall, forget]
