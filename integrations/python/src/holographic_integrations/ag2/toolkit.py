# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Semantic memory tools for AG2 (the `ag2` 1.x package)."""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Union

from ag2.tools import Toolkit, tool

from holographic_integrations.core import AsyncHmsBackend, HmsConfig, Record

Scalar = Union[str, int, float, bool]


class HolographicMemoryToolkit(Toolkit):
    """`remember`, `recall` and `forget` tools backed by Holographic Memory.

    Pass it in `Agent(tools=[HolographicMemoryToolkit(...)])`. Tool results are plain
    JSON-serializable values. `recall` filters on the metadata given to `remember` (equality
    is evaluated by the service; see docs for the other operators). Set `allow_forget=False`
    to leave out the destructive tool.

    This is a tool-level adapter, not an `ag2.knowledge.KnowledgeStore`: that protocol is a
    path-addressed file store with atomic `append` and unbounded byte-range reads, which a
    64 KiB-per-document vector service cannot honor. See docs/integrations/frameworks/autogen.md.
    """

    def __init__(
        self,
        hms: Optional[HmsConfig] = None,
        *,
        k: int = 5,
        allow_forget: bool = True,
        name: str = "holographic_memory",
    ) -> None:
        if hms is None:
            hms = HmsConfig(api_key=os.environ.get("HMS_API_KEY"))
        if hms.collection is None:
            hms.collection = name
        self.backend = AsyncHmsBackend(hms)
        default_k = k
        backend = self.backend

        @tool(name="remember", description="Store a fact or note in long-term memory. Returns its id.")
        async def remember(text: str, metadata: Optional[Dict[str, Scalar]] = None) -> str:
            rid = str(uuid.uuid4())
            await backend.remember([Record(rid, text, dict(metadata or {}))])
            return rid

        @tool(
            name="recall",
            description="Search long-term memory. Returns the best matches as {id, text, metadata, score}.",
        )
        async def recall(
            query: str, k: Optional[int] = None, filter: Optional[Dict[str, Scalar]] = None
        ) -> List[Dict[str, Any]]:
            hits = await backend.recall(query, k=k or default_k, filter=filter)
            return [{"id": h.id, "text": h.text, "metadata": h.metadata, "score": h.score} for h in hits]

        tools = [remember, recall]
        if allow_forget:

            @tool(name="forget", description="Delete a memory by id. Returns whether it existed.")
            async def forget(id: str) -> bool:
                return bool(await backend.forget([id]))

            tools.append(forget)
        super().__init__(*tools, name=name)

    def __deepcopy__(self, memo: Dict[int, Any]) -> "HolographicMemoryToolkit":
        # AG2 deep-copies tools when an agent adopts them; the toolkit owns a live HTTP
        # client that cannot be copied, and its tools hold no mutable per-agent state.
        return self

    async def aclose(self) -> None:
        await self.backend.aclose()
