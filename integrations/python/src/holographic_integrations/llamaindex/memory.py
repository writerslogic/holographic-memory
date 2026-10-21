# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""A `BaseMemoryBlock` that keeps conversation turns in Holographic Memory.

llama-index-core already ships `VectorMemoryBlock`, which works unchanged with
`HolographicVectorStore`. This block removes the LlamaIndex embedding model from the loop: it
uses the HMS encoder (`HmsConfig.embedder`, or the service's own models with
`text_queries=True`), stores one record per message and scopes recall by `session_id`.
"""

from __future__ import annotations

import uuid
from typing import Any, List, Optional

from llama_index.core.base.llms.types import ChatMessage, TextBlock
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.memory.memory import BaseMemoryBlock
from llama_index.core.prompts import BasePromptTemplate, RichPromptTemplate

from ..core import AsyncHmsBackend, HmsConfig, Record


def _text_of(message: ChatMessage) -> str:
    return "".join(b.text for b in message.blocks if isinstance(b, TextBlock))


class HolographicMemoryBlock(BaseMemoryBlock[str]):
    """Retrieves earlier messages relevant to the latest turns.

    `similarity_top_k` hits for the last `retrieval_context_window` messages are joined and
    passed through `format_template`. With a `session_id` (Memory supplies its own) only that
    session's messages are recalled. Metadata filters on messages beyond the session go in
    `filter`.
    """

    name: str = "HolographicMemory"
    similarity_top_k: int = Field(default=3, ge=1)
    retrieval_context_window: int = Field(default=5, ge=1)
    format_template: BasePromptTemplate = Field(default_factory=lambda: RichPromptTemplate("{{ text }}"))
    filter: Optional[dict] = Field(default=None, description="Extra metadata filter (equality terms are pushed to the service).")

    _backend: AsyncHmsBackend = PrivateAttr()

    def __init__(self, config: Optional[HmsConfig] = None, **fields: Any) -> None:
        hms_names = HmsConfig.__dataclass_fields__.keys()
        hms_kwargs = {k: fields.pop(k) for k in list(fields) if k in hms_names}
        if config is not None and hms_kwargs:
            raise TypeError("pass either config or HMS keyword arguments, not both")
        super().__init__(**fields)
        self._backend = AsyncHmsBackend(config or HmsConfig(**hms_kwargs))

    async def aclose(self) -> None:
        await self._backend.aclose()

    def _format(self, text: str) -> str:
        tpl = self.format_template
        return tpl.format(text=text)

    async def _aget(self, messages: Optional[List[ChatMessage]] = None, session_id: Optional[str] = None, **kwargs: Any) -> str:
        if not messages:
            return ""
        context = messages[-self.retrieval_context_window :]
        query = " ".join(t for t in (_text_of(m) for m in context) if t)
        if not query:
            return ""
        terms: dict = dict(self.filter or {})
        if session_id is not None:
            terms["session_id"] = session_id
        hits = await self._backend.recall(query, k=self.similarity_top_k, filter=terms or None)
        if not hits:
            return ""
        return self._format("\n\n".join(h.text for h in hits))

    async def _aput(self, messages: List[ChatMessage]) -> None:
        records = []
        for m in messages:
            text = _text_of(m)
            if not text:
                continue
            meta: dict = {"role": m.role.value}
            session = m.additional_kwargs.get("session_id")
            if session is not None:
                meta["session_id"] = session
            records.append(Record(id=uuid.uuid4().hex, text=f"{m.role.value}: {text}", metadata=meta))
        if records:
            await self._backend.remember(records)
