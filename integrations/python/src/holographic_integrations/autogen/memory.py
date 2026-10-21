# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""`autogen_core.memory.Memory` implementation for AutoGen (autogen-agentchat 0.4+)."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, Mapping, Optional

from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import BaseModel
from typing_extensions import Self

from holographic_integrations.core import AsyncHmsBackend, HmsConfig, Record
from holographic_integrations.core.models import Hit

_M = "ag_m_"
_MIME = "ag_mime"


def _prefix(spec: Any) -> Any:
    """Prefix the metadata keys of a MongoDB-style filter with the storage namespace."""
    if isinstance(spec, Mapping):
        out: Dict[str, Any] = {}
        for key, value in spec.items():
            if key in ("$and", "$or"):
                out[key] = [_prefix(v) for v in value]
            elif key == "$not":
                out[key] = _prefix(value)
            elif key.startswith("$"):
                out[key] = value
            else:
                out[_M + key] = value
        return out
    return spec


class HolographicMemoryConfig(BaseModel):
    """Serializable configuration. The embedder and API key are not part of it: the key is read
    from `HMS_API_KEY`, and a custom embedder is supplied again when the component is loaded."""

    name: str = "holographic_memory"
    url: str = "http://127.0.0.1:8080"
    tenant_id: Optional[str] = None
    input_dim: int = 384
    text_queries: bool = False
    k: int = 5
    score_threshold: Optional[float] = None
    overfetch: int = 5
    max_fetch: int = 100


class HolographicMemory(Memory, Component[HolographicMemoryConfig]):
    """AutoGen `Memory` backed by Holographic Memory.

    Pass it to `AssistantAgent(memory=[...])`: before each model call `update_context` queries
    with the latest message and adds the hits to the model context as a system message, in the
    same format as `ListMemory`. `name` is the collection, so one tenant can hold several
    agents' memories side by side.

    Content types: text, Markdown and JSON (`dict`) are stored; bytes and images raise
    `ValueError`. `query(..., k=, score_threshold=, filter=)` accepts a MongoDB-style filter
    over the `MemoryContent.metadata` you added (equality is evaluated by the service,
    other operators here after an over-fetch; see docs). Results carry `metadata["id"]` and
    `metadata["score"]` (cosine in vector mode).
    """

    component_type = "memory"
    component_provider_override = "holographic_integrations.autogen.HolographicMemory"
    component_config_schema = HolographicMemoryConfig

    def __init__(
        self,
        name: str = "holographic_memory",
        *,
        hms: Optional[HmsConfig] = None,
        k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> None:
        if hms is None:
            hms = HmsConfig(api_key=os.environ.get("HMS_API_KEY"))
        if hms.collection is None:
            hms.collection = name
        self._name = name
        self._k = k
        self._threshold = score_threshold
        self._hms = hms
        self._backend = AsyncHmsBackend(hms)

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def _to_record(content: MemoryContent) -> Record:
        mime = content.mime_type.value if isinstance(content.mime_type, MemoryMimeType) else str(content.mime_type)
        meta = dict(content.metadata or {})
        rid = str(meta.pop("id", None) or uuid.uuid4())
        if isinstance(content.content, str):
            text = content.content
        elif isinstance(content.content, dict):
            text = json.dumps(content.content)
        else:
            raise ValueError("HolographicMemory stores text, Markdown and JSON content only")
        stored = {_M + k: v for k, v in meta.items()}
        stored[_MIME] = mime
        return Record(rid, text, stored)

    @staticmethod
    def _to_content(hit: Hit) -> MemoryContent:
        mime: Any = hit.metadata.get(_MIME, "text/plain")
        try:
            mime = MemoryMimeType(mime)
        except ValueError:
            pass
        user = {k[len(_M):]: v for k, v in hit.metadata.items() if k.startswith(_M)}
        body: Any = hit.text
        if mime == MemoryMimeType.JSON:
            body = json.loads(hit.text)
        return MemoryContent(content=body, mime_type=mime, metadata={**user, "id": hit.id, "score": hit.score})

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        await self._backend.remember([self._to_record(content)])

    async def query(
        self,
        query: str | MemoryContent = "",
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        text = query if isinstance(query, str) else (
            query.content if isinstance(query.content, str) else json.dumps(query.content)
        )
        if not text.strip():
            return MemoryQueryResult(results=[])
        threshold = kwargs.get("score_threshold", self._threshold)
        flt = _prefix(kwargs["filter"]) if kwargs.get("filter") else None
        hits = await self._backend.recall(text, k=kwargs.get("k", self._k), filter=flt)
        return MemoryQueryResult(
            results=[self._to_content(h) for h in hits if threshold is None or h.score >= threshold]
        )

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))
        last = messages[-1].content
        result = await self.query(last if isinstance(last, str) else str(last))
        if result.results:
            lines = [f"{i}. {m.content}" for i, m in enumerate(result.results, 1)]
            await model_context.add_message(
                SystemMessage(content="\nRelevant memory content (in chronological order):\n" + "\n".join(lines) + "\n")
            )
        return UpdateContextResult(memories=result)

    async def clear(self) -> None:
        await self._backend.forget_where(None)

    async def close(self) -> None:
        await self._backend.aclose()

    def _to_config(self) -> HolographicMemoryConfig:
        c = self._hms
        return HolographicMemoryConfig(
            name=self._name, url=c.url, tenant_id=c.tenant_id, input_dim=c.input_dim or 384,
            text_queries=c.text_queries, k=self._k, score_threshold=self._threshold,
            overfetch=c.overfetch, max_fetch=c.max_fetch,
        )

    @classmethod
    def _from_config(cls, config: HolographicMemoryConfig) -> Self:
        hms = HmsConfig(
            url=config.url, api_key=os.environ.get("HMS_API_KEY"), tenant_id=config.tenant_id,
            input_dim=config.input_dim, text_queries=config.text_queries,
            overfetch=config.overfetch, max_fetch=config.max_fetch,
        )
        return cls(config.name, hms=hms, k=config.k, score_threshold=config.score_threshold)
