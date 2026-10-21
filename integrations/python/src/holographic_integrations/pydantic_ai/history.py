# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Persistent Pydantic AI message history.

Each `ModelMessage` is stored as one JSON item (through `ModelMessagesTypeAdapter`, so every
part type round-trips) in a `core.paged` log keyed by conversation id.

    store = MessageStore(HmsConfig(url=...))
    result = agent.run_sync("hi", message_history=store.load("chat-1"))
    store.save("chat-1", result.all_messages())
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from ..core import AsyncHmsBackend, HmsBackend, HmsConfig
from ..core.paged import AsyncPagedLog, PagedLog

_NAMESPACE = "pai-history"


def _dump(message: ModelMessage) -> str:
    return ModelMessagesTypeAdapter.dump_json([message]).decode("utf-8")


def _load(text: str) -> ModelMessage:
    return ModelMessagesTypeAdapter.validate_json(text)[0]


class MessageStore:
    """Synchronous store of conversations (`ModelMessage` lists).

    `save` assumes `messages` extends what is stored (what `result.all_messages()` returns
    for a run that started from `load`): it appends only the new tail. When `messages` is
    shorter than the stored history (a history processor compacted it) the conversation is
    rewritten. Use `replace` to force a rewrite when a processor changed messages in place.
    """

    def __init__(self, config: Optional[HmsConfig] = None, *, backend: Optional[HmsBackend] = None, **log_kwargs):
        self._owns = backend is None
        self.backend = backend or HmsBackend(config)
        self._log_kwargs = log_kwargs

    def _log(self, conversation_id: str) -> PagedLog:
        return PagedLog(self.backend, conversation_id, namespace=_NAMESPACE, **self._log_kwargs)

    def close(self) -> None:
        if self._owns:
            self.backend.close()

    def load(self, conversation_id: str, limit: Optional[int] = None) -> List[ModelMessage]:
        """The stored messages, or the last `limit` of them. Empty for an unknown id."""
        return [_load(t) for t in self._log(conversation_id).read(limit)]

    def append(self, conversation_id: str, messages: Sequence[ModelMessage]) -> None:
        self._log(conversation_id).append([_dump(m) for m in messages])

    def save(self, conversation_id: str, messages: Sequence[ModelMessage]) -> None:
        log = self._log(conversation_id)
        stored = len(log)
        if len(messages) < stored:
            log.clear()
            stored = 0
        log.append([_dump(m) for m in messages[stored:]])

    def replace(self, conversation_id: str, messages: Sequence[ModelMessage]) -> None:
        log = self._log(conversation_id)
        log.clear()
        log.append([_dump(m) for m in messages])

    def delete(self, conversation_id: str) -> None:
        self._log(conversation_id).clear()


class AsyncMessageStore:
    """Asynchronous counterpart of `MessageStore` with the same semantics."""

    def __init__(self, config: Optional[HmsConfig] = None, *, backend: Optional[AsyncHmsBackend] = None, **log_kwargs):
        self._owns = backend is None
        self.backend = backend or AsyncHmsBackend(config)
        self._log_kwargs = log_kwargs
        self._logs: dict[str, AsyncPagedLog] = {}

    def _log(self, conversation_id: str) -> AsyncPagedLog:
        # one object per id so its write lock serializes this process's writers
        log = self._logs.get(conversation_id)
        if log is None:
            log = self._logs[conversation_id] = AsyncPagedLog(
                self.backend, conversation_id, namespace=_NAMESPACE, **self._log_kwargs
            )
        return log

    async def aclose(self) -> None:
        if self._owns:
            await self.backend.aclose()

    async def load(self, conversation_id: str, limit: Optional[int] = None) -> List[ModelMessage]:
        return [_load(t) for t in await self._log(conversation_id).read(limit)]

    async def append(self, conversation_id: str, messages: Sequence[ModelMessage]) -> None:
        await self._log(conversation_id).append([_dump(m) for m in messages])

    async def save(self, conversation_id: str, messages: Sequence[ModelMessage]) -> None:
        log = self._log(conversation_id)
        stored = await log.length()
        if len(messages) < stored:
            await log.clear()
            stored = 0
        await log.append([_dump(m) for m in messages[stored:]])

    async def replace(self, conversation_id: str, messages: Sequence[ModelMessage]) -> None:
        log = self._log(conversation_id)
        await log.clear()
        await log.append([_dump(m) for m in messages])

    async def delete(self, conversation_id: str) -> None:
        await self._log(conversation_id).clear()
