# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Chat stores over Holographic Memory.

`HolographicChatStore` implements `BaseChatStore` (used by `ChatMemoryBuffer`);
`HolographicAsyncChatStore` implements `AsyncDBChatStore` (used by `Memory.from_defaults(chat_store=...)`,
the current FIFO memory). Both share one layout:

- message record id `chat/<key>/<seq>`, metadata `chat_key`, `chat_seq`, `chat_page`
  (`seq // page_size`), `chat_status` and the JSON message;
- one header record per key holding `first_seq` and `next_seq`, which is also the key registry.

The service cannot list more than `max_fetch` records per query, so messages are read page by
page (`page_size` must not exceed `max_fetch`) and a conversation can be arbitrarily long.
Appends read and rewrite the header: one writer per key is assumed. A message must fit the
service's 64 KiB metadata limit. `delete_message(key, idx)` takes the message's sequence
number (its 0-based position in the key's history since creation), not its current index.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.storage.chat_store.base import BaseChatStore
from llama_index.core.storage.chat_store.base_db import AsyncDBChatStore, MessageStatus

from ..core import AsyncHmsBackend, HashingEmbedder, HmsBackend, HmsConfig, HmsError, Record

_ACTIVE = MessageStatus.ACTIVE.value


def _text(message: ChatMessage) -> str:
    return message.content if isinstance(message.content, str) and message.content else ""


class _Layout:
    def __init__(self, config: HmsConfig, page_size: int):
        if not 1 <= page_size <= config.max_fetch:
            raise ValueError(f"page_size must be 1..max_fetch ({config.max_fetch})")
        self.config = dataclasses.replace(config, embedder=HashingEmbedder(config.input_dim), text_queries=False)
        self.hash = HashingEmbedder(self.config.input_dim)
        self.page = page_size

    @staticmethod
    def header_id(key: str) -> str:
        return f"chatkey/{key}"

    def header(self, key: str, first: int, nxt: int) -> Record:
        return Record(
            self.header_id(key), key,
            {"chat_header": True, "chat_key": key, "first_seq": first, "next_seq": nxt},
            self.hash.embed_query(key),
        )

    def message(self, key: str, seq: int, status: str, message: ChatMessage) -> Record:
        text = _text(message)
        return Record(
            f"chat/{key}/{seq:012d}", text,
            {
                "chat_key": key, "chat_seq": seq, "chat_page": seq // self.page,
                "chat_status": status, "chat_message": message.model_dump(mode="json"),
            },
            self.hash.embed_query(text[:256] or key),
        )

    def pages(self, first: int, nxt: int) -> range:
        return range(first // self.page, (nxt - 1) // self.page + 1) if nxt > first else range(0)

    @staticmethod
    def parse(rec: Record) -> Tuple[int, str, ChatMessage]:
        m = rec.metadata
        return m["chat_seq"], m["chat_status"], ChatMessage.model_validate(m["chat_message"])

    @staticmethod
    def window(items: List[Any], limit: Optional[int], offset: Optional[int]) -> List[Any]:
        start = offset or 0
        return items[start : None if limit is None else start + limit]


class _SyncLog(_Layout):
    def __init__(self, config: HmsConfig, page_size: int):
        super().__init__(config, page_size)
        self.backend = HmsBackend(self.config)

    def head(self, key: str) -> Optional[Tuple[int, int]]:
        rec = self.backend.get([self.header_id(key)])[0]
        return None if rec is None else (rec.metadata["first_seq"], rec.metadata["next_seq"])

    def scan(self, key: str) -> List[Tuple[int, str, ChatMessage]]:
        h = self.head(key)
        if h is None:
            return []
        out = []
        for p in self.pages(*h):
            for rec in self.backend.list({"chat_key": key, "chat_page": p}).records:
                out.append(self.parse(rec))
        return sorted(out, key=lambda t: t[0])

    def add(self, key: str, messages: List[ChatMessage], status: str) -> None:
        if not messages:
            return
        first, nxt = self.head(key) or (0, 0)
        recs = [self.message(key, nxt + i, status, m) for i, m in enumerate(messages)]
        self.backend.remember(recs + [self.header(key, first, nxt + len(messages))])

    def drop(self, key: str, seqs: List[int]) -> None:
        self.backend.forget([f"chat/{key}/{s:012d}" for s in seqs])

    def clear(self, key: str, status: Optional[str] = None) -> List[ChatMessage]:
        rows = self.scan(key)
        doomed = [r for r in rows if status is None or r[1] == status]
        self.drop(key, [r[0] for r in doomed])
        if len(doomed) == len(rows):
            self.backend.forget([self.header_id(key)])
        return [r[2] for r in doomed]

    def keys(self) -> List[str]:
        page = self.backend.list({"chat_header": True})
        if page.truncated:
            raise HmsError(f"more than max_fetch={self.config.max_fetch} chat keys; cannot enumerate")
        return sorted(r.metadata["chat_key"] for r in page.records)


class _AsyncLog(_Layout):
    def __init__(self, config: HmsConfig, page_size: int):
        super().__init__(config, page_size)
        self.backend = AsyncHmsBackend(self.config)

    async def head(self, key: str) -> Optional[Tuple[int, int]]:
        rec = (await self.backend.get([self.header_id(key)]))[0]
        return None if rec is None else (rec.metadata["first_seq"], rec.metadata["next_seq"])

    async def scan(self, key: str) -> List[Tuple[int, str, ChatMessage]]:
        h = await self.head(key)
        if h is None:
            return []
        out = []
        for p in self.pages(*h):
            for rec in (await self.backend.list({"chat_key": key, "chat_page": p})).records:
                out.append(self.parse(rec))
        return sorted(out, key=lambda t: t[0])

    async def add(self, key: str, messages: List[ChatMessage], status: str) -> None:
        if not messages:
            return
        first, nxt = await self.head(key) or (0, 0)
        recs = [self.message(key, nxt + i, status, m) for i, m in enumerate(messages)]
        await self.backend.remember(recs + [self.header(key, first, nxt + len(messages))])

    async def drop(self, key: str, seqs: List[int]) -> None:
        await self.backend.forget([f"chat/{key}/{s:012d}" for s in seqs])

    async def clear(self, key: str, status: Optional[str] = None) -> List[ChatMessage]:
        rows = await self.scan(key)
        doomed = [r for r in rows if status is None or r[1] == status]
        await self.drop(key, [r[0] for r in doomed])
        if len(doomed) == len(rows):
            await self.backend.forget([self.header_id(key)])
        return [r[2] for r in doomed]

    async def keys(self) -> List[str]:
        page = await self.backend.list({"chat_header": True})
        if page.truncated:
            raise HmsError(f"more than max_fetch={self.config.max_fetch} chat keys; cannot enumerate")
        return sorted(r.metadata["chat_key"] for r in page.records)


class HolographicChatStore(BaseChatStore):
    """`BaseChatStore` (sync; the async methods run it in a worker thread, as the base class does)."""

    page_size: int = 50
    _log: _SyncLog = PrivateAttr()

    def __init__(self, config: Optional[HmsConfig] = None, *, page_size: int = 50, **hms_kwargs: Any) -> None:
        super().__init__(page_size=page_size)
        if config is not None and hms_kwargs:
            raise TypeError("pass either config or keyword arguments, not both")
        self._log = _SyncLog(config or HmsConfig(**hms_kwargs), page_size)

    @classmethod
    def class_name(cls) -> str:
        return "HolographicChatStore"

    def close(self) -> None:
        self._log.backend.close()

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        self._log.clear(key)
        self._log.add(key, list(messages), _ACTIVE)

    def get_messages(self, key: str) -> List[ChatMessage]:
        return [m for _, _, m in self._log.scan(key)]

    def add_message(self, key: str, message: ChatMessage) -> None:
        self._log.add(key, [message], _ACTIVE)

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        if self._log.head(key) is None:
            return None
        return self._log.clear(key)

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        found = [r for r in self._log.scan(key) if r[0] == idx]
        if not found:
            return None
        self._log.drop(key, [idx])
        return found[0][2]

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        rows = self._log.scan(key)
        if not rows:
            return None
        self._log.drop(key, [rows[-1][0]])
        return rows[-1][2]

    def get_keys(self) -> List[str]:
        return self._log.keys()


class HolographicAsyncChatStore(AsyncDBChatStore):
    """`AsyncDBChatStore` for `Memory.from_defaults(chat_store=...)`."""

    page_size: int = 50
    _log: _AsyncLog = PrivateAttr()

    def __init__(self, config: Optional[HmsConfig] = None, *, page_size: int = 50, **hms_kwargs: Any) -> None:
        super().__init__(page_size=page_size)
        if config is not None and hms_kwargs:
            raise TypeError("pass either config or keyword arguments, not both")
        self._log = _AsyncLog(config or HmsConfig(**hms_kwargs), page_size)

    @classmethod
    def class_name(cls) -> str:
        return "HolographicAsyncChatStore"

    async def aclose(self) -> None:
        await self._log.backend.aclose()

    async def get_messages(
        self, key: str, status: Optional[MessageStatus] = MessageStatus.ACTIVE,
        limit: Optional[int] = None, offset: Optional[int] = None,
    ) -> List[ChatMessage]:
        rows = [r for r in await self._log.scan(key) if status is None or r[1] == status.value]
        return [m for _, _, m in _Layout.window(rows, limit, offset)]

    async def count_messages(self, key: str, status: Optional[MessageStatus] = MessageStatus.ACTIVE) -> int:
        return len([r for r in await self._log.scan(key) if status is None or r[1] == status.value])

    async def add_message(self, key: str, message: ChatMessage, status: MessageStatus = MessageStatus.ACTIVE) -> None:
        await self._log.add(key, [message], status.value)

    async def add_messages(self, key: str, messages: List[ChatMessage], status: MessageStatus = MessageStatus.ACTIVE) -> None:
        await self._log.add(key, list(messages), status.value)

    async def set_messages(self, key: str, messages: List[ChatMessage], status: MessageStatus = MessageStatus.ACTIVE) -> None:
        await self._log.clear(key)
        await self._log.add(key, list(messages), status.value)

    async def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        found = [r for r in await self._log.scan(key) if r[0] == idx]
        if not found:
            return None
        await self._log.drop(key, [idx])
        return found[0][2]

    async def delete_messages(self, key: str, status: Optional[MessageStatus] = None) -> None:
        await self._log.clear(key, None if status is None else status.value)

    async def delete_oldest_messages(self, key: str, n: int) -> List[ChatMessage]:
        rows = [r for r in await self._log.scan(key) if r[1] == _ACTIVE][:n]
        await self._log.drop(key, [r[0] for r in rows])
        return [r[2] for r in rows]

    async def archive_oldest_messages(self, key: str, n: int) -> List[ChatMessage]:
        rows = [r for r in await self._log.scan(key) if r[1] == _ACTIVE][:n]
        archived = [self._log.message(key, seq, MessageStatus.ARCHIVED.value, m) for seq, _, m in rows]
        if archived:
            await self._log.backend.remember(archived)
        return [r[2] for r in rows]

    async def get_keys(self) -> List[str]:
        return await self._log.keys()
