# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""LangChain chat message history over WARM.

The service cannot list or page records, so a session is stored as numbered messages grouped
into pages of `page_size` (metadata `page = seq // page_size`). Reading fetches page 0, 1, ...
until a page comes back short, so a session of any length reads back in order with
`ceil(n / page_size) + 1` queries. One writer per session: the next sequence number is cached
per instance and found by reading the pages when the instance first writes.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

from ..core import AsyncWarmBackend, WarmBackend, WarmConfig, Record
from ..core.filters import Eq, And


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return " ".join(p if isinstance(p, str) else str(p.get("text", "")) for p in content)


class WarmChatMessageHistory(BaseChatMessageHistory):
    """Persistent message history for one `session_id`.

    Each message's searchable text is its content; the full serialized message is kept in
    metadata, so a message must fit the service's 64 KiB metadata limit (the content is
    stored twice). Pass `config` (an `WarmConfig`) or keyword options. The default collection
    is `chat-history`; sessions are separated by `session_id`.
    """

    def __init__(
        self,
        session_id: str,
        *,
        config: Optional[WarmConfig] = None,
        page_size: int = 50,
        **warm_kwargs: Any,
    ) -> None:
        if not session_id:
            raise ValueError("session_id must be non-empty")
        if config is not None and warm_kwargs:
            raise ValueError("pass either config or keyword options, not both")
        warm_kwargs.setdefault("collection", "chat-history")
        self.config = config or WarmConfig(**warm_kwargs)
        if not 1 <= page_size <= self.config.max_fetch:
            raise ValueError(f"page_size must be 1..{self.config.max_fetch} (max_fetch)")
        self.session_id = session_id
        self.page_size = page_size
        self._sync: Optional[WarmBackend] = None
        self._async: Optional[AsyncWarmBackend] = None
        self._next: Optional[int] = None

    @property
    def backend(self) -> WarmBackend:
        if self._sync is None:
            self._sync = WarmBackend(self.config)
        return self._sync

    @property
    def abackend(self) -> AsyncWarmBackend:
        if self._async is None:
            self._async = AsyncWarmBackend(self.config)
        return self._async

    def close(self) -> None:
        if self._sync is not None:
            self._sync.close()
            self._sync = None

    async def aclose(self) -> None:
        if self._async is not None:
            await self._async.aclose()
            self._async = None

    # ---- encoding ---------------------------------------------------------------------

    def _record(self, message: BaseMessage, seq: int) -> Record:
        return Record(
            id=f"{self.session_id}:{seq:012d}",
            text=_text(message),
            metadata={
                "session_id": self.session_id,
                "seq": seq,
                "page": seq // self.page_size,
                "message": json.dumps(message_to_dict(message)),
            },
        )

    def _page_filter(self, page: int) -> And:
        return And((Eq("session_id", self.session_id), Eq("page", page)))

    @staticmethod
    def _decode(records) -> List[tuple]:
        return [(r.metadata["seq"], r.metadata["message"]) for r in records]

    # ---- sync -------------------------------------------------------------------------

    def _read(self) -> List[tuple]:
        out: List[tuple] = []
        page = 0
        while True:
            recs = self.backend.list(self._page_filter(page)).records
            out.extend(self._decode(recs))
            if len(recs) < self.page_size:
                return sorted(out)
            page += 1

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore[override]
        rows = self._read()
        self._next = rows[-1][0] + 1 if rows else 0
        return messages_from_dict([json.loads(m) for _, m in rows])

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return
        if self._next is None:
            rows = self._read()
            self._next = rows[-1][0] + 1 if rows else 0
        records = [self._record(m, self._next + i) for i, m in enumerate(messages)]
        self.backend.remember(records)
        self._next += len(records)

    def clear(self) -> None:
        self.backend.forget_where(Eq("session_id", self.session_id))
        self._next = 0

    # ---- async ------------------------------------------------------------------------

    async def _aread(self) -> List[tuple]:
        out: List[tuple] = []
        page = 0
        while True:
            recs = (await self.abackend.list(self._page_filter(page))).records
            out.extend(self._decode(recs))
            if len(recs) < self.page_size:
                return sorted(out)
            page += 1

    async def aget_messages(self) -> List[BaseMessage]:
        rows = await self._aread()
        self._next = rows[-1][0] + 1 if rows else 0
        return messages_from_dict([json.loads(m) for _, m in rows])

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return
        if self._next is None:
            rows = await self._aread()
            self._next = rows[-1][0] + 1 if rows else 0
        records = [self._record(m, self._next + i) for i, m in enumerate(messages)]
        await self.abackend.remember(records)
        self._next += len(records)

    async def aclear(self) -> None:
        await self.abackend.forget_where(Eq("session_id", self.session_id))
        self._next = 0

HolographicChatMessageHistory = WarmChatMessageHistory
