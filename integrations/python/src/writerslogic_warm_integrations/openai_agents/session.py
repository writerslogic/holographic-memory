# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Agents SDK `Session` backed by WARM.

Items are stored as JSON text in a `core.paged.AsyncPagedLog` (see that module for the
layout and its limits): any number of items per session, an item of up to
`chunk_chars * max_parts` characters, O(log pages) queries to find the tail, and no support
for concurrent writers to one session id from several processes.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from agents.items import TResponseInputItem
from agents.memory import SessionABC, SessionSettings
from agents.memory.session_settings import coerce_session_settings, resolve_session_limit

from ..core import AsyncWarmBackend, WarmConfig
from ..core.paged import AsyncPagedLog


class WarmSession(SessionABC):
    """`agents.memory.Session` implementation.

    Args:
        session_id: conversation identifier.
        config: connection and behavior knobs (`WarmConfig`); ignored when `backend` is given.
        backend: an existing `AsyncWarmBackend` to share between sessions.
        session_settings: SDK session settings (`limit` is the default for `get_items`).
        page_size: items per page; at most the backend's `max_fetch`.
        chunk_chars: characters of item JSON per stored record (at most 8000, which is
            always inside the service's 4096-word and 64 KiB record limits).
        max_parts: maximum records one item may span.
    """

    def __init__(
        self,
        session_id: str,
        config: Optional[WarmConfig] = None,
        *,
        backend: Optional[AsyncWarmBackend] = None,
        session_settings: Optional[SessionSettings | dict[str, Any]] = None,
        page_size: Optional[int] = None,
        chunk_chars: int = 8000,
        max_parts: Optional[int] = None,
    ):
        self.session_id = session_id
        self.session_settings = (
            coerce_session_settings(session_settings) if session_settings is not None else SessionSettings()
        )
        self._owns_backend = backend is None
        self.backend = backend or AsyncWarmBackend(config)
        self._log = AsyncPagedLog(
            self.backend, session_id, namespace="oa-session",
            page_size=page_size, chunk_chars=chunk_chars, max_parts=max_parts,
        )

    async def aclose(self) -> None:
        if self._owns_backend:
            await self.backend.aclose()

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        limit = resolve_session_limit(limit, self.session_settings)
        return [json.loads(t) for t in await self._log.read(limit)]

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        await self._log.append([json.dumps(i, ensure_ascii=False) for i in items])

    async def pop_item(self) -> TResponseInputItem | None:
        text = await self._log.pop()
        return None if text is None else json.loads(text)

    async def clear_session(self) -> None:
        await self._log.clear()

HolographicSession = WarmSession
