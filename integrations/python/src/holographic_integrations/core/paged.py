# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Ordered, unbounded append logs of text items on top of the memory backend.

`PagedLog` stores an ordered sequence of strings (chat items, serialized messages) by
partitioning it into pages of `page_size` consecutive sequence numbers; a page is read with
one equality-filtered query. The last page is located by exponential then binary search,
so a log holds any number of items and costs O(log pages) queries to find its tail.

- One record per item, id `<log>/<seq:012d>`, metadata `hms_log`, `hms_seq`, `hms_page`,
  `hms_parts`. The service accepts at most 4096 words and 64 KiB per record, so text longer
  than `chunk_chars` (at most 8000) continues in extra records `<log>/<seq:012d>.<n>`
  (metadata `hms_log`, `hms_blob_of`, `hms_part`); an item spans at most `max_parts` records.
- Pages stay contiguous, so concurrent writers to one log from several processes are not
  supported. Writers sharing one `PagedLog` object are serialized.
"""

from __future__ import annotations

import asyncio
import threading
from typing import List, Optional, Sequence

from .backend import AsyncHmsBackend, HmsBackend
from .models import HmsError, Record

_LOG = "hms_log"
_SEQ = "hms_seq"
_PAGE = "hms_page"
_PARTS = "hms_parts"
_BLOB_OF = "hms_blob_of"
_PART = "hms_part"
_LEADING = "hms_text_leading"
_TRAILING = "hms_text_trailing"
_BLANK = "hms_text_blank"


class _Layout:
    def __init__(self, max_fetch: int, namespace: str, key: str, page_size: Optional[int], chunk_chars: int, max_parts: Optional[int]):
        page_size = min(50, max_fetch) if page_size is None else page_size
        max_parts = min(100, max_fetch) if max_parts is None else max_parts
        if not 1 <= page_size <= max_fetch:
            raise ValueError(f"page_size must be 1..{max_fetch}")
        if not 1 <= max_parts <= max_fetch:
            raise ValueError(f"max_parts must be 1..{max_fetch}")
        if not 1 <= chunk_chars <= 8000:
            raise ValueError("chunk_chars must be 1..8000")
        self.name = f"{namespace}:{key}"
        self.page_size, self.chunk_chars, self.max_parts = page_size, chunk_chars, max_parts

    def page_filter(self, page: int) -> dict:
        return {_LOG: self.name, _PAGE: page}

    def blob_filter(self, seq: int) -> dict:
        return {_LOG: self.name, _BLOB_OF: seq}

    def item_id(self, seq: int) -> str:
        return f"{self.name}/{seq:012d}"

    def part_ids(self, seq: int, parts: int) -> List[str]:
        return [f"{self.item_id(seq)}.{n}" for n in range(1, parts)]

    @staticmethod
    def record(ident: str, text: str, metadata: dict) -> Record:
        trimmed = text.strip()
        if not trimmed:
            return Record(ident, "_", {**metadata, _BLANK: text})
        return Record(ident, trimmed, {
            **metadata,
            _LEADING: text[:len(text) - len(text.lstrip())],
            _TRAILING: text[len(text.rstrip()):],
        })

    @staticmethod
    def piece(record: Record) -> str:
        if _BLANK in record.metadata:
            return record.metadata[_BLANK]
        return record.metadata.get(_LEADING, "") + record.text + record.metadata.get(_TRAILING, "")

    def records(self, first_seq: int, texts: Sequence[str]) -> List[Record]:
        blobs: List[Record] = []
        mains: List[Record] = []
        seq = first_seq
        for text in texts:
            chunks = [text[i : i + self.chunk_chars] for i in range(0, len(text), self.chunk_chars)] or [""]
            if len(chunks) > self.max_parts:
                raise ValueError(
                    f"item needs {len(chunks)} parts; limit is max_parts={self.max_parts} x chunk_chars={self.chunk_chars}"
                )
            for n, chunk in enumerate(chunks[1:], start=1):
                blobs.append(self.record(f"{self.item_id(seq)}.{n}", chunk, {_LOG: self.name, _BLOB_OF: seq, _PART: n}))
            mains.append(
                self.record(self.item_id(seq), chunks[0],
                       {_LOG: self.name, _SEQ: seq, _PAGE: seq // self.page_size, _PARTS: len(chunks)})
            )
            seq += 1
        return blobs + mains  # parts first: a reader that sees an item's head finds its parts

    @staticmethod
    def join(head: Record, tail: Sequence[Record]) -> str:
        parts = head.metadata.get(_PARTS, 1)
        if len(tail) != parts - 1:
            raise HmsError(f"item {head.id} is missing parts ({len(tail) + 1} of {parts})")
        return _Layout.piece(head) + "".join(_Layout.piece(r) for r in sorted(tail, key=lambda r: r.metadata[_PART]))


def _by_seq(records: Sequence[Record]) -> List[Record]:
    return sorted(records, key=lambda r: r.metadata[_SEQ])


class PagedLog:
    """Synchronous log over `HmsBackend`."""

    def __init__(self, backend: HmsBackend, key: str, *, namespace: str = "log", page_size: Optional[int] = None,
                 chunk_chars: int = 8000, max_parts: Optional[int] = None):
        self.backend = backend
        self._l = _Layout(backend.config.max_fetch, namespace, key, page_size, chunk_chars, max_parts)
        self._lock = threading.Lock()

    def _page(self, page: int) -> List[Record]:
        return _by_seq(self.backend.list(self._l.page_filter(page)).records)

    def _last_page(self) -> int:
        if not self._page(0):
            return -1
        lo, hi = 0, 1
        while self._page(hi):
            lo, hi = hi, hi * 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self._page(mid):
                lo = mid
            else:
                hi = mid
        return lo

    def _text(self, head: Record) -> str:
        tail = self.backend.list(self._l.blob_filter(head.metadata[_SEQ])).records if head.metadata.get(_PARTS, 1) > 1 else []
        return self._l.join(head, tail)

    def read(self, limit: Optional[int] = None) -> List[str]:
        """All items in order, or the last `limit` of them."""
        if limit is not None and limit <= 0:
            return []
        records: List[Record] = []
        page = self._last_page()
        while page >= 0 and (limit is None or len(records) < limit):
            records = self._page(page) + records
            page -= 1
        if limit is not None:
            records = records[-limit:]
        return [self._text(r) for r in records]

    def __len__(self) -> int:
        last = self._last_page()
        if last < 0:
            return 0
        return self._page(last)[-1].metadata[_SEQ] + 1

    def append(self, texts: Sequence[str]) -> None:
        if not texts:
            return
        with self._lock:
            self.backend.remember(self._l.records(len(self), texts))

    def pop(self) -> Optional[str]:
        with self._lock:
            last = self._last_page()
            if last < 0:
                return None
            head = self._page(last)[-1]
            text = self._text(head)
            seq = head.metadata[_SEQ]
            self.backend.forget([head.id, *self._l.part_ids(seq, head.metadata.get(_PARTS, 1))])
            return text

    def clear(self) -> None:
        with self._lock:
            self.backend.forget_where({_LOG: self._l.name})


class AsyncPagedLog:
    """Asynchronous log over `AsyncHmsBackend`, same layout and semantics as `PagedLog`."""

    def __init__(self, backend: AsyncHmsBackend, key: str, *, namespace: str = "log", page_size: Optional[int] = None,
                 chunk_chars: int = 8000, max_parts: Optional[int] = None):
        self.backend = backend
        self._l = _Layout(backend.config.max_fetch, namespace, key, page_size, chunk_chars, max_parts)
        self._lock = asyncio.Lock()

    async def _page(self, page: int) -> List[Record]:
        return _by_seq((await self.backend.list(self._l.page_filter(page))).records)

    async def _last_page(self) -> int:
        if not await self._page(0):
            return -1
        lo, hi = 0, 1
        while await self._page(hi):
            lo, hi = hi, hi * 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if await self._page(mid):
                lo = mid
            else:
                hi = mid
        return lo

    async def _text(self, head: Record) -> str:
        tail = []
        if head.metadata.get(_PARTS, 1) > 1:
            tail = (await self.backend.list(self._l.blob_filter(head.metadata[_SEQ]))).records
        return self._l.join(head, tail)

    async def read(self, limit: Optional[int] = None) -> List[str]:
        if limit is not None and limit <= 0:
            return []
        records: List[Record] = []
        page = await self._last_page()
        while page >= 0 and (limit is None or len(records) < limit):
            records = await self._page(page) + records
            page -= 1
        if limit is not None:
            records = records[-limit:]
        return [await self._text(r) for r in records]

    async def length(self) -> int:
        last = await self._last_page()
        if last < 0:
            return 0
        return (await self._page(last))[-1].metadata[_SEQ] + 1

    async def append(self, texts: Sequence[str]) -> None:
        if not texts:
            return
        async with self._lock:
            await self.backend.remember(self._l.records(await self.length(), texts))

    async def pop(self) -> Optional[str]:
        async with self._lock:
            last = await self._last_page()
            if last < 0:
                return None
            head = (await self._page(last))[-1]
            text = await self._text(head)
            seq = head.metadata[_SEQ]
            await self.backend.forget([head.id, *self._l.part_ids(seq, head.metadata.get(_PARTS, 1))])
            return text

    async def clear(self) -> None:
        async with self._lock:
            await self.backend.forget_where({_LOG: self._l.name})
