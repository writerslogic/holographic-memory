# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Anthropic SDK memory tools backed by WARM.

`WarmMemoryTool` (async) subclasses `BetaAsyncAbstractMemoryTool` and
`WarmMemoryToolSync` subclasses `BetaAbstractMemoryTool`, so either plugs into
`client.beta.messages.tool_runner(tools=[...])`.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Coroutine, Optional, TypeVar

from anthropic.lib.tools import (
    BetaAbstractMemoryTool,
    BetaAsyncAbstractMemoryTool,
    BetaAsyncFunctionTool,
    BetaFunctionTool,
    ToolError,
)
from anthropic.types.beta import (
    BetaMemoryTool20250818CreateCommand,
    BetaMemoryTool20250818DeleteCommand,
    BetaMemoryTool20250818InsertCommand,
    BetaMemoryTool20250818RenameCommand,
    BetaMemoryTool20250818StrReplaceCommand,
    BetaMemoryTool20250818ViewCommand,
)

from .core import MemoryFs, MemoryToolFailure
from .store import FileStore, McpFileStore

T = TypeVar("T")

_RECALL_DESCRIPTION = (
    "Search your memory files for the ones most relevant to a query. Returns file paths with a "
    "relevance score and a text excerpt. Use `memory` view to read a file in full."
)


async def _failing(fs_call: Coroutine[Any, Any, str]) -> str:
    try:
        return await fs_call
    except MemoryToolFailure as error:
        raise ToolError(str(error)) from error


class WarmMemoryTool(BetaAsyncAbstractMemoryTool):
    """Async memory tool over a `FileStore` (by default an warm-mcp server)."""

    def __init__(self, store: FileStore, *, max_view_chars: int = 16_000, edit_retries: int = 25, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fs = MemoryFs(store, max_view_chars=max_view_chars, edit_retries=edit_retries)

    @classmethod
    async def connect(cls, *, max_view_chars: int = 16_000, edit_retries: int = 25, **connection: Any) -> "WarmMemoryTool":
        """Connect to warm-mcp: ``command=``/``args=``/``env=`` for stdio, or ``url=``/``token=``."""
        store = await McpFileStore.connect(**connection)
        return cls(store, max_view_chars=max_view_chars, edit_retries=edit_retries)

    async def aclose(self) -> None:
        await self.fs.store.close()

    async def __aenter__(self) -> "WarmMemoryTool":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def view(self, command: BetaMemoryTool20250818ViewCommand) -> str:  # type: ignore[override]
        return await _failing(self.fs.view(command.path, getattr(command, "view_range", None) or None))

    async def create(self, command: BetaMemoryTool20250818CreateCommand) -> str:  # type: ignore[override]
        return await _failing(self.fs.create(command.path, command.file_text))

    async def str_replace(self, command: BetaMemoryTool20250818StrReplaceCommand) -> str:  # type: ignore[override]
        return await _failing(self.fs.str_replace(command.path, command.old_str, getattr(command, "new_str", None)))

    async def insert(self, command: BetaMemoryTool20250818InsertCommand) -> str:  # type: ignore[override]
        return await _failing(self.fs.insert(command.path, command.insert_line, command.insert_text))

    async def delete(self, command: BetaMemoryTool20250818DeleteCommand) -> str:  # type: ignore[override]
        return await _failing(self.fs.delete(command.path))

    async def rename(self, command: BetaMemoryTool20250818RenameCommand) -> str:  # type: ignore[override]
        return await _failing(self.fs.rename(command.old_path, command.new_path))

    async def recall(self, query: str, k: int = 5) -> str:
        """Semantic recall over memory files: the files most relevant to ``query``, best first."""
        return await self.fs.recall(query, k)

    def recall_tool(self) -> BetaAsyncFunctionTool[Any]:
        """A companion tool that lets Claude search its memory files by meaning.

        Add it next to this tool: ``tools=[memory, memory.recall_tool()]``.
        """
        owner = self

        async def memory_recall(query: str, k: int = 5) -> str:
            try:
                return await owner.recall(query, k)
            except MemoryToolFailure as error:
                raise ToolError(str(error)) from error

        memory_recall.__doc__ = _RECALL_DESCRIPTION
        from anthropic import beta_async_tool

        return beta_async_tool(memory_recall)


class _LoopThread:
    """A private event loop on a daemon thread, so the sync tool can drive the async MCP client."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, name="warm-memory-tool", daemon=True)
        self._thread.start()

    def run(self, coroutine: Coroutine[Any, Any, T]) -> T:
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result()

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)


class WarmMemoryToolSync(BetaAbstractMemoryTool):
    """Sync memory tool: the same commands, for `client.beta.messages.tool_runner`."""

    def __init__(
        self,
        store: FileStore,
        *,
        max_view_chars: int = 16_000,
        edit_retries: int = 25,
        loop: Optional[_LoopThread] = None,
        **kwargs: Any,
    ) -> None:
        """``store`` must be usable from the loop thread; pass ``loop`` only when ``store`` was created on it."""
        super().__init__(**kwargs)
        self._loop = loop or _LoopThread()
        self.fs = MemoryFs(store, max_view_chars=max_view_chars, edit_retries=edit_retries)

    @classmethod
    def connect(cls, *, max_view_chars: int = 16_000, edit_retries: int = 25, **connection: Any) -> "WarmMemoryToolSync":
        """Connect to warm-mcp: ``command=``/``args=``/``env=`` for stdio, or ``url=``/``token=``."""
        loop = _LoopThread()
        try:
            store = loop.run(McpFileStore.connect(**connection))
        except BaseException:
            loop.stop()
            raise
        return cls(store, max_view_chars=max_view_chars, edit_retries=edit_retries, loop=loop)

    def close(self) -> None:
        try:
            self._loop.run(self.fs.store.close())
        finally:
            self._loop.stop()

    def __enter__(self) -> "WarmMemoryToolSync":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _run(self, coroutine: Coroutine[Any, Any, str]) -> str:
        return self._loop.run(_failing(coroutine))

    def view(self, command: BetaMemoryTool20250818ViewCommand) -> str:  # type: ignore[override]
        return self._run(self.fs.view(command.path, getattr(command, "view_range", None) or None))

    def create(self, command: BetaMemoryTool20250818CreateCommand) -> str:  # type: ignore[override]
        return self._run(self.fs.create(command.path, command.file_text))

    def str_replace(self, command: BetaMemoryTool20250818StrReplaceCommand) -> str:  # type: ignore[override]
        return self._run(self.fs.str_replace(command.path, command.old_str, getattr(command, "new_str", None)))

    def insert(self, command: BetaMemoryTool20250818InsertCommand) -> str:  # type: ignore[override]
        return self._run(self.fs.insert(command.path, command.insert_line, command.insert_text))

    def delete(self, command: BetaMemoryTool20250818DeleteCommand) -> str:  # type: ignore[override]
        return self._run(self.fs.delete(command.path))

    def rename(self, command: BetaMemoryTool20250818RenameCommand) -> str:  # type: ignore[override]
        return self._run(self.fs.rename(command.old_path, command.new_path))

    def recall(self, query: str, k: int = 5) -> str:
        return self._loop.run(self.fs.recall(query, k))

    def recall_tool(self) -> BetaFunctionTool[Any]:
        owner = self

        def memory_recall(query: str, k: int = 5) -> str:
            try:
                return owner.recall(query, k)
            except MemoryToolFailure as error:
                raise ToolError(str(error)) from error

        memory_recall.__doc__ = _RECALL_DESCRIPTION
        from anthropic import beta_tool

        return beta_tool(memory_recall)


__all__ = ["WarmMemoryTool", "WarmMemoryToolSync", "MemoryToolFailure"]
