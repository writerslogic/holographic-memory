# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Storage behind the memory tool: files addressed by path, kept in WARM through
the warm-mcp server so every file is also searchable."""

from __future__ import annotations

import asyncio
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

from mcp import Client, StdioServerParameters

TAG = "memory-file"
_PAGE = 200


@dataclass(frozen=True)
class StoredFile:
    path: str
    text: str
    version: str


@dataclass(frozen=True)
class FileEntry:
    path: str
    bytes: int


@dataclass(frozen=True)
class RecallHit:
    path: str
    score: float
    text: str


class ConflictError(Exception):
    """A put precondition failed: kind is ``"exists"`` or ``"version"``."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class WarmToolError(Exception):
    def __init__(self, code: str, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(f"{message} ({code})")
        self.code = code
        self.message = message
        self.details = details


class FileStore(Protocol):
    max_file_bytes: int

    async def get(self, path: str) -> Optional[StoredFile]:
        """The file at ``path``, or None."""

    async def put(
        self, path: str, text: str, *, create_only: bool = False, if_version: Optional[str] = None
    ) -> str:
        """Create (``create_only``) or replace (``if_version``) a file; returns its new version."""

    async def list(self, prefix: str) -> list[FileEntry]:
        """Every file whose path starts with ``prefix``, in path order."""

    async def delete(self, paths: Sequence[str]) -> int:
        """Delete the given files; returns how many existed."""

    async def delete_prefix(self, prefix: str) -> int:
        """Delete every file under ``prefix``."""

    async def recall(self, query: str, k: int) -> list[RecallHit]:
        """The files most relevant to ``query``, best first."""

    async def close(self) -> None:
        """Release the connection."""


class McpFileStore:
    """`FileStore` over an warm-mcp server: stdio (spawns the binary) or streamable HTTP.

    The connection lives in a task of its own, opened and closed there, so `close()` may be
    called from any task (the MCP client's task groups must exit in the task that entered them).
    Requires asyncio.
    """

    def __init__(self, client: Client, namespace: Optional[str], max_file_bytes: int) -> None:
        self._client = client
        self._namespace = namespace
        self.max_file_bytes = max_file_bytes
        self._stop: Optional[asyncio.Event] = None
        self._owner: Optional["asyncio.Task[None]"] = None

    @classmethod
    async def connect(
        cls,
        *,
        command: str = "warm-mcp",
        args: Sequence[str] = (),
        env: Optional[dict[str, str]] = None,
        url: Optional[str] = None,
        token: Optional[str] = None,
        namespace: Optional[str] = None,
        max_file_bytes: Optional[int] = None,
        request_timeout: float = 60.0,
    ) -> "McpFileStore":
        """Spawn ``command`` over stdio, or connect to ``url`` (an ``warm-mcp --transport http`` endpoint).

        Every request, the handshake included, fails after ``request_timeout`` seconds, for example when the server
        exits at startup because its store is held by another process.
        """
        loop = asyncio.get_running_loop()
        ready: "asyncio.Future[McpFileStore]" = loop.create_future()
        stop = asyncio.Event()

        async def owner() -> None:
            stack = AsyncExitStack()
            try:
                store = await cls._open(stack, command, args, env, url, token, namespace, max_file_bytes, request_timeout)
            except Exception as error:  # handed to the caller below
                try:
                    await stack.aclose()
                except Exception:  # the open failure is the useful one
                    pass
                if not ready.done():
                    ready.set_exception(error)
                return
            ready.set_result(store)
            await stop.wait()
            await stack.aclose()

        task = loop.create_task(owner())
        try:
            store = await ready
        except (Exception, asyncio.CancelledError):
            task.cancel()
            raise
        store._stop, store._owner = stop, task
        return store

    @classmethod
    async def _open(
        cls,
        stack: AsyncExitStack,
        command: str,
        args: Sequence[str],
        env: Optional[dict[str, str]],
        url: Optional[str],
        token: Optional[str],
        namespace: Optional[str],
        max_file_bytes: Optional[int],
        timeout: float,
    ) -> "McpFileStore":
        if url is not None:
            token = token or os.environ.get("WARM_MCP_TOKEN")
            if token:
                import httpx2
                from mcp.client.streamable_http import streamable_http_client

                http = httpx2.AsyncClient(headers={"authorization": f"Bearer {token}"})
                await stack.enter_async_context(http)
                transport = streamable_http_client(url, http_client=http)
                client = await stack.enter_async_context(Client(transport, read_timeout_seconds=timeout))
            else:
                client = await stack.enter_async_context(Client(url, read_timeout_seconds=timeout))
        else:
            merged = {**os.environ, **(env or {})}
            params = StdioServerParameters(command=command, args=list(args), env=merged)
            client = await stack.enter_async_context(Client(params, read_timeout_seconds=timeout))
        store = cls(client, namespace, 2**53)
        stats = await store._call("stats", {})
        reported = int(stats["limits"]["max_text_bytes"])
        store.max_file_bytes = min(max_file_bytes, reported) if max_file_bytes else reported
        return store

    async def _call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._namespace:
            arguments = {**arguments, "namespace": self._namespace}
        result = await self._client.call_tool(name, arguments)
        structured = result.structured_content
        if result.is_error:
            error = (structured or {}).get("error", {})
            raise WarmToolError(error.get("code", "unknown"), error.get("message", "tool call failed"), error.get("details"))
        return structured or {}

    async def get(self, path: str) -> Optional[StoredFile]:
        try:
            out = await self._call("get", {"id": path})
        except WarmToolError as error:
            if error.code == "not_found":
                return None
            raise
        return StoredFile(path, out["memory"]["text"], out["memory"]["version"])

    async def put(
        self, path: str, text: str, *, create_only: bool = False, if_version: Optional[str] = None
    ) -> str:
        arguments: dict[str, Any] = {"id": path, "text": text, "tags": [TAG], "allow_empty": True}
        if create_only:
            arguments["create_only"] = True
        if if_version:
            arguments["if_version"] = if_version
        try:
            out = await self._call("remember", arguments)
        except WarmToolError as error:
            if error.code == "already_exists":
                raise ConflictError("exists", error.message) from error
            if error.code == "version_conflict":
                raise ConflictError("version", error.message) from error
            raise
        return out["version"]

    async def list(self, prefix: str) -> list[FileEntry]:
        entries: list[FileEntry] = []
        cursor: Optional[str] = None
        while True:
            arguments: dict[str, Any] = {"prefix": prefix, "limit": _PAGE, "tag": TAG}
            if cursor:
                arguments["cursor"] = cursor
            out = await self._call("list", arguments)
            entries.extend(FileEntry(m["id"], m["bytes"]) for m in out["memories"])
            cursor = out.get("next_cursor")
            if not cursor:
                return entries

    async def delete(self, paths: Sequence[str]) -> int:
        deleted = 0
        for i in range(0, len(paths), 500):
            out = await self._call("forget", {"ids": list(paths[i : i + 500])})
            deleted += out["deleted"]
        return deleted

    async def delete_prefix(self, prefix: str) -> int:
        return await self.delete([entry.path for entry in await self.list(prefix)])

    async def recall(self, query: str, k: int) -> list[RecallHit]:
        out = await self._call("recall", {"query": query, "k": k, "tags": [TAG]})
        return [RecallHit(r["memory"]["id"], r["score"], r["memory"]["text"]) for r in out["results"]]

    async def close(self) -> None:
        if self._stop is not None and self._owner is not None:
            self._stop.set()
            await self._owner
