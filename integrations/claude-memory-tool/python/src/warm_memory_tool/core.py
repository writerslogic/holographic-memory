# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""The six memory_20250818 commands over a `FileStore`, with the result strings the documented
reference behaviour uses, plus semantic recall. Framework-agnostic: errors are `MemoryToolFailure`."""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass
from typing import Optional, Sequence

from .paths import MemoryPathError, ROOT, ancestors, is_inside, normalize_path
from .store import ConflictError, FileEntry, FileStore, WarmToolError

LINE_NUMBER_WIDTH = 6
MAX_LINES = 999_999
LISTING_DEPTH = 2


class MemoryToolFailure(Exception):
    """A failure to report to Claude as an error tool result."""


def human_size(size: int) -> str:
    if size == 0:
        return "0B"
    units = ["B", "K", "M", "G"]
    i = min(int(math.log2(size)) // 10, len(units) - 1)
    value = size / (1024**i)
    return f"{int(value)}{units[i]}" if value == int(value) else f"{value:.1f}{units[i]}"


def numbered(lines: Sequence[str], start: int) -> str:
    return "\n".join(f"{str(i + start).rjust(LINE_NUMBER_WIDTH)}\t{line}" for i, line in enumerate(lines))


def split_lines(text: str) -> list[str]:
    """Lines of a file the way the tool counts them: a trailing newline adds no empty line."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


@dataclass
class _Node:
    name: str
    bytes: int = 0
    is_dir: bool = True

    def __post_init__(self) -> None:
        self.children: dict[str, _Node] = {}


class MemoryFs:
    def __init__(self, store: FileStore, *, max_view_chars: int = 16_000, edit_retries: int = 25) -> None:
        self.store = store
        self.max_view_chars = max_view_chars
        self.edit_retries = edit_retries

    @staticmethod
    def _path(path: object) -> str:
        try:
            return normalize_path(path)
        except MemoryPathError as error:
            raise MemoryToolFailure(str(error)) from error

    async def _guard(self, call):  # type: ignore[no-untyped-def]
        try:
            return await call
        except WarmToolError as error:
            raise MemoryToolFailure(str(error)) from error

    async def _kind(self, path: str) -> str:
        if path == ROOT:
            return "dir"
        if await self.store.get(path):
            return "file"
        return "dir" if await self.store.list(path + "/") else "none"

    def _check_size(self, text: str) -> None:
        size = len(text.encode())
        if size > self.store.max_file_bytes:
            raise MemoryToolFailure(
                f"Error: the file would be {size} bytes; the limit is {self.store.max_file_bytes}. "
                "Split the content into several files."
            )

    async def _check_creatable(self, path: str) -> None:
        for ancestor in ancestors(path):
            if await self.store.get(ancestor):
                raise MemoryToolFailure(f"Error: {ancestor} is a file, so {path} cannot be created inside it")
        if await self.store.list(path + "/"):
            raise MemoryToolFailure(f"Error: {path} is a directory")

    # ---- view ---------------------------------------------------------------------------

    async def view(self, raw_path: str, view_range: Optional[Sequence[int]] = None) -> str:
        path = self._path(raw_path)
        file = None if path == ROOT else await self._guard(self.store.get(path))
        if file is not None:
            return self._view_file(path, file.text, view_range)
        entries = await self._guard(self.store.list(path + "/"))
        if not entries and path != ROOT:
            raise MemoryToolFailure(f"The path {raw_path} does not exist. Please provide a valid path.")
        return self._view_directory(path, entries)

    def _view_file(self, path: str, text: str, view_range: Optional[Sequence[int]]) -> str:
        lines = split_lines(text)
        if len(lines) > MAX_LINES:
            raise MemoryToolFailure(f"File {path} exceeds maximum line limit of 999,999 lines.")
        start, shown = 1, lines
        if view_range is not None and len(view_range) == 2:
            first, last = view_range
            if not isinstance(first, int) or not isinstance(last, int) or first < 1 or (last != -1 and last < first):
                raise MemoryToolFailure(
                    f"Error: invalid view_range {list(view_range)}; use [start, end] with 1 <= start <= end, or end = -1"
                )
            start = first
            shown = lines[first - 1 :] if last == -1 else lines[first - 1 : last]
        header = f"Here's the content of {path} with line numbers:\n"
        body = numbered(shown, start)
        if len(header) + len(body) > self.max_view_chars:
            kept: list[str] = []
            size = len(header)
            for i, line in enumerate(shown):
                row = f"{str(i + start).rjust(LINE_NUMBER_WIDTH)}\t{line}"
                if size + len(row) + 1 > self.max_view_chars and kept:
                    break
                kept.append(row if len(row) <= self.max_view_chars else row[: self.max_view_chars - 200] + " [line truncated]")
                size += len(row) + 1
            body = (
                "\n".join(kept)
                + f"\n[Output truncated at {self.max_view_chars} characters. "
                f"Use view_range starting at line {start + len(kept)} to read the rest.]"
            )
        return header + body

    def _view_directory(self, path: str, entries: Sequence[FileEntry]) -> str:
        root = _Node("")
        prefix = path + "/"
        for entry in entries:
            parts = entry.path[len(prefix) :].split("/")
            node = root
            node.bytes += entry.bytes
            for i, part in enumerate(parts):
                child = node.children.get(part)
                if child is None:
                    child = node.children[part] = _Node(part, is_dir=i < len(parts) - 1)
                child.bytes += entry.bytes
                node = child
        rows = [f"{human_size(root.bytes)}\t{path}"]

        def walk(node: _Node, base: str, depth: int) -> None:
            for child in sorted(node.children.values(), key=lambda c: c.name):
                if child.name == "node_modules":
                    continue
                shown = f"{base}/{child.name}"
                rows.append(f"{human_size(child.bytes)}\t{shown}{'/' if child.is_dir else ''}")
                if child.is_dir and depth < LISTING_DEPTH:
                    walk(child, shown, depth + 1)

        walk(root, path, 1)
        return (
            f"Here're the files and directories up to 2 levels deep in {path}, "
            "excluding hidden items and node_modules:\n" + "\n".join(rows)
        )

    # ---- create / edit ------------------------------------------------------------------

    async def create(self, raw_path: str, file_text: str) -> str:
        path = self._path(raw_path)
        if path == ROOT:
            raise MemoryToolFailure("Error: /memories is a directory")
        self._check_size(file_text)
        await self._guard(self._check_creatable(path))
        try:
            await self._guard(self.store.put(path, file_text, create_only=True))
        except ConflictError as error:
            if error.kind == "exists":
                raise MemoryToolFailure(f"Error: File {raw_path} already exists") from error
            raise
        return f"File created successfully at: {raw_path}"

    async def _edit(self, path: str, not_found: str, change) -> str:  # type: ignore[no-untyped-def]
        """Read-modify-write with compare-and-set, retried when another writer got in between."""
        for attempt in range(self.edit_retries):
            file = await self._guard(self.store.get(path))
            if file is None:
                raise MemoryToolFailure(not_found)
            text, result = change(file.text)
            self._check_size(text)
            try:
                await self._guard(self.store.put(path, text, if_version=file.version))
                return result
            except ConflictError as error:
                if error.kind == "version":
                    await asyncio.sleep(random.random() * 0.005 * (attempt + 1))
                    continue
                raise
        raise MemoryToolFailure(f"Error: {path} kept changing while it was edited; try again")

    async def str_replace(self, raw_path: str, old_str: str, new_str: Optional[str]) -> str:
        path = self._path(raw_path)
        if not isinstance(old_str, str) or not old_str:
            raise MemoryToolFailure("Error: old_str must not be empty")
        replacement = new_str or ""

        def change(text: str) -> tuple[str, str]:
            positions: list[int] = []
            at = text.find(old_str)
            while at != -1:
                positions.append(at)
                at = text.find(old_str, at + 1)
            if not positions:
                raise MemoryToolFailure(
                    f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {raw_path}."
                )
            if len(positions) > 1:
                lines = ", ".join(str(text[:p].count("\n") + 1) for p in positions)
                raise MemoryToolFailure(
                    f"No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines: {lines}. "
                    "Please ensure it is unique"
                )
            at = positions[0]
            updated = text[:at] + replacement + text[at + len(old_str) :]
            changed_line = text[:at].count("\n")
            all_lines = updated.split("\n")
            start = max(0, changed_line - 2)
            end = min(len(all_lines), changed_line + 3)
            snippet = numbered(all_lines[start:end], start + 1)
            return updated, (
                "The memory file has been edited. Here is the snippet showing the change (with line numbers):\n"
                + snippet
            )

        return await self._edit(
            path, f"Error: The path {raw_path} does not exist. Please provide a valid path.", change
        )

    async def insert(self, raw_path: str, insert_line: int, insert_text: str) -> str:
        path = self._path(raw_path)

        def change(text: str) -> tuple[str, str]:
            lines = split_lines(text)
            if not isinstance(insert_line, int) or insert_line < 0 or insert_line > len(lines):
                raise MemoryToolFailure(
                    f"Error: Invalid `insert_line` parameter: {insert_line}. "
                    f"It should be within the range of lines of the file: [0, {len(lines)}]"
                )
            lines.insert(insert_line, insert_text.rstrip("\n"))
            return "\n".join(lines) + "\n", f"The file {raw_path} has been edited."

        return await self._edit(path, f"Error: The path {raw_path} does not exist", change)

    # ---- delete / rename ----------------------------------------------------------------

    async def delete(self, raw_path: str) -> str:
        path = self._path(raw_path)
        if path == ROOT:
            raise MemoryToolFailure("Error: Cannot delete the /memories directory itself")
        kind = await self._guard(self._kind(path))
        if kind == "none":
            raise MemoryToolFailure(f"Error: The path {raw_path} does not exist")
        if kind == "file":
            await self._guard(self.store.delete([path]))
        else:
            await self._guard(self.store.delete_prefix(path + "/"))
        return f"Successfully deleted {raw_path}"

    async def rename(self, raw_old: str, raw_new: str) -> str:
        source = self._path(raw_old)
        target = self._path(raw_new)
        if source == ROOT:
            raise MemoryToolFailure("Error: Cannot rename the /memories directory itself")
        if target == ROOT or is_inside(target, source):
            raise MemoryToolFailure(f"Error: Cannot move {raw_old} into itself or onto /memories")
        kind = await self._guard(self._kind(source))
        if kind == "none":
            raise MemoryToolFailure(f"Error: The path {raw_old} does not exist")
        if await self._guard(self._kind(target)) != "none":
            raise MemoryToolFailure(f"Error: The destination {raw_new} already exists")
        for ancestor in ancestors(target):
            if await self._guard(self.store.get(ancestor)):
                raise MemoryToolFailure(f"Error: {ancestor} is a file, so {raw_new} cannot be created inside it")
        sources = [source] if kind == "file" else [e.path for e in await self._guard(self.store.list(source + "/"))]
        created: list[str] = []
        try:
            for path in sources:
                file = await self._guard(self.store.get(path))
                if file is None:
                    raise MemoryToolFailure(f"Error: The path {path} does not exist")
                destination = target if kind == "file" else target + path[len(source) :]
                await self._guard(self.store.put(destination, file.text, create_only=True))
                created.append(destination)
        except ConflictError as error:
            await self._undo(created)
            raise MemoryToolFailure(f"Error: The destination {raw_new} already exists") from error
        except BaseException:
            # Copy first, delete second: a failure here leaves the original intact.
            await self._undo(created)
            raise
        await self._guard(self.store.delete(sources))
        return f"Successfully renamed {raw_old} to {raw_new}"

    async def _undo(self, created: Sequence[str]) -> None:
        try:
            await self.store.delete(list(created))
        except Exception:  # noqa: BLE001 - best effort; the originals are untouched
            pass

    # ---- recall -------------------------------------------------------------------------

    async def recall(self, query: str, k: int = 5) -> str:
        if not isinstance(query, str) or not query.strip():
            raise MemoryToolFailure("Error: query must be a non-empty string")
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 50:
            raise MemoryToolFailure("Error: k must be an integer from 1 to 50")
        hits = await self._guard(self.store.recall(query, k))
        if not hits:
            return "No memory files match that query."
        rows = []
        for i, hit in enumerate(hits, 1):
            snippet = hit.text if len(hit.text) <= 400 else hit.text[:400] + "..."
            rows.append(f"{i}. {hit.path} (relevance {hit.score:.2f})\n{snippet}")
        return "\n\n".join(rows)
