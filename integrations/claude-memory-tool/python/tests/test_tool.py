# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Drives the backend with SDK-typed commands against a real hms-mcp process (HMS_MCP_BIN)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, AsyncIterator

import anyio
import httpx2
import pytest
from anthropic import Anthropic, AsyncAnthropic
from anthropic.lib.tools import ToolError
from anthropic.types.beta import (
    BetaMemoryTool20250818CreateCommand,
    BetaMemoryTool20250818DeleteCommand,
    BetaMemoryTool20250818InsertCommand,
    BetaMemoryTool20250818RenameCommand,
    BetaMemoryTool20250818StrReplaceCommand,
    BetaMemoryTool20250818ViewCommand,
)

from hms_memory_tool import (
    HolographicMemoryTool,
    HolographicMemoryToolSync,
    MemoryPathError,
    normalize_path,
)

BIN = os.environ.get("HMS_MCP_BIN")
pytestmark = [
    pytest.mark.skipif(not BIN, reason="set HMS_MCP_BIN to the hms-mcp binary"),
    pytest.mark.anyio,
]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith("HMS_MCP_")}


async def open_tool(tmp_path: Path, *extra: str, namespace: str | None = None) -> HolographicMemoryTool:
    assert BIN
    return await HolographicMemoryTool.connect(
        command=BIN,
        args=["--data-dir", str(tmp_path / "data"), "--namespace-arg", *extra],
        env=clean_env(),
        namespace=namespace,
    )


@pytest.fixture
async def tool(tmp_path: Path) -> AsyncIterator[HolographicMemoryTool]:
    t = await open_tool(tmp_path)
    try:
        yield t
    finally:
        await t.aclose()


async def run(tool: HolographicMemoryTool, **command: Any) -> str:
    """The runner entry point: raw tool input in, tool result text out."""
    result = await tool.call(command)
    assert isinstance(result, str)
    return result


def failing(tool: HolographicMemoryTool, pattern: str, **command: Any):
    class _Ctx:
        async def __aenter__(self) -> None: ...

        async def __aexit__(self, exc_type, exc, tb) -> bool:  # type: ignore[no-untyped-def]
            assert exc_type is ToolError, f"expected ToolError, got {exc_type}"
            assert re.search(pattern, str(exc)), f"{pattern!r} not in {exc}"
            return True

    return _Ctx()


@pytest.mark.parametrize(
    "bad",
    [
        "/memories/../etc/passwd",
        "/memories/a/../../b",
        "/memories/..",
        "/memories/%2e%2e/x",
        "/memories/%2E%2E%2Fx",
        "/memories/..%2fx",
        "/memories/a%5cb",
        "/memories/a\\b",
        "/memories/a/./b",
        "/memories//a",
        "/memories/.hidden",
        "/memoriesx",
        "/memories-other/a",
        "memories/a",
        "/etc/passwd",
        "",
        "/memories/a\x00b",
        "/memories/a\nb",
        "/memories/" + "x" * 129,
        "/memories/" + "a/" * 17 + "f",
    ],
)
def test_path_validation_rejects(bad: str) -> None:
    with pytest.raises(MemoryPathError):
        normalize_path(bad)


def test_path_validation_accepts() -> None:
    assert normalize_path("/memories") == "/memories"
    assert normalize_path("/memories/") == "/memories"
    assert normalize_path("/memories/a/b.md") == "/memories/a/b.md"
    assert normalize_path("/memories/a/") == "/memories/a"
    assert normalize_path("/memories/100%.md") == "/memories/100%.md"
    assert normalize_path("/memories/naïve café.txt") == "/memories/naïve café.txt"


async def test_every_command_validates_before_touching_storage(tool: HolographicMemoryTool) -> None:
    evil = "/memories/../../secrets.env"
    for command in (
        {"command": "view", "path": evil},
        {"command": "create", "path": evil, "file_text": "x"},
        {"command": "str_replace", "path": evil, "old_str": "a", "new_str": "b"},
        {"command": "insert", "path": evil, "insert_line": 0, "insert_text": "x"},
        {"command": "delete", "path": evil},
        {"command": "rename", "old_path": "/memories/a", "new_path": evil},
        {"command": "rename", "old_path": evil, "new_path": "/memories/b"},
    ):
        async with failing(tool, "escape"):
            await tool.call(command)
    async with failing(tool, "encoded"):
        await tool.call({"command": "view", "path": "/memories/%2e%2e/x"})
    assert await tool.fs.store.list("/") == []


async def test_command_walkthrough(tool: HolographicMemoryTool) -> None:
    assert await run(tool, command="view", path="/memories") == (
        "Here're the files and directories up to 2 levels deep in /memories, excluding hidden items and node_modules:\n0B\t/memories"
    )
    assert (
        await run(tool, command="create", path="/memories/notes.txt", file_text="Meeting notes:\n- timeline\n- next steps\n")
        == "File created successfully at: /memories/notes.txt"
    )
    async with failing(tool, r"File /memories/notes.txt already exists"):
        await run(tool, command="create", path="/memories/notes.txt", file_text="again")
    assert await run(tool, command="view", path="/memories/notes.txt") == (
        "Here's the content of /memories/notes.txt with line numbers:\n     1\tMeeting notes:\n     2\t- timeline\n     3\t- next steps"
    )
    assert await run(tool, command="view", path="/memories/notes.txt", view_range=[2, 3]) == (
        "Here's the content of /memories/notes.txt with line numbers:\n     2\t- timeline\n     3\t- next steps"
    )
    assert await run(tool, command="view", path="/memories/notes.txt", view_range=[3, -1]) == (
        "Here's the content of /memories/notes.txt with line numbers:\n     3\t- next steps"
    )
    async with failing(tool, "invalid view_range"):
        await run(tool, command="view", path="/memories/notes.txt", view_range=[3, 1])

    edited = await run(tool, command="str_replace", path="/memories/notes.txt", old_str="- timeline", new_str="- project timeline")
    assert edited.startswith("The memory file has been edited.")
    assert "     2\t- project timeline" in edited
    async with failing(tool, "did not appear verbatim"):
        await run(tool, command="str_replace", path="/memories/notes.txt", old_str="absent", new_str="x")
    async with failing(tool, "does not exist"):
        await run(tool, command="str_replace", path="/memories/missing.txt", old_str="a", new_str="b")
    await run(tool, command="create", path="/memories/dup.txt", file_text="x\ny\nx\n")
    async with failing(tool, r"Multiple occurrences of old_str `x` in lines: 1, 3"):
        await run(tool, command="str_replace", path="/memories/dup.txt", old_str="x", new_str="z")
    # The documented schema makes new_str optional: omitted deletes old_str.
    await run(tool, command="str_replace", path="/memories/dup.txt", old_str="y\n")
    assert (await tool.fs.store.get("/memories/dup.txt")).text == "x\nx\n"

    assert (
        await run(tool, command="insert", path="/memories/notes.txt", insert_line=0, insert_text="# Title\n")
        == "The file /memories/notes.txt has been edited."
    )
    await run(tool, command="insert", path="/memories/notes.txt", insert_line=4, insert_text="- last")
    assert (await tool.fs.store.get("/memories/notes.txt")).text == "# Title\nMeeting notes:\n- project timeline\n- next steps\n- last\n"
    async with failing(tool, r"Invalid `insert_line` parameter: 6.*\[0, 5\]"):
        await run(tool, command="insert", path="/memories/notes.txt", insert_line=6, insert_text="x")
    async with failing(tool, "does not exist"):
        await run(tool, command="insert", path="/memories/nope.txt", insert_line=0, insert_text="x")

    assert (
        await run(tool, command="rename", old_path="/memories/notes.txt", new_path="/memories/project/notes.md")
        == "Successfully renamed /memories/notes.txt to /memories/project/notes.md"
    )
    async with failing(tool, r"destination /memories/project/notes.md already exists"):
        await run(tool, command="rename", old_path="/memories/dup.txt", new_path="/memories/project/notes.md")
    async with failing(tool, r"The path /memories/ghost does not exist"):
        await run(tool, command="rename", old_path="/memories/ghost", new_path="/memories/x")
    async with failing(tool, "Cannot rename"):
        await run(tool, command="rename", old_path="/memories", new_path="/memories/x")
    async with failing(tool, "into itself"):
        await run(tool, command="rename", old_path="/memories/project", new_path="/memories/project/inner")

    listing = await run(tool, command="view", path="/memories")
    assert "\t/memories/dup.txt\n" in listing
    assert "\t/memories/project/\n" in listing
    assert listing.endswith("\t/memories/project/notes.md")

    assert await run(tool, command="delete", path="/memories/dup.txt") == "Successfully deleted /memories/dup.txt"
    async with failing(tool, r"The path /memories/dup.txt does not exist"):
        await run(tool, command="delete", path="/memories/dup.txt")
    async with failing(tool, r"Cannot delete the /memories directory"):
        await run(tool, command="delete", path="/memories")
    assert await run(tool, command="delete", path="/memories/project") == "Successfully deleted /memories/project"
    assert await tool.fs.store.list("/memories/") == []


async def test_typed_sdk_commands(tool: HolographicMemoryTool) -> None:
    """The abstract-class entry points take the SDK's own command models."""
    await tool.create(BetaMemoryTool20250818CreateCommand(command="create", path="/memories/t.txt", file_text="a\nb\n"))
    assert "     2\tb" in await tool.view(BetaMemoryTool20250818ViewCommand(command="view", path="/memories/t.txt"))
    await tool.str_replace(BetaMemoryTool20250818StrReplaceCommand(command="str_replace", path="/memories/t.txt", old_str="a", new_str="c"))
    await tool.insert(BetaMemoryTool20250818InsertCommand(command="insert", path="/memories/t.txt", insert_line=2, insert_text="d"))
    await tool.rename(BetaMemoryTool20250818RenameCommand(command="rename", old_path="/memories/t.txt", new_path="/memories/u.txt"))
    assert (await tool.fs.store.get("/memories/u.txt")).text == "c\nb\nd\n"
    await tool.delete(BetaMemoryTool20250818DeleteCommand(command="delete", path="/memories/u.txt"))
    assert await tool.fs.store.list("/memories/") == []
    assert tool.to_dict() == {"type": "memory_20250818", "name": "memory"}


async def test_directories(tool: HolographicMemoryTool) -> None:
    for p in ("a/one.txt", "a/b/two.txt", "a/b/c/three.txt", "z.txt"):
        await run(tool, command="create", path=f"/memories/{p}", file_text=f"{p} body\n")
    listing = await run(tool, command="view", path="/memories")
    assert [line.split("\t")[1] for line in listing.splitlines()[1:]] == [
        "/memories",
        "/memories/a/",
        "/memories/a/b/",
        "/memories/a/one.txt",
        "/memories/z.txt",
    ]
    sub = await run(tool, command="view", path="/memories/a")
    assert [line.split("\t")[1] for line in sub.splitlines()[1:]] == [
        "/memories/a",
        "/memories/a/b/",
        "/memories/a/b/c/",
        "/memories/a/b/two.txt",
        "/memories/a/one.txt",
    ]
    async with failing(tool, "does not exist"):
        await run(tool, command="view", path="/memories/a/one.txt/x")
    async with failing(tool, "is a directory"):
        await run(tool, command="create", path="/memories/a", file_text="x")
    async with failing(tool, "is a file"):
        await run(tool, command="create", path="/memories/z.txt/inner", file_text="x")
    await run(tool, command="rename", old_path="/memories/a", new_path="/memories/moved")
    assert [e.path for e in await tool.fs.store.list("/memories/")] == [
        "/memories/moved/b/c/three.txt",
        "/memories/moved/b/two.txt",
        "/memories/moved/one.txt",
        "/memories/z.txt",
    ]
    await run(tool, command="delete", path="/memories/moved/b")
    assert [e.path for e in await tool.fs.store.list("/memories/moved/")] == ["/memories/moved/one.txt"]
    async with failing(tool, "is a file"):
        await run(tool, command="rename", old_path="/memories/moved", new_path="/memories/z.txt/in")


async def test_paging_and_size_limit(tmp_path: Path) -> None:
    tool = await open_tool(tmp_path, "--max-text-bytes", "4000")
    try:
        lines = [f"line {i + 1} wwww" for i in range(250)]
        await tool.fs.store.put("/memories/long.txt", "\n".join(lines) + "\n", create_only=True)
        tool.fs.max_view_chars = 500
        first = await run(tool, command="view", path="/memories/long.txt")
        assert len(first) < 700
        match = re.search(r"starting at line (\d+)", first)
        assert match, first
        second = await run(tool, command="view", path="/memories/long.txt", view_range=[int(match[1]), -1])
        assert f"{match[1]}\tline {match[1]} wwww" in second
        assert "\tline 1 wwww" not in second
        async with failing(tool, "limit is 4000"):
            await run(tool, command="create", path="/memories/big.txt", file_text="x" * 4001)
        await run(tool, command="create", path="/memories/ok.txt", file_text="x" * 4000)
        async with failing(tool, "limit is 4000"):
            await run(tool, command="insert", path="/memories/ok.txt", insert_line=1, insert_text="more")
        assert len((await tool.fs.store.get("/memories/ok.txt")).text) == 4000
    finally:
        await tool.aclose()


async def test_concurrent_edits_lose_nothing(tool: HolographicMemoryTool) -> None:
    await run(tool, command="create", path="/memories/log.txt", file_text="start\n")

    async def edit(i: int) -> None:
        await run(tool, command="insert", path="/memories/log.txt", insert_line=0, insert_text=f"entry {i}")

    async with anyio.create_task_group() as group:
        for i in range(8):
            group.start_soon(edit, i)
    text = (await tool.fs.store.get("/memories/log.txt")).text
    for i in range(8):
        assert f"\nentry {i}\n" in "\n" + text
    assert "start" in text.splitlines()


async def test_persistence_recall_and_namespaces(tmp_path: Path) -> None:
    tool = await open_tool(tmp_path)
    await run(tool, command="create", path="/memories/prefs/coffee.md", file_text="Alice prefers dark roast coffee with oat milk.\n")
    await run(tool, command="create", path="/memories/work/deploy.md", file_text="Deployments use blue green releases on Kubernetes.\n")
    await tool.aclose()
    tool = await open_tool(tmp_path)
    try:
        assert "dark roast" in await run(tool, command="view", path="/memories/prefs/coffee.md")
        hits = await tool.recall("what kind of release process do deployments use")
        assert hits.startswith("1. /memories/work/deploy.md (relevance 1.00)")
        assert await tool.recall("zzzqqq nonexistent") == "No memory files match that query."
        with pytest.raises(Exception, match="non-empty"):
            await tool.recall("")
        recall = tool.recall_tool()
        assert recall.name == "memory_recall"
        assert "coffee.md" in await recall.call({"query": "coffee"})
    finally:
        await tool.aclose()
    alice = await open_tool(tmp_path, namespace="alice")
    try:
        await run(alice, command="create", path="/memories/x.txt", file_text="alice data\n")
        assert len(await alice.fs.store.list("/memories/")) == 1
    finally:
        await alice.aclose()
    bob = await open_tool(tmp_path, namespace="bob")
    try:
        assert await bob.fs.store.list("/memories/") == []
        async with failing(bob, "does not exist"):
            await run(bob, command="view", path="/memories/x.txt")
    finally:
        await bob.aclose()


async def test_a_held_store_fails_fast_instead_of_hanging(tool: HolographicMemoryTool, tmp_path: Path) -> None:
    assert BIN
    # The fixture's server holds tmp_path/data; a second server on it exits with store_locked.
    with pytest.raises((TimeoutError, Exception)):
        await HolographicMemoryTool.connect(
            command=BIN,
            args=["--data-dir", str(tmp_path / "data")],
            env=clean_env(),
            request_timeout=8,
        )


def message(content: list[dict[str, Any]], stop: str) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": content,
    }


def scripted() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    replies = [
        message(
            [
                {"type": "tool_use", "id": "toolu_1", "name": "memory", "input": {"command": "view", "path": "/memories"}},
                {
                    "type": "tool_use",
                    "id": "toolu_2",
                    "name": "memory",
                    "input": {"command": "create", "path": "/memories/acme.md", "file_text": "Acme Corp prefers email follow-ups.\n"},
                },
            ],
            "tool_use",
        ),
        message(
            [{"type": "tool_use", "id": "toolu_3", "name": "memory", "input": {"command": "view", "path": "/memories/../../etc/passwd"}}],
            "tool_use",
        ),
        message([{"type": "text", "text": "Done."}], "end_turn"),
    ]
    return replies, []


def check_loop(seen: list[dict[str, Any]]) -> None:
    assert seen[0]["tools"][0] == {"type": "memory_20250818", "name": "memory"}
    first = seen[1]["messages"][-1]["content"]
    assert first[0]["tool_use_id"] == "toolu_1"
    assert first[0]["content"].startswith("Here're the files and directories")
    assert first[1]["content"] == "File created successfully at: /memories/acme.md"
    blocked = seen[2]["messages"][-1]["content"][0]
    assert blocked["is_error"] is True
    assert "escape" in json.dumps(blocked["content"])


async def test_async_tool_runner(tool: HolographicMemoryTool) -> None:
    replies, seen = scripted()
    calls = iter(replies)

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json=next(calls))

    client = AsyncAnthropic(api_key="test", max_retries=0, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    runner = client.beta.messages.tool_runner(
        model="claude-sonnet-5-5",
        max_tokens=256,
        messages=[{"role": "user", "content": "Remember that Acme prefers email."}],
        tools=[tool],
        max_iterations=5,
    )
    final = await runner.until_done()
    assert final.stop_reason == "end_turn"
    check_loop(seen)
    assert (await tool.fs.store.get("/memories/acme.md")).text == "Acme Corp prefers email follow-ups.\n"


def test_sync_tool_runner(tmp_path: Path) -> None:
    assert BIN
    sync = HolographicMemoryToolSync.connect(command=BIN, args=["--data-dir", str(tmp_path / "data")], env=clean_env())
    try:
        replies, seen = scripted()
        calls = iter(replies)

        def handler(request: httpx2.Request) -> httpx2.Response:
            seen.append(json.loads(request.content))
            return httpx2.Response(200, json=next(calls))

        client = Anthropic(api_key="test", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
        runner = client.beta.messages.tool_runner(
            model="claude-sonnet-5-5",
            max_tokens=256,
            messages=[{"role": "user", "content": "Remember that Acme prefers email."}],
            tools=[sync, sync.recall_tool()],
            max_iterations=5,
        )
        final = runner.until_done()
        assert final.stop_reason == "end_turn"
        check_loop(seen)
        assert "Acme Corp" in sync.recall("email follow-ups")
        assert any(t.get("name") == "memory_recall" for t in seen[0]["tools"])
    finally:
        sync.close()
