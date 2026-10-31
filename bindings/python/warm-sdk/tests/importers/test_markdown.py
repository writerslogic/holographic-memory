# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Markdown memory folders against real temporary directories (no mocks involved)."""

import os
import sys
import types

import pytest

from writerslogic_warm_sdk.importers.sources.markdown import MarkdownSource, parse_frontmatter, split_sections

from .conftest import assert_faithful, assert_resumable, vec


@pytest.fixture
def embedder(monkeypatch):
    mod = types.ModuleType("md_test_embedder")
    mod.embed = lambda texts: [vec(sum(map(ord, t)) % 10007) for t in texts]
    monkeypatch.setitem(sys.modules, "md_test_embedder", mod)
    return "md_test_embedder:embed"


@pytest.fixture
def memory_dir(tmp_path):
    root = tmp_path / "mem"
    (root / "memory").mkdir(parents=True)
    (root / "CLAUDE.md").write_text("# Project\n\nUse uv for Python.\n\n## Style\n\nTerse answers.\n")
    (root / "memory" / "MEMORY.md").write_text("- [Role](user_role.md) - who the user is\n")
    (root / "memory" / "user_role.md").write_text(
        "---\nname: user role\ndescription: data scientist\ntype: user\ntags: [a, b]\n---\n\nDeep Go expertise.\n"
    )
    (root / "memory" / "only_front.md").write_text("---\nname: stub\ndescription: frontmatter only\n---\n")
    long_body = "# Big\n\n" + "intro words " * 5 + "\n\n## One\n\n" + "alpha " * 30 + "\n\n## One\n\n" + "beta " * 30 + "\n"
    (root / "notes.md").write_text(long_body)
    return root


def test_frontmatter_and_sections():
    meta, body = parse_frontmatter("---\nname: x\nn: 3\nflag: true\nitems:\n  - a\n  - b\n---\nbody\n")
    assert meta == {"name": "x", "n": 3, "flag": True, "items": ["a", "b"]} and body == "body\n"
    assert parse_frontmatter("no frontmatter") == ({}, "no frontmatter")
    secs = split_sections("pre\n\n# A\nx\n```\n# not a heading\n```\n## B\ny\n")
    assert [p for p, _ in secs] == [[], ["A"], ["A", "B"]]
    assert "# not a heading" in secs[1][1]


def test_markdown_real_roundtrip(memory_dir, run, server, embedder):
    src = MarkdownSource(str(memory_dir), [], [], "auto", 40, "agent-a")
    report, tenant = run(src, embedder=embedder)
    ids = report.stats.counts["docs_acked"]
    assert report.stats.rejected_total == 0
    assert_faithful(report, ids, vectors=False)
    sink = server.sink(tenant)
    try:
        got = {m["id"]: m for m in sink.query(vector=vec(1), top_k=100)}
    finally:
        sink.close()
    assert set(got) == {
        "agent-a::CLAUDE.md", "agent-a::memory/MEMORY.md", "agent-a::memory/user_role.md",
        "agent-a::memory/only_front.md", "agent-a::notes.md#big", "agent-a::notes.md#one", "agent-a::notes.md#one-2",
    }
    role = got["agent-a::memory/user_role.md"]["metadata"]
    assert role["description"] == "data scientist" and role["tags"] == ["a", "b"] and role["hms_namespace"] == "agent-a"
    assert "warm_created_at" in role and "warm_updated_at" in role
    assert "frontmatter only" in got["agent-a::memory/only_front.md"]["text"]
    assert got["agent-a::notes.md#one-2"]["metadata"]["heading_path"] == ["Big", "One"]


def test_symlink_escape_and_binary_are_rejected_not_lost(memory_dir, tmp_path, run, embedder):
    outside = tmp_path / "secret.md"
    outside.write_text("outside the memory folder")
    os.symlink(outside, memory_dir / "link.md")
    (memory_dir / "bad.md").write_bytes(b"\xff\xfe\x00binary")
    report, _ = run(MarkdownSource(str(memory_dir), [], [], "none", 800, ""), sample=0, embedder=embedder,
                    allow_rejects=True)
    assert report.stats.rejects == {"source_unreadable": 2}
    assert sorted(report.stats.reject_samples["source_unreadable"]) == ["bad.md", "link.md"]


def test_markdown_resume(memory_dir, server, tmp_path, embedder):
    for i in range(12):
        (memory_dir / f"extra{i:02d}.md").write_text(f"note {i}\n")
    expected = 4 + 12 + 1  # CLAUDE.md, MEMORY.md, user_role.md, only_front.md, notes.md, extras
    assert_resumable(lambda: MarkdownSource(str(memory_dir), [], [], "none", 800, ""), server, tmp_path,
                     expected_docs=expected, batch_size=4, embedder=embedder)
