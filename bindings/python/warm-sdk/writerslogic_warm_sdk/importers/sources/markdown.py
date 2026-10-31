# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Markdown agent-memory folders: CLAUDE.md files, Claude memory-tool directories,
Claude Code auto-memory folders (MEMORY.md plus linked files with YAML frontmatter)."""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from ..model import ImporterError, Record
from .base import Source

_DEFAULT_INCLUDE = ["*.md", "*.markdown", "*.txt"]
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """Split a leading `---` YAML block from the body. Unparseable blocks stay in the body."""
    if not text.startswith("---"):
        return {}, text
    lines = text.split("\n")
    if lines[0].strip() != "---":
        return {}, text
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            block = "\n".join(lines[1:i])
            data = _parse_yaml(block)
            if data is None:
                return {}, text
            return data, "\n".join(lines[i + 1:]).lstrip("\n")
    return {}, text


def _parse_yaml(block: str) -> Optional[Dict[str, Any]]:
    try:
        import yaml  # optional; the fallback below covers the flat frontmatter agents write

        data = yaml.safe_load(block)
        return data if isinstance(data, dict) else ({} if data is None else None)
    except ImportError:
        pass
    except Exception:
        return None
    out: Dict[str, Any] = {}
    key: Optional[str] = None
    for raw in block.split("\n"):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_.-]+):\s*(.*)$", raw)
        if m and not raw.startswith((" ", "\t")):
            key, val = m.group(1), m.group(2).strip()
            out[key] = _scalar(val) if val else []
        elif key is not None and raw.lstrip().startswith("- ") and isinstance(out.get(key), list):
            out[key].append(_scalar(raw.lstrip()[2:].strip()))
        else:
            return None
    return out


def _scalar(val: str) -> Any:
    if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
        return val[1:-1]
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        return [_scalar(v.strip()) for v in inner.split(",")] if inner else []
    low = val.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "~", ""):
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        return val


def _slug(title: str) -> str:
    s = re.sub(r"[^\w\s-]", "", title.lower()).strip()
    return re.sub(r"[\s]+", "-", s) or "section"


def split_sections(body: str) -> List[Tuple[List[str], str]]:
    """(heading path, text) per ATX heading section; text before the first heading is its own
    section with an empty path. Fenced code blocks never start a section."""
    sections: List[Tuple[List[str], List[str]]] = [([], [])]
    stack: List[Tuple[int, str]] = []
    fenced = False
    for line in body.split("\n"):
        if _FENCE.match(line):
            fenced = not fenced
        m = None if fenced else _HEADING.match(line)
        if m:
            level, title = len(m.group(1)), m.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            sections.append(([t for _, t in stack], [line]))
        else:
            sections[-1][1].append(line)
    return [(path, "\n".join(lines).strip("\n")) for path, lines in sections if "".join(lines).strip()]


class MarkdownSource(Source):
    name = "markdown"
    summary = "Markdown memory folders: CLAUDE.md-style files, Claude memory-tool directories, Claude Code auto-memory"

    def __init__(self, root: str, include: List[str], exclude: List[str], split: str, max_section_words: int,
                 namespace: str):
        if not os.path.isdir(root):
            raise ImporterError(f"not a directory: {root}")
        if split not in ("auto", "headings", "none"):
            raise ImporterError("--split must be auto, headings or none")
        self.root = os.path.realpath(root)
        self.include = include or list(_DEFAULT_INCLUDE)
        self.exclude = exclude
        self.split = split
        self.max_section_words = max_section_words
        self.namespace = namespace

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--root", required=True, help="memory folder (or a single file's folder)")
        parser.add_argument("--include", action="append", default=[],
                            help="filename glob, repeatable (default: *.md, *.markdown, *.txt)")
        parser.add_argument("--exclude", action="append", default=[], help="relative-path glob to skip, repeatable")
        parser.add_argument("--split", default="auto", choices=["auto", "headings", "none"],
                            help="auto (default): one record per file, per heading section when a file exceeds "
                                 "--max-section-words; headings: always per section; none: always per file")
        parser.add_argument("--max-section-words", type=int, default=800)
        parser.add_argument("--namespace", default="", help="namespace for every record")

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        return cls(args.root, args.include, args.exclude, args.split, args.max_section_words, args.namespace)

    def describe(self) -> Dict[str, Any]:
        return {"root": self.root, "include": self.include, "exclude": self.exclude, "split": self.split,
                "max_section_words": self.max_section_words, "namespace": self.namespace}

    def _files(self) -> Iterator[Tuple[str, str]]:
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            dirnames.sort()
            for fn in sorted(filenames):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, self.root).replace(os.sep, "/")
                if not any(fnmatch.fnmatch(fn, pat) for pat in self.include):
                    continue
                if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(fn, pat) for pat in self.exclude):
                    continue
                yield full, rel

    def count(self) -> Optional[int]:
        return None

    def records(self, cursor: Any = None) -> Iterator[Record]:
        for full, rel in self._files():
            real = os.path.realpath(full)
            if os.path.commonpath([real, self.root]) != self.root:
                yield Record(id=rel, error=f"{rel} is a symlink that leaves the memory folder; not followed")
                continue
            try:
                st = os.stat(real)
                with open(real, "r", encoding="utf-8") as fh:
                    raw = fh.read()
            except (OSError, UnicodeDecodeError) as exc:
                yield Record(id=rel, error=f"{rel}: {type(exc).__name__}")
                continue
            created = getattr(st, "st_birthtime", st.st_mtime)
            front, body = parse_frontmatter(raw.replace("\r\n", "\n"))
            base_meta: Dict[str, Any] = {"path": rel, "filename": os.path.basename(rel)}
            base_meta.update(front)
            if not body.strip() and front:  # a frontmatter-only file still carries its description
                body = "\n".join(f"{k}: {v}" for k, v in front.items() if isinstance(v, (str, int, float)))
            words = len(body.split())
            by_section = self.split == "headings" or (self.split == "auto" and words > self.max_section_words)
            if not by_section:
                title = next((m.group(2) for m in map(_HEADING.match, body.split("\n")) if m), None)
                if title:
                    base_meta["title"] = title
                yield Record(id=rel, text=body, metadata=base_meta, namespace=self.namespace,
                             created_at=created, updated_at=st.st_mtime)
                continue
            seen: Dict[str, int] = {}
            for path, text in split_sections(body):
                slug = _slug(path[-1]) if path else "preamble"
                n = seen[slug] = seen.get(slug, 0) + 1
                sid = f"{rel}#{slug}" if n == 1 else f"{rel}#{slug}-{n}"
                meta = dict(base_meta)
                meta["heading_path"] = path
                if path:
                    meta["title"] = path[-1]
                yield Record(id=sid, text=text, metadata=meta, namespace=self.namespace,
                             created_at=created, updated_at=st.st_mtime)
