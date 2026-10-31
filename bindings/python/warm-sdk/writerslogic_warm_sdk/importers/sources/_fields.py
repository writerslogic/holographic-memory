# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Helpers for sources that read JSON-ish documents (Elasticsearch, Weaviate, Vespa, Pinecone)."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import httpx

from ..model import ImporterError

DEFAULT_TEXT_FIELDS = ("text", "content", "body", "chunk_text", "chunk", "page_content", "document", "message", "description")


def get_path(doc: Mapping[str, Any], path: str) -> Any:
    """Value at a dotted path ('a.b'); a literal key containing dots wins over nesting."""
    if path in doc:
        return doc[path]
    cur: Any = doc
    for part in path.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def has_path(doc: Mapping[str, Any], path: str) -> bool:
    if path in doc:
        return True
    cur: Any = doc
    for part in path.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return False
    return True


def without_paths(doc: Mapping[str, Any], paths: Sequence[str]) -> Dict[str, Any]:
    """Copy of `doc` minus the given dotted paths (shallow copies along each path only)."""
    out: Dict[str, Any] = dict(doc)
    for path in paths:
        if not path:
            continue
        if path in out:
            del out[path]
            continue
        parts = path.split(".")
        node: Any = out
        trail = []
        for part in parts[:-1]:
            if isinstance(node, dict) and isinstance(node.get(part), dict):
                node[part] = dict(node[part])
                trail.append((node, part))
                node = node[part]
            else:
                node = None
                break
        if isinstance(node, dict):
            node.pop(parts[-1], None)
            for parent, key in reversed(trail):  # drop containers emptied by the removal
                if not parent[key]:
                    del parent[key]
    return out


def text_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return "\n".join(value)
    return json.dumps(value, ensure_ascii=False, default=str)


def pick_text_field(doc: Mapping[str, Any], preferred: Optional[str]) -> Optional[str]:
    """The field holding the text: the requested one, else the first conventional name present."""
    if preferred:
        return preferred if has_path(doc, preferred) else None
    for name in DEFAULT_TEXT_FIELDS:
        if isinstance(doc.get(name), str):
            return name
    return None


def numeric_vector(value: Any) -> Optional[Sequence[float]]:
    """A flat list of numbers from a list value, or from a one-element list wrapping it."""
    if isinstance(value, (list, tuple)) and value:
        if isinstance(value[0], (list, tuple)) and len(value) == 1:
            value = value[0]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            return list(value)
    return None


def http_error(exc: httpx.HTTPError, what: str) -> ImporterError:
    """A printable error that carries the status and no request headers or URL credentials."""
    if isinstance(exc, httpx.HTTPStatusError):
        body = exc.response.text[:200].replace("\n", " ")
        return ImporterError(f"{what}: HTTP {exc.response.status_code} {body}")
    return ImporterError(f"{what}: {type(exc).__name__}")


def httpx_verify(insecure: bool, ca_cert: Optional[str]) -> Any:
    if insecure:
        return False
    return ca_cert or True


__all__: Tuple[str, ...] = (
    "DEFAULT_TEXT_FIELDS", "get_path", "has_path", "without_paths", "text_value", "pick_text_field",
    "numeric_vector", "http_error", "httpx_verify",
)
