# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Letta source: archival memory passages, core memory blocks and (optionally) message history."""

from __future__ import annotations

import argparse
from typing import Any, Dict, Iterator, List, Mapping, Optional

from ..common import secret_from, to_jsonable
from ..model import ImporterError, MissingDependency, Record
from .base import Source

_KINDS = ("passages", "blocks", "messages")


def _text_of(content: Any) -> Optional[str]:
    """Text of a Letta message `content`: a string, or a list of typed parts."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        text = getattr(part, "text", None)
        if text is None and isinstance(part, dict):
            text = part.get("text")
        if text:
            parts.append(text)
    return "\n".join(parts) if parts else None


class LettaSource(Source):
    name = "letta"
    summary = ("Letta (cloud or self-hosted): per agent, archival memory passages, core memory blocks and "
               "optionally message history (env LETTA_API_KEY)")

    def __init__(self, api_key: Optional[str], base_url: Optional[str], kinds: List[str], agent_ids: List[str],
                 with_vectors: bool, page_size: int):
        bad = [k for k in kinds if k not in _KINDS]
        if bad:
            raise ImporterError(f"unknown --kinds {bad}; choose from {list(_KINDS)}")
        self.api_key, self.base_url, self.kinds, self.agent_ids = api_key, base_url, kinds, agent_ids
        self.with_vectors, self.page_size = with_vectors, page_size
        self._client = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--base-url", default=None, help="Letta server URL (default: Letta Cloud; "
                            "e.g. http://localhost:8283 for a self-hosted server)")
        parser.add_argument("--kinds", default="passages,blocks",
                            help="comma-separated subset of passages, blocks, messages (default passages,blocks)")
        parser.add_argument("--agent-id", action="append", default=[], help="only this agent (repeatable)")
        parser.add_argument("--with-vectors", action="store_true",
                            help="carry the passage embeddings the server returns (they must match the WARM dimension)")
        parser.add_argument("--page-size", type=int, default=100)

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        key = secret_from(None, env, "LETTA_API_KEY")
        return cls(key, args.base_url, [k.strip() for k in args.kinds.split(",") if k.strip()], args.agent_id,
                   args.with_vectors, args.page_size)

    def describe(self) -> Dict[str, Any]:
        return {"base_url": self.base_url or "letta-cloud", "kinds": self.kinds, "agents": sorted(self.agent_ids),
                "with_vectors": self.with_vectors}

    def _api(self):
        if self._client is None:
            try:
                from letta_client import Letta
            except ImportError:
                raise MissingDependency("letta", "letta-client", "import-letta") from None
            kwargs: Dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = Letta(**kwargs)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _agents(self) -> Iterator[Any]:
        client = self._api()
        if self.agent_ids:
            for aid in sorted(self.agent_ids):
                yield client.agents.retrieve(aid)
            return
        yield from client.agents.list(limit=self.page_size, ascending=True)

    def records(self, cursor: Any = None) -> Iterator[Record]:
        client = self._api()
        for agent in self._agents():
            ns = agent.id
            base = {"agent_id": agent.id, "agent_name": getattr(agent, "name", None)}
            if "blocks" in self.kinds:
                for b in client.agents.blocks.list(agent.id, limit=self.page_size, order="asc", order_by="created_at"):
                    yield self._block(b, base, ns)
            if "passages" in self.kinds:
                after: Optional[str] = None
                while True:
                    kwargs: Dict[str, Any] = dict(limit=self.page_size, ascending=True)
                    if after:
                        kwargs["after"] = after
                    page = list(client.agents.passages.list(agent.id, **kwargs))
                    for p in page:
                        yield self._passage(p, base, ns)
                    if len(page) < self.page_size:
                        break
                    after = page[-1].id
            if "messages" in self.kinds:
                for m in client.agents.messages.list(agent.id, limit=self.page_size, order="asc", order_by="created_at"):
                    yield self._message(m, base, ns)

    def _passage(self, p: Any, base: Dict[str, Any], ns: str) -> Record:
        meta = dict(base, kind="archival_passage", tags=p.tags, archive_id=p.archive_id, file_id=p.file_id,
                    file_name=p.file_name, source_id=p.source_id)
        if p.metadata:
            meta["letta_metadata"] = p.metadata
        vec = p.embedding if self.with_vectors and p.embedding else None
        return Record(id=f"passage:{p.id}", text=p.text, vector=vec, metadata=meta, namespace=ns,
                      created_at=p.created_at, updated_at=p.updated_at)

    @staticmethod
    def _block(b: Any, base: Dict[str, Any], ns: str) -> Record:
        meta = dict(base, kind="core_memory_block", label=b.label, description=b.description, limit=b.limit,
                    read_only=b.read_only, tags=b.tags, block_id=b.id)
        if b.metadata:
            meta["letta_metadata"] = b.metadata
        return Record(id=f"block:{ns}:{b.id}", text=b.value, metadata=meta, namespace=ns)

    @staticmethod
    def _message(m: Any, base: Dict[str, Any], ns: str) -> Record:
        mtype = getattr(m, "message_type", None) or "message"
        text = _text_of(getattr(m, "content", None))
        if text is None:  # reasoning, tool calls and returns carry their payload in other fields
            for attr in ("reasoning", "tool_return", "tool_call", "tool_calls", "summary"):
                val = getattr(m, attr, None)
                if val:
                    text = val if isinstance(val, str) else str(to_jsonable(val))
                    break
        meta = dict(base, kind="message", message_type=mtype, otid=getattr(m, "otid", None),
                    name=getattr(m, "name", None), run_id=getattr(m, "run_id", None))
        return Record(id=f"message:{m.id}", text=text, metadata=meta, namespace=ns, created_at=getattr(m, "date", None))
