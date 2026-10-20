# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Zep source: thread messages and graph memory (episodes, facts, entities) via `zep-cloud`."""

from __future__ import annotations

import argparse
from typing import Any, Dict, Iterator, List, Mapping, Optional

from ..common import secret_from, to_jsonable
from ..model import ImporterError, MissingDependency, Record
from .base import Source

_KINDS = ("messages", "episodes", "facts", "entities")



class ZepSource(Source):
    name = "zep"
    summary = ("Zep (cloud or self-hosted API): thread messages, graph episodes, facts (edges) and entities "
               "(nodes) of every user and standalone graph (env ZEP_API_KEY)")

    def __init__(self, api_key: Optional[str], base_url: Optional[str], kinds: List[str], page_size: int):
        bad = [k for k in kinds if k not in _KINDS]
        if bad:
            raise ImporterError(f"unknown --kinds {bad}; choose from {list(_KINDS)}")
        if not api_key:
            raise ImporterError("Zep needs an API key: set ZEP_API_KEY ")
        self.api_key, self.base_url, self.kinds, self.page_size = api_key, base_url, kinds, page_size
        self._client = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--base-url", default=None, help="Zep API base URL (default: Zep Cloud)")
        parser.add_argument("--kinds", default=",".join(_KINDS),
                            help=f"comma-separated subset of {', '.join(_KINDS)} (default all)")
        parser.add_argument("--page-size", type=int, default=100)

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        key = secret_from(None, env, "ZEP_API_KEY")
        return cls(key, args.base_url, [k.strip() for k in args.kinds.split(",") if k.strip()], args.page_size)

    def describe(self) -> Dict[str, Any]:
        return {"base_url": self.base_url or "zep-cloud", "kinds": self.kinds}

    def _api(self):
        if self._client is None:
            try:
                from zep_cloud import Zep
            except ImportError:
                raise MissingDependency("zep", "zep-cloud", "import-zep") from None
            kwargs: Dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = Zep(**kwargs)
        return self._client

    def close(self) -> None:
        self._client = None

    def count(self) -> Optional[int]:
        return None

    # -- enumeration ----------------------------------------------------------------

    def records(self, cursor: Any = None) -> Iterator[Record]:
        zep = self._api()
        users: Dict[str, str] = {}  # user uuid -> user_id (the namespace)
        user_graphs: List[Any] = []
        for u in zep.user.list(page_size=self.page_size, order_by="created_at", order="asc"):
            users[u.uuid_] = u.user_id or u.uuid_
            if u.graph_uuid:
                user_graphs.append((u.graph_uuid, u.user_id or u.uuid_))
        if "messages" in self.kinds:
            yield from self._threads(zep, users)
        graph_kinds = [k for k in self.kinds if k != "messages"]
        if graph_kinds:
            graphs = list(user_graphs)
            for g in zep.graph.list(page_size=self.page_size, order_by="created_at", order="asc"):
                if g.uuid_ not in {gu for gu, _ in graphs}:
                    graphs.append((g.uuid_, f"graph:{g.graph_id or g.uuid_}"))
            for graph_uuid, ns in sorted(graphs):
                yield from self._graph(zep, graph_uuid, ns, graph_kinds)

    def _threads(self, zep: Any, users: Dict[str, str]) -> Iterator[Record]:
        for th in zep.thread.list(page_size=self.page_size, order_by="created_at", order="asc"):
            thread_id = th.thread_id or th.uuid_
            ns = users.get(th.user_uuid or "", "")
            for m in zep.thread.list_messages(th.uuid_, page_size=self.page_size, order_by="created_at", order="asc"):
                meta: Dict[str, Any] = {
                    "kind": "message", "thread_id": thread_id, "thread_uuid": th.uuid_, "role": m.role,
                    "name": m.name, "processed": m.processed, "reference_time": m.reference_time,
                }
                if m.metadata:
                    meta["zep_metadata"] = m.metadata
                if m.content is None:
                    yield Record(id=f"message:{m.uuid_}", error=f"message {m.uuid_} has no content")
                    continue
                yield Record(id=f"message:{m.uuid_}", text=m.content, metadata=meta, namespace=ns,
                             created_at=m.created_at or m.reference_time)

    def _graph(self, zep: Any, graph_uuid: str, ns: str, kinds: List[str]) -> Iterator[Record]:
        order = dict(page_size=self.page_size, order_by="created_at", order="asc")
        if "episodes" in kinds:
            for e in zep.graph.episode.list(graph_uuid, **order):
                if e.thread_uuid:
                    continue  # a thread message: already imported from its thread, same content
                meta = {"kind": "episode", "source": e.source, "source_description": e.source_description,
                        "role": e.role, "role_name": e.role_name, "reference_time": e.reference_time,
                        "document_id": e.document_id, "processed": e.processed, "graph_uuid": graph_uuid}
                if e.metadata:
                    meta["zep_metadata"] = e.metadata
                if e.content is None:
                    yield Record(id=f"episode:{e.uuid_}", error=f"episode {e.uuid_} has no content")
                    continue
                yield Record(id=f"episode:{e.uuid_}", text=e.content, metadata=meta, namespace=ns,
                             created_at=e.created_at)
        if "facts" in kinds:
            for f in zep.graph.edge.list(graph_uuid, **order):
                meta = {
                    "kind": "fact", "relation": f.name, "source_node_uuid": f.source_node_uuid,
                    "source_node_name": f.source_node_name, "target_node_uuid": f.target_node_uuid,
                    "target_node_name": f.target_node_name, "valid_at": f.valid_at, "invalid_at": f.invalid_at,
                    "expired_at": f.expired_at, "episode_uuids": f.episode_uuids, "graph_uuid": graph_uuid,
                }
                if f.attributes:
                    meta["attributes"] = f.attributes
                if not f.fact:
                    yield Record(id=f"fact:{f.uuid_}", error=f"edge {f.uuid_} has no fact text")
                    continue
                yield Record(id=f"fact:{f.uuid_}", text=f.fact, metadata=meta, namespace=ns,
                             created_at=f.created_at, updated_at=f.expired_at)
        if "entities" in kinds:
            for n in zep.graph.node.list(graph_uuid, **order):
                text = ": ".join(p for p in (n.name, n.summary) if p)
                meta = {"kind": "entity", "name": n.name, "labels": n.labels, "graph_uuid": graph_uuid}
                if n.attributes:
                    meta["attributes"] = n.attributes
                if not text:
                    yield Record(id=f"entity:{n.uuid_}", error=f"node {n.uuid_} has no name or summary")
                    continue
                yield Record(id=f"entity:{n.uuid_}", text=text, metadata=to_jsonable(meta), namespace=ns,
                             created_at=n.created_at)
