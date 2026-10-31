# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""mem0 sources: the open-source library (`mem0ai`) and the hosted platform API."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from ..common import register_secret, secret_from
from ..model import ImporterError, MissingDependency, Record
from .base import Source

_SECRET_KEYS = ("key", "token", "password", "secret", "dsn", "url")


def _register_config_secrets(node: Any) -> None:
    """Register credential-looking values anywhere in a mem0 config so errors never echo them."""
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str) and any(s in str(k).lower() for s in _SECRET_KEYS[:4]):
                register_secret(v)
            else:
                _register_config_secrets(v)
    elif isinstance(node, list):
        for v in node:
            _register_config_secrets(v)


# Memories can carry several scopes; each belongs to exactly one importing scope, chosen by
# this priority, so a memory returned by two scope queries is emitted once.
_PRIORITY = ("user_id", "agent_id", "app_id", "run_id")
_NS_LABEL = {"user_id": "", "agent_id": "agent:", "app_id": "app:", "run_id": "run:"}


def _owner(mem: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    for key in _PRIORITY:
        value = mem.get(key)
        if key == "run_id" and not value:
            value = mem.get("session_id")  # the platform's v2 listing names the run session_id
        if value:
            return key, str(value)
    return None


def _namespace(owner: Optional[Tuple[str, str]]) -> str:
    return "" if owner is None else f"{_NS_LABEL[owner[0]]}{owner[1]}"


class Mem0PlatformSource(Source):
    name = "mem0-platform"
    summary = "mem0 platform (api.mem0.ai): every memory of every user/agent/app/run entity, paged (env MEM0_API_KEY)"

    def __init__(self, api_key: str, host: Optional[str], org_id: Optional[str], project_id: Optional[str],
                 scopes: List[Tuple[str, str]], page_size: int):
        if not api_key:
            raise ImporterError("mem0 platform needs an API key: set MEM0_API_KEY ")
        if not 1 <= page_size <= 200:
            raise ImporterError("--page-size must be 1..200 (the platform maximum)")
        self.api_key, self.host, self.org_id, self.project_id = api_key, host, org_id, project_id
        self.scopes, self.page_size = scopes, page_size
        self._client = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--host", default=None, help="API base URL (default https://api.mem0.ai)")
        parser.add_argument("--org-id")
        parser.add_argument("--project-id")
        for key in ("user", "agent", "app", "run"):
            parser.add_argument(f"--{key}-id", dest=f"scope_{key}", action="append", default=[],
                                help=f"only this {key} id (repeatable); default: enumerate every entity")
        parser.add_argument("--page-size", type=int, default=100, help="1..200")

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        key = secret_from(None, env, "MEM0_API_KEY")
        scopes = [(f"{k}_id", v) for k in ("user", "agent", "app", "run") for v in getattr(args, f"scope_{k}")]
        return cls(key or "", args.host, args.org_id, args.project_id, scopes, args.page_size)

    def describe(self) -> Dict[str, Any]:
        return {"host": self.host or "https://api.mem0.ai", "org_id": self.org_id, "project_id": self.project_id,
                "scopes": self.scopes}

    def _api(self):
        if self._client is None:
            os.environ.setdefault("MEM0_TELEMETRY", "false")
            try:
                from mem0 import MemoryClient
            except ImportError:
                raise MissingDependency("mem0-platform", "mem0ai", "import-mem0") from None
            try:
                self._client = MemoryClient(api_key=self.api_key, host=self.host, org_id=self.org_id,
                                            project_id=self.project_id)
            except ValueError as exc:
                raise ImporterError(f"mem0 platform rejected the credentials or host: {exc}") from None
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.client.close()
            self._client = None

    def _entities(self) -> List[Tuple[str, str]]:
        if self.scopes:
            return sorted(self.scopes)
        client = self._api()
        found = set()
        data = client.users()
        while True:
            for ent in data.get("results", []):
                kind = {"user": "user_id", "agent": "agent_id", "app": "app_id", "run": "run_id"}.get(ent.get("type"))
                if kind is None:
                    raise ImporterError(f"unknown mem0 entity type {ent.get('type')!r}")
                found.add((kind, ent["name"]))
            nxt = data.get("next")
            if not nxt:
                break
            resp = client.client.get(nxt)
            resp.raise_for_status()
            data = resp.json()
        return sorted(found)

    def records(self, cursor: Any = None) -> Iterator[Record]:
        client = self._api()
        for kind, name in self._entities():
            page = 1
            while True:
                data = client.get_all(version="v2", filters={kind: name}, page=page, page_size=self.page_size)
                items = data if isinstance(data, list) else data.get("results", [])
                for mem in items:
                    owner = _owner(mem)
                    if owner is not None and owner != (kind, name):
                        continue  # emitted under its higher-priority scope
                    yield self._record(mem, owner)
                nxt = None if isinstance(data, list) else data.get("next")
                if not items or not nxt:
                    break
                page += 1

    @staticmethod
    def _record(mem: Mapping[str, Any], owner: Optional[Tuple[str, str]]) -> Record:
        skip = {"id", "memory", "created_at", "updated_at"}
        meta = {k: v for k, v in mem.items() if k not in skip and k != "metadata"}
        user_meta = mem.get("metadata")
        if isinstance(user_meta, dict):
            for k, v in user_meta.items():
                meta.setdefault(k, v)
            meta["mem0_metadata"] = user_meta
        elif user_meta is not None:
            meta["mem0_metadata"] = user_meta
        meta["kind"] = "memory"
        if mem.get("id") is None or mem.get("memory") is None:
            return Record(id=str(mem.get("id", "?")), error="mem0 memory without id or text")
        return Record(id=str(mem["id"]), text=str(mem["memory"]), metadata=meta, namespace=_namespace(owner),
                      created_at=mem.get("created_at"), updated_at=mem.get("updated_at"))


class Mem0Source(Source):
    name = "mem0"
    summary = "mem0 open-source library: all memories in the configured vector store (qdrant, chroma; others by scope)"
    seekable = True

    def __init__(self, config: Dict[str, Any], scopes: List[Tuple[str, str]], with_vectors: bool,
                 page_size: int, scope_limit: int):
        self.config, self.scopes = config, scopes
        self.with_vectors, self.page_size, self.scope_limit = with_vectors, page_size, scope_limit
        self._vs = None
        # Scope-based listing has no cursor; only direct scans can resume mid-way.
        self.seekable = not scopes

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--config", required=True,
                            help="mem0 Memory.from_config dictionary as a JSON file path or inline JSON "
                                 "(the same config the application used; credentials in it are never logged)")
        for key in ("user", "agent", "run"):
            parser.add_argument(f"--{key}-id", dest=f"scope_{key}", action="append", default=[],
                                help=f"for vector stores that cannot be scanned: list this {key}'s memories (repeatable)")
        parser.add_argument("--with-vectors", action="store_true",
                            help="also read the stored embeddings (qdrant, chroma) so they can be carried")
        parser.add_argument("--page-size", type=int, default=256)
        parser.add_argument("--scope-limit", type=int, default=100000,
                            help="max memories per scope for scope-based listing; the import fails rather than truncating")

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        text = args.config
        if not text.lstrip().startswith("{"):
            try:
                with open(text, "r", encoding="utf-8") as fh:
                    text = fh.read()
            except OSError as exc:
                raise ImporterError(f"cannot read --config: {exc.strerror}") from None
        try:
            cfg = json.loads(text)
        except ValueError as exc:
            raise ImporterError(f"--config is not valid JSON: {exc}") from None
        _register_config_secrets(cfg)
        scopes = [(f"{k}_id", v) for k in ("user", "agent", "run") for v in getattr(args, f"scope_{k}")]
        return cls(cfg, scopes, args.with_vectors, args.page_size, args.scope_limit)

    def describe(self) -> Dict[str, Any]:
        vs = self.config.get("vector_store", {})
        cfg = vs.get("config", {}) if isinstance(vs, dict) else {}
        where = {k: cfg[k] for k in ("path", "collection_name", "host", "port") if k in cfg}
        return {"vector_store": vs.get("provider") if isinstance(vs, dict) else None, "where": where,
                "scopes": self.scopes, "with_vectors": self.with_vectors}

    def _store(self):
        """The configured vector store only. Building a full `Memory` would also create the LLM and
        embedder clients, a history database and a telemetry store under ~/.mem0 that can lock the
        application's own local store."""
        if self._vs is None:
            os.environ.setdefault("MEM0_TELEMETRY", "false")
            try:
                from mem0.configs.base import MemoryConfig
                from mem0.utils.factory import VectorStoreFactory
            except ImportError:
                raise MissingDependency("mem0", "mem0ai", "import-mem0") from None
            try:
                cfg = MemoryConfig(**json.loads(json.dumps(self.config)))
                self._vs = VectorStoreFactory.create(cfg.vector_store.provider, cfg.vector_store.config)
            except Exception as exc:  # config problems surface from several libraries
                raise ImporterError(f"cannot open the mem0 vector store from --config: {type(exc).__name__}") from None
        return self._vs

    def close(self) -> None:
        client = getattr(self._vs, "client", None)
        if hasattr(client, "close"):
            client.close()  # releases an embedded qdrant store's lock
        self._vs = None

    # -- scanning -------------------------------------------------------------------

    def _pages(self, vs: Any, cursor: Any) -> Iterator[Tuple[Any, List[Tuple[str, Dict[str, Any], Any]]]]:
        """Yield (page_start_offset, [(id, payload, vector)]) for the stores that can be scanned."""
        kind = type(vs).__module__.rsplit(".", 1)[-1]
        offset = cursor[0] if cursor else None
        if kind == "qdrant":
            while True:
                points, nxt = vs.client.scroll(collection_name=vs.collection_name, limit=self.page_size, offset=offset,
                                               with_payload=True, with_vectors=self.with_vectors)
                yield offset, [(str(p.id), dict(p.payload or {}), p.vector) for p in points]
                if nxt is None:
                    return
                offset = nxt
        elif kind == "chroma":
            offset = offset or 0
            include = ["metadatas"] + (["embeddings"] if self.with_vectors else [])
            while True:
                got = vs.collection.get(limit=self.page_size, offset=offset, include=include)
                ids = got.get("ids") or []
                metas = got.get("metadatas")
                metas = metas if metas is not None else [None] * len(ids)
                embs = got.get("embeddings")
                rows = [(str(i), dict(m or {}), None if embs is None else embs[n]) for n, (i, m) in enumerate(zip(ids, metas))]
                yield offset, rows
                if len(ids) < self.page_size:
                    return
                offset += len(ids)
        else:
            raise ImporterError(
                f"cannot scan a '{kind}' vector store; pass --user-id/--agent-id/--run-id to list by scope"
            )

    def records(self, cursor: Any = None) -> Iterator[Record]:
        vs = self._store()
        if self.scopes:
            yield from self._by_scope(vs)
            return
        skip = (cursor[1] + 1) if cursor else 0
        resume_page = True
        for start, rows in self._pages(vs, cursor):
            for idx, (pid, payload, vec) in enumerate(rows):
                if resume_page and idx < skip:
                    continue
                yield self._record(pid, payload, vec, [start, idx])
            resume_page = False

    def _by_scope(self, vs: Any) -> Iterator[Record]:
        for key, value in sorted(self.scopes):
            res = vs.list(filters={key: value}, limit=self.scope_limit)
            items = res[0] if isinstance(res, (tuple, list)) and res else res
            if len(items) >= self.scope_limit:
                raise ImporterError(f"{key}={value} has at least {self.scope_limit} memories; raise --scope-limit")
            for item in items:
                payload = dict(item.payload or {})
                owner = _owner(payload)
                if owner is not None and owner != (key, value):
                    continue  # emitted under its higher-priority scope
                yield self._record(str(item.id), payload, None, None)

    @staticmethod
    def _record(pid: str, payload: Dict[str, Any], vec: Any, cursor: Any) -> Record:
        if payload.get("data") is None:
            return Record(id=pid, error="mem0 payload without 'data'", cursor=cursor)
        meta = {k: v for k, v in payload.items() if k not in ("data", "created_at", "updated_at")}
        meta["kind"] = "memory"
        return Record(id=pid, text=str(payload["data"]), vector=vec, metadata=meta,
                      namespace=_namespace(_owner(payload)), created_at=payload.get("created_at"),
                      updated_at=payload.get("updated_at"), cursor=cursor)
