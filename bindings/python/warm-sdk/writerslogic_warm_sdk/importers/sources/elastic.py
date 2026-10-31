# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Elasticsearch and OpenSearch sources: scroll over an index (or pattern) through the REST API."""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional

import httpx

from ..common import redact_uri, secret_from, uri_secret
from ..model import ImporterError, Record
from ._fields import (
    get_path,
    http_error,
    httpx_verify,
    numeric_vector,
    pick_text_field,
    text_value,
    without_paths,
)
from .base import Source, parse_cli_json

_VECTOR_TYPES = {"dense_vector", "knn_vector"}
_RETRY = {429, 502, 503, 504}


def _find_vector_fields(properties: Mapping[str, Any], prefix: str = "") -> List[str]:
    found: List[str] = []
    for name, spec in properties.items():
        path = f"{prefix}{name}"
        if not isinstance(spec, Mapping):
            continue
        if spec.get("type") in _VECTOR_TYPES:
            found.append(path)
        if isinstance(spec.get("properties"), Mapping):
            found.extend(_find_vector_fields(spec["properties"], path + "."))
    return found


class _SearchEngineSource(Source):
    """Shared implementation; the subclasses differ in credentials and defaults only."""

    seekable = False  # scroll contexts cannot be resumed; the pipeline re-reads and verifies the prefix
    flavour = ""
    url_envs: tuple = ()
    key_envs: tuple = ()
    user_envs: tuple = ()
    password_envs: tuple = ()

    def __init__(
        self,
        url: str,
        index: str,
        *,
        api_key: Optional[str] = None,
        bearer: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        vector_field: Optional[str] = None,
        text_field: Optional[str] = None,
        timestamp_field: Optional[str] = None,
        updated_field: Optional[str] = None,
        namespace_from: str = "auto",
        query: Optional[Dict[str, Any]] = None,
        page_size: int = 500,
        scroll: str = "5m",
        insecure: bool = False,
        ca_cert: Optional[str] = None,
        timeout: float = 60.0,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        if namespace_from not in ("auto", "index", "none"):
            raise ImporterError("--namespace-from must be auto, index or none")
        uri_secret(url)
        self.url = url.rstrip("/")
        self.index = index
        self.vector_field, self.text_field = vector_field, text_field
        self.timestamp_field, self.updated_field = timestamp_field, updated_field
        wildcard = any(c in index for c in "*,")
        self.namespace_from = ("index" if wildcard else "none") if namespace_from == "auto" else namespace_from
        self.query = query or {"match_all": {}}
        self.page_size, self.scroll = page_size, scroll
        headers = {"Content-Type": "application/json"}
        auth = None
        if api_key:
            headers["Authorization"] = f"ApiKey {api_key}"
        elif bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        elif user and password is not None:
            auth = httpx.BasicAuth(user, password)
        self._http = httpx.Client(
            headers=headers, auth=auth, timeout=timeout, verify=httpx_verify(insecure, ca_cert), transport=transport
        )
        self._auth_mode = "api_key" if api_key else "bearer" if bearer else "basic" if auth else "none"
        self._detected: Optional[str] = None

    # -- CLI ------------------------------------------------------------------------

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        env = lambda names: ", ".join("$" + n for n in names)  # noqa: E731
        parser.add_argument("--url", help=f"cluster URL (default: {env(cls.url_envs)})")
        parser.add_argument("--index", required=True, help="index name, comma list or wildcard pattern")
        parser.add_argument("--user", help=f"basic-auth user (default: {env(cls.user_envs)})")
        parser.add_argument("--password", help=f"discouraged: visible in the process list; prefer {env(cls.password_envs)}. "
                                               f"API key / bearer token come from {env(cls.key_envs + getattr(cls, 'bearer_envs', ()))}")
        parser.add_argument("--vector-field", help="dense_vector/knn_vector field (default: detected from the mapping)")
        parser.add_argument("--text-field", help="_source field holding the text (default: text, content, body, ...)")
        parser.add_argument("--timestamp-field", help="_source field holding the creation time")
        parser.add_argument("--updated-field", help="_source field holding the last-update time")
        parser.add_argument("--namespace-from", default="auto", choices=("auto", "index", "none"),
                            help="index: the index name becomes the namespace (auto: only for patterns and lists)")
        parser.add_argument("--query", help="JSON query DSL selecting what to import (default: match_all)")
        parser.add_argument("--page-size", type=int, default=500)
        parser.add_argument("--scroll", default="5m", help="scroll keep-alive")
        parser.add_argument("--insecure", action="store_true", help="skip TLS certificate verification")
        parser.add_argument("--ca-cert", help="CA bundle for TLS verification")

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        url = args.url or next((env[n] for n in cls.url_envs if env.get(n)), None)
        if not url:
            raise ImporterError(f"give --url or set {cls.url_envs[0]}")
        return cls(
            url,
            args.index,
            api_key=secret_from(None, env, *cls.key_envs),
            bearer=secret_from(None, env, *getattr(cls, "bearer_envs", ())),
            user=args.user or next((env[n] for n in cls.user_envs if env.get(n)), None),
            password=secret_from(args.password, env, *cls.password_envs),
            vector_field=args.vector_field,
            text_field=args.text_field,
            timestamp_field=args.timestamp_field,
            updated_field=args.updated_field,
            namespace_from=args.namespace_from,
            query=parse_cli_json(args.query, "--query"),
            page_size=args.page_size,
            scroll=args.scroll,
            insecure=args.insecure,
            ca_cert=args.ca_cert,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "flavour": self.flavour,
            "url": redact_uri(self.url),
            "index": self.index,
            "query": self.query,
            "vector_field": self.vector_field,
            "text_field": self.text_field,
            "namespace_from": self.namespace_from,
        }

    def close(self) -> None:
        self._http.close()

    # -- HTTP -----------------------------------------------------------------------

    def _request(self, method: str, path: str, **kw: Any) -> Any:
        delay = 0.5
        for attempt in range(4):
            try:
                res = self._http.request(method, f"{self.url}{path}", **kw)
                if res.status_code in _RETRY and attempt < 3:
                    time.sleep(delay)
                    delay *= 2
                    continue
                res.raise_for_status()
                return res.json() if res.content else {}
            except httpx.HTTPStatusError as exc:
                raise http_error(exc, f"{self.flavour} {method} {path.split('?')[0]}") from None
            except httpx.TransportError as exc:
                if attempt < 3:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise http_error(exc, f"cannot reach {self.flavour} at {redact_uri(self.url)}") from None
        raise AssertionError("unreachable")

    # -- reading --------------------------------------------------------------------

    def _vector_field(self) -> Optional[str]:
        if self.vector_field:
            return self.vector_field
        if self._detected is None:
            mapping = self._request("GET", f"/{self.index}/_mapping")
            found: List[str] = []
            for body in mapping.values():
                props = (body.get("mappings") or {}).get("properties") or {}
                for path in _find_vector_fields(props):
                    if path not in found:
                        found.append(path)
            if len(found) > 1:
                raise ImporterError(f"several vector fields in the mapping {found}; choose one with --vector-field")
            self._detected = found[0] if found else ""
        return self._detected or None

    def count(self) -> Optional[int]:
        try:
            return int(self._request("POST", f"/{self.index}/_count", json={"query": self.query})["count"])
        except (ImporterError, KeyError, ValueError):
            return None

    def records(self, cursor: Any = None) -> Iterator[Record]:
        vfield = self._vector_field()
        body: Dict[str, Any] = {"size": self.page_size, "query": self.query, "sort": ["_doc"], "_source": True}
        if vfield:
            body["fields"] = [vfield]  # vectors are excluded from _source by default in recent Elasticsearch
        page = self._request("POST", f"/{self.index}/_search?scroll={self.scroll}", json=body)
        scroll_id = page.get("_scroll_id")
        try:
            while True:
                hits = (page.get("hits") or {}).get("hits") or []
                if not hits:
                    break
                for hit in hits:
                    yield self._record(hit, vfield)
                scroll_id = page.get("_scroll_id", scroll_id)
                page = self._request("POST", "/_search/scroll", json={"scroll": self.scroll, "scroll_id": scroll_id})
        finally:
            if scroll_id:
                try:
                    self._http.request("DELETE", f"{self.url}/_search/scroll", json={"scroll_id": [scroll_id]})
                except httpx.HTTPError:
                    pass  # the context expires on its own

    def _record(self, hit: Mapping[str, Any], vfield: Optional[str]) -> Record:
        ident = str(hit.get("_id"))
        src = hit.get("_source")
        if not isinstance(src, Mapping):
            src = {}
        vec = None
        if vfield:
            vec = numeric_vector(get_path(src, vfield))
            if vec is None:
                vec = numeric_vector((hit.get("fields") or {}).get(vfield))
        text_key = pick_text_field(src, self.text_field)
        text = text_value(get_path(src, text_key)) if text_key else None
        drop = [p for p in (vfield, text_key, self.timestamp_field, self.updated_field) if p]
        meta = without_paths(src, drop)
        return Record(
            id=ident,
            text=text,
            vector=vec,
            metadata=meta,
            namespace=str(hit.get("_index", "")) if self.namespace_from == "index" else "",
            created_at=get_path(src, self.timestamp_field) if self.timestamp_field else None,
            updated_at=get_path(src, self.updated_field) if self.updated_field else None,
        )


class ElasticsearchSource(_SearchEngineSource):
    name = "elasticsearch"
    summary = "Elasticsearch index or pattern via the scroll API (REST; API key, bearer or basic auth)"
    flavour = "elasticsearch"
    url_envs = ("ELASTICSEARCH_URL", "ELASTIC_URL")
    key_envs = ("ELASTIC_API_KEY",)
    bearer_envs = ("ELASTIC_BEARER_TOKEN",)
    user_envs = ("ELASTIC_USER",)
    password_envs = ("ELASTIC_PASSWORD",)


class OpenSearchSource(_SearchEngineSource):
    name = "opensearch"
    summary = "OpenSearch index or pattern via the scroll API (REST; basic auth or API key; no AWS SigV4)"
    flavour = "opensearch"
    url_envs = ("OPENSEARCH_URL",)
    key_envs = ("OPENSEARCH_API_KEY",)
    user_envs = ("OPENSEARCH_USER",)
    password_envs = ("OPENSEARCH_PASSWORD",)


__all__ = ["ElasticsearchSource", "OpenSearchSource"]
