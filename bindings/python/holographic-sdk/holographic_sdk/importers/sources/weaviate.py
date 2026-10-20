# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Weaviate source: cursor listing through the REST objects API, per class and tenant."""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import httpx

from ..common import redact_uri, secret_from, uri_secret
from ..model import ImporterError, Record
from ._fields import DEFAULT_TEXT_FIELDS, http_error, httpx_verify, numeric_vector, text_value
from .base import Source

_RETRY = {429, 502, 503, 504}


class WeaviateSource(Source):
    name = "weaviate"
    summary = "Weaviate classes and tenants via the REST objects cursor API (seekable; tenant becomes the namespace)"
    seekable = True

    def __init__(
        self,
        url: str,
        *,
        api_key: Optional[str] = None,
        classes: Optional[List[str]] = None,
        tenants: Optional[List[str]] = None,
        text_property: Optional[str] = None,
        vector_name: Optional[str] = None,
        page_size: int = 200,
        insecure: bool = False,
        ca_cert: Optional[str] = None,
        timeout: float = 60.0,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        uri_secret(url)
        self.url = url.rstrip("/")
        self.classes, self.tenants = sorted(classes or []), sorted(tenants or [])
        self.text_property, self.vector_name, self.page_size = text_property, vector_name, page_size
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(headers=headers, timeout=timeout, verify=httpx_verify(insecure, ca_cert),
                                  transport=transport)
        self._schema: Dict[str, Mapping[str, Any]] = {}

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--url", help="Weaviate HTTP URL (default: $WEAVIATE_URL)")
        parser.add_argument("--class", dest="classes", action="append", default=[],
                            help="class to import, repeatable (default: every class in the schema)")
        parser.add_argument("--tenant-name", dest="tenants", action="append", default=[],
                            help="tenant to import, repeatable (default: every tenant of multi-tenant classes)")
        parser.add_argument("--text-property", help="property holding the text (default: detected per class)")
        parser.add_argument("--vector-name", help="named vector to carry (default: the object's own vector, else the only named one)")
        parser.add_argument("--page-size", type=int, default=200)
        parser.add_argument("--insecure", action="store_true")
        parser.add_argument("--ca-cert")

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        url = args.url or env.get("WEAVIATE_URL")
        if not url:
            raise ImporterError("give --url or set WEAVIATE_URL")
        return cls(url, api_key=secret_from(None, env, "WEAVIATE_API_KEY"), classes=args.classes,
                   tenants=args.tenants, text_property=args.text_property, vector_name=args.vector_name,
                   page_size=args.page_size, insecure=args.insecure, ca_cert=args.ca_cert)

    def describe(self) -> Dict[str, Any]:
        return {"url": redact_uri(self.url), "classes": self.classes, "tenants": self.tenants,
                "text_property": self.text_property, "vector_name": self.vector_name}

    def close(self) -> None:
        self._http.close()

    def _get(self, path: str, **params: Any) -> Any:
        delay = 0.5
        for attempt in range(4):
            try:
                res = self._http.get(f"{self.url}{path}", params={k: v for k, v in params.items() if v is not None})
                if res.status_code in _RETRY and attempt < 3:
                    time.sleep(delay)
                    delay *= 2
                    continue
                res.raise_for_status()
                return res.json()
            except httpx.HTTPStatusError as exc:
                raise http_error(exc, f"weaviate GET {path}") from None
            except httpx.TransportError as exc:
                if attempt < 3:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise http_error(exc, f"cannot reach weaviate at {redact_uri(self.url)}") from None
        raise AssertionError("unreachable")

    # -- enumeration ----------------------------------------------------------------

    def _units(self) -> List[Tuple[str, Optional[str], Optional[str]]]:
        """(class, tenant, problem) units in a stable order; problem set for unreadable tenants."""
        schema = self._get("/v1/schema")
        classes = {c["class"]: c for c in schema.get("classes", [])}
        self._schema = classes
        wanted = self.classes or sorted(classes)
        units: List[Tuple[str, Optional[str], Optional[str]]] = []
        for cname in wanted:
            cdef = classes.get(cname)
            if cdef is None:
                raise ImporterError(f"class '{cname}' is not in the Weaviate schema")
            if (cdef.get("multiTenancyConfig") or {}).get("enabled"):
                tenants = self._get(f"/v1/schema/{cname}/tenants")
                for t in sorted(tenants, key=lambda t: t["name"]):
                    if self.tenants and t["name"] not in self.tenants:
                        continue
                    status = (t.get("activityStatus") or "HOT").upper()
                    problem = None if status in ("HOT", "ACTIVE") else f"tenant is {status}; activate it to read its objects"
                    units.append((cname, t["name"], problem))
            else:
                units.append((cname, None, None))
        return units

    def count(self) -> Optional[int]:
        return None

    def _text_property(self, cname: str) -> Optional[str]:
        if self.text_property:
            return self.text_property
        props = [p for p in (self._schema.get(cname) or {}).get("properties", [])
                 if any(str(t).lower() in ("text", "text[]") for t in p.get("dataType", []))]
        names = [p["name"] for p in props]
        for pref in DEFAULT_TEXT_FIELDS:
            if pref in names:
                return pref
        return names[0] if names else None

    def _vector(self, obj: Mapping[str, Any]) -> Optional[List[float]]:
        named = obj.get("vectors")
        if self.vector_name:
            vec = (named or {}).get(self.vector_name)
        else:
            vec = obj.get("vector")
            if vec is None and isinstance(named, Mapping) and len(named) == 1:
                vec = next(iter(named.values()))
        got = numeric_vector(vec)
        return list(got) if got is not None else None

    def records(self, cursor: Any = None) -> Iterator[Record]:
        units = self._units()
        resume = tuple(cursor[:2]) if cursor else None
        after = cursor[2] if cursor else None
        for cname, tenant, problem in units:
            if resume is not None and (cname, tenant or "") < (resume[0], resume[1] or ""):
                continue
            if problem:
                yield Record(id=f"{cname}/{tenant}", error=f"class {cname}: {problem}", namespace=tenant or "",
                             cursor=[cname, tenant, after if resume == (cname, tenant) else None])
                continue
            start = after if resume == (cname, tenant) else None
            tprop = self._text_property(cname)
            while True:
                page = self._get("/v1/objects", **{"class": cname, "limit": self.page_size, "after": start,
                                                    "include": "vector", "tenant": tenant})
                objs = page.get("objects") or []
                if not objs:
                    break
                for obj in objs:
                    props = dict(obj.get("properties") or {})
                    text = text_value(props.pop(tprop, None)) if tprop else None
                    props.setdefault("weaviate_class", cname)
                    yield Record(
                        id=str(obj["id"]),
                        text=text,
                        vector=self._vector(obj),
                        metadata=props,
                        namespace=tenant or "",
                        created_at=obj.get("creationTimeUnix"),
                        updated_at=obj.get("lastUpdateTimeUnix"),
                        cursor=[cname, tenant, obj["id"]],
                    )
                start = objs[-1]["id"]  # read until an empty page: a server may cap the page below `limit`


__all__ = ["WeaviateSource"]
