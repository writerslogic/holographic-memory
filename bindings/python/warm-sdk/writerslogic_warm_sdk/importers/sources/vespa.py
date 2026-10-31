# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Vespa source: the /document/v1 visit API with continuation tokens."""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import httpx

from ..common import redact_uri, secret_from, uri_secret
from ..model import ImporterError, Record
from ._fields import get_path, has_path, http_error, httpx_verify, pick_text_field, text_value, without_paths
from .base import Source

_RETRY = {429, 502, 503, 504}


def tensor_vector(value: Any) -> Tuple[Optional[List[float]], Optional[str]]:
    """(vector, problem) from a Vespa tensor in any JSON rendering of a one-dimensional dense tensor."""
    if isinstance(value, list):
        if value and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            return [float(v) for v in value], None
        return None, "not a flat list of numbers"
    if not isinstance(value, Mapping):
        return None, "not a tensor"
    if "values" in value:
        vals = value["values"]
        if isinstance(vals, str):  # int8 tensors render as a hex string
            try:
                raw = bytes.fromhex(vals)
            except ValueError:
                return None, "unreadable hex tensor"
            return [float(b - 256 if b > 127 else b) for b in raw], None
        return tensor_vector(vals)
    if "cells" in value:
        cells = value["cells"]
        if isinstance(cells, list):
            indexed: List[Tuple[int, float]] = []
            for cell in cells:
                address = cell.get("address") or {}
                if len(address) != 1:
                    return None, "tensor has more than one dimension"
                try:
                    indexed.append((int(next(iter(address.values()))), float(cell["value"])))
                except (ValueError, KeyError, TypeError):
                    return None, "tensor is mapped (sparse), not indexed"
            indexed.sort()
            if [i for i, _ in indexed] != list(range(len(indexed))):
                return None, "tensor cells are not a dense 0..n range"
            return [v for _, v in indexed], None
        if isinstance(cells, Mapping):
            try:
                pairs = sorted((int(k), float(v)) for k, v in cells.items())
            except (ValueError, TypeError):
                return None, "tensor is mapped (sparse), not indexed"
            if [i for i, _ in pairs] != list(range(len(pairs))):
                return None, "tensor cells are not a dense 0..n range"
            return [v for _, v in pairs], None
    if "blocks" in value:
        return None, "mixed tensors (blocks) are not vectors"
    return None, "not a tensor"


class VespaSource(Source):
    name = "vespa"
    summary = "Vespa document type via the /document/v1 visit API (continuation tokens; bearer token or mTLS)"
    seekable = False

    def __init__(
        self,
        url: str,
        namespace: str,
        doctype: str,
        *,
        token: Optional[str] = None,
        cert: Optional[str] = None,
        key: Optional[str] = None,
        cluster: Optional[str] = None,
        selection: Optional[str] = None,
        vector_field: Optional[str] = None,
        text_field: Optional[str] = None,
        timestamp_field: Optional[str] = None,
        updated_field: Optional[str] = None,
        namespace_from: str = "none",
        page_size: int = 100,
        insecure: bool = False,
        ca_cert: Optional[str] = None,
        timeout: float = 120.0,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        if namespace_from not in ("none", "namespace", "doctype", "group"):
            raise ImporterError("--namespace-from must be none, namespace, doctype or group")
        uri_secret(url)
        self.url = url.rstrip("/")
        self.vnamespace, self.doctype, self.cluster, self.selection = namespace, doctype, cluster, selection
        self.vector_field, self.text_field = vector_field, text_field
        self.timestamp_field, self.updated_field = timestamp_field, updated_field
        self.namespace_from, self.page_size = namespace_from, page_size
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._http = httpx.Client(headers=headers, timeout=timeout, verify=httpx_verify(insecure, ca_cert),
                                  cert=(cert, key) if cert and key else cert, transport=transport)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--url", help="Vespa container URL (default: $VESPA_URL)")
        parser.add_argument("--vespa-namespace", required=True, help="document namespace (the 'ns' in id:ns:type::user)")
        parser.add_argument("--doctype", required=True, help="document type")
        parser.add_argument("--token", help="discouraged: visible in the process list; prefer $VESPA_TOKEN (bearer token)")
        parser.add_argument("--cert", help="client certificate (mTLS, Vespa Cloud)")
        parser.add_argument("--key", help="client private key for --cert")
        parser.add_argument("--cluster", help="content cluster (sent as the cluster parameter)")
        parser.add_argument("--selection", help="document selection expression restricting the visit")
        parser.add_argument("--vector-field", help="tensor field to carry (default: the only tensor field)")
        parser.add_argument("--text-field", help="field holding the text (default: text, content, body, ...)")
        parser.add_argument("--timestamp-field")
        parser.add_argument("--updated-field")
        parser.add_argument("--namespace-from", default="none", choices=("none", "namespace", "doctype", "group"),
                            help="what becomes the WARM namespace (group: the g= locality of the document id)")
        parser.add_argument("--page-size", type=int, default=100, help="wantedDocumentCount per request")
        parser.add_argument("--insecure", action="store_true")
        parser.add_argument("--ca-cert")

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        url = args.url or env.get("VESPA_URL")
        if not url:
            raise ImporterError("give --url or set VESPA_URL")
        return cls(url, args.vespa_namespace, args.doctype, token=secret_from(args.token, env, "VESPA_TOKEN"),
                   cert=args.cert, key=args.key, cluster=args.cluster, selection=args.selection,
                   vector_field=args.vector_field, text_field=args.text_field,
                   timestamp_field=args.timestamp_field, updated_field=args.updated_field,
                   namespace_from=args.namespace_from, page_size=args.page_size, insecure=args.insecure,
                   ca_cert=args.ca_cert)

    def describe(self) -> Dict[str, Any]:
        return {"url": redact_uri(self.url), "namespace": self.vnamespace, "doctype": self.doctype,
                "cluster": self.cluster, "selection": self.selection, "vector_field": self.vector_field,
                "text_field": self.text_field, "namespace_from": self.namespace_from}

    def close(self) -> None:
        self._http.close()

    def _visit(self, continuation: Optional[str]) -> Dict[str, Any]:
        params: Dict[str, Any] = {"wantedDocumentCount": self.page_size, "format.tensors": "short"}
        if self.cluster:
            params["cluster"] = self.cluster
        if self.selection:
            params["selection"] = self.selection
        if continuation:
            params["continuation"] = continuation
        path = f"/document/v1/{self.vnamespace}/{self.doctype}/docid"
        delay = 0.5
        for attempt in range(4):
            try:
                res = self._http.get(f"{self.url}{path}", params=params)
                if res.status_code in _RETRY and attempt < 3:
                    time.sleep(delay)
                    delay *= 2
                    continue
                res.raise_for_status()
                return res.json()
            except httpx.HTTPStatusError as exc:
                raise http_error(exc, "vespa visit") from None
            except httpx.TransportError as exc:
                if attempt < 3:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise http_error(exc, f"cannot reach vespa at {redact_uri(self.url)}") from None
        raise AssertionError("unreachable")

    def _detect_vector_field(self, fields: Mapping[str, Any]) -> Optional[str]:
        cands = sorted(k for k, v in fields.items()
                       if isinstance(v, Mapping) and ("values" in v or "cells" in v or "blocks" in v))
        if len(cands) > 1:
            raise ImporterError(f"several tensor fields {cands}; choose one with --vector-field")
        return cands[0] if cands else ""

    def records(self, cursor: Any = None) -> Iterator[Record]:
        continuation: Optional[str] = None
        vfield = self.vector_field
        while True:
            page = self._visit(continuation)
            for doc in page.get("documents") or []:
                vfield = self._resolve_field(vfield, doc)
                yield self._record(doc, vfield)
            continuation = page.get("continuation")
            if not continuation:
                break

    def _resolve_field(self, vfield: Optional[str], doc: Mapping[str, Any]) -> Optional[str]:
        if vfield is None:
            vfield = self._detect_vector_field(doc.get("fields") or {})
            self.vector_field = vfield  # fixed for the rest of the run
        return vfield

    def _record(self, doc: Mapping[str, Any], vfield: Optional[str]) -> Record:
        did = str(doc.get("id", ""))
        parts = did.split(":", 4)
        user = parts[4] if len(parts) == 5 and parts[0] == "id" else did
        locality = parts[3] if len(parts) == 5 else ""
        fields = doc.get("fields") or {}
        problem = None
        vec = None
        if vfield and has_path(fields, vfield):
            vec, problem = tensor_vector(get_path(fields, vfield))
        text_key = pick_text_field(fields, self.text_field)
        text = text_value(get_path(fields, text_key)) if text_key else None
        meta = without_paths(fields, [p for p in (vfield, text_key, self.timestamp_field, self.updated_field) if p])
        meta.setdefault("vespa_namespace", self.vnamespace)
        meta.setdefault("vespa_doctype", self.doctype)
        group = None
        if locality:
            key, _, val = locality.partition("=")
            meta.setdefault("vespa_group" if key == "g" else "vespa_number", val)
            group = val if key == "g" else None
        ns = {"none": "", "namespace": self.vnamespace, "doctype": self.doctype, "group": group or ""}[self.namespace_from]
        return Record(
            id=user,
            text=text,
            vector=vec,
            metadata=meta,
            namespace=ns,
            created_at=get_path(fields, self.timestamp_field) if self.timestamp_field else None,
            updated_at=get_path(fields, self.updated_field) if self.updated_field else None,
            error=f"tensor field '{vfield}' unusable: {problem}" if problem else None,
        )


__all__ = ["VespaSource", "tensor_vector"]
