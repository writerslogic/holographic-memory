# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""A small stateful HTTP mock for cloud APIs, driven by the vendors' real client libraries.

Each service writes handlers that implement the documented wire behavior (paging, auth,
response shapes) over an in-memory dataset. Every request is logged; a request that
matches no route is answered 599 and recorded in `unmatched`, so tests can assert that
the importer called only documented endpoints (`server.assert_clean()`).

These are MOCKS: they prove the importer speaks the documented protocol, not that a
live service behaves identically. Tests using them are labeled `mock` in their names.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlsplit


@dataclass
class Request:
    method: str
    path: str
    query: Dict[str, List[str]]
    headers: Dict[str, str]
    body: Any
    raw: bytes
    match: "re.Match[str]"

    def q(self, name: str, default: Optional[str] = None) -> Optional[str]:
        vals = self.query.get(name)
        return vals[0] if vals else default


@dataclass
class Response:
    status: int = 200
    body: Any = None
    headers: Dict[str, str] = field(default_factory=dict)


Handler = Callable[[Request], Any]


class MockApi:
    """Register routes with `@api.route("GET", r"/v1/things/(?P<id>[^/]+)")`; a handler
    returns a JSON-able value, a `Response`, or raises `ApiError`."""

    def __init__(self, required_auth: Optional[Tuple[str, str]] = None):
        self.routes: List[Tuple[str, "re.Pattern[str]", Handler]] = []
        self.requests: List[Request] = []
        self.unmatched: List[str] = []
        self.required_auth = required_auth  # (header name, exact value), None to accept anything
        self.auth_failures = 0
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # -- registration ---------------------------------------------------------------

    def route(self, method: str, pattern: str):
        def deco(fn: Handler) -> Handler:
            self.routes.append((method.upper(), re.compile(pattern + r"$"), fn))
            return fn

        return deco

    # -- lifecycle ------------------------------------------------------------------

    @property
    def url(self) -> str:
        assert self._httpd is not None
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "MockApi":
        api = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:  # keep pytest output clean
                return

            def _serve(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                parts = urlsplit(self.path)
                headers = {k.lower(): v for k, v in self.headers.items()}
                body: Any = None
                if raw:
                    try:
                        body = json.loads(raw)
                    except ValueError:
                        body = raw
                status, out, extra = api._dispatch(self.command, parts.path, parse_qs(parts.query), headers, body, raw)
                payload = b"" if out is None else (out if isinstance(out, bytes) else json.dumps(out).encode())
                self.send_response(status)
                self.send_header("Content-Type", extra.pop("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(payload)))
                for k, v in extra.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(payload)

            do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = _serve

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def __enter__(self) -> "MockApi":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- dispatch -------------------------------------------------------------------

    def _dispatch(self, method, path, query, headers, body, raw):
        if self.required_auth is not None:
            name, value = self.required_auth
            if headers.get(name.lower()) != value:
                self.auth_failures += 1
                return 401, {"error": "unauthorized"}, {}
        for m, pat, fn in self.routes:
            hit = pat.match(path)
            if m == method and hit:
                req = Request(method, path, query, headers, body, raw, hit)
                self.requests.append(req)
                try:
                    out = fn(req)
                except ApiError as exc:
                    return exc.status, exc.body, {}
                if isinstance(out, Response):
                    return out.status, out.body, dict(out.headers)
                return 200, out, {}
        self.unmatched.append(f"{method} {path}")
        return 599, {"error": f"mock has no route for {method} {path}"}, {}

    def assert_clean(self) -> None:
        assert not self.unmatched, f"importer called undocumented endpoints: {self.unmatched}"
        assert not self.auth_failures, f"{self.auth_failures} request(s) lacked the required credential"


class ApiError(Exception):
    def __init__(self, status: int, body: Any = None):
        super().__init__(f"{status}")
        self.status = status
        self.body = body if body is not None else {"error": str(status)}
