"""Keep-alive HTTP/1.1 transport on ``http.client`` for plain-HTTP service URLs.

httpx's default transport frames each request through httpcore and h11, which costs more per
call on loopback than the server spends answering a vector query. This transport keeps one
persistent ``http.client`` connection per thread and hands httpx a complete response, so the
``httpx.Client`` API, its errors and its response objects stay the same.
"""

import http.client
import os
import select
import socket
import threading
from typing import List, Optional

import httpx

_PROXY_VARS = ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")


def usable_for(url: str) -> bool:
    """True for a plain ``http`` URL with no proxy configured in the environment.

    Passing any transport to ``httpx.Client`` turns off its environment proxies, so a URL that
    a proxy variable could apply to keeps httpx's own transport.
    """
    return httpx.URL(url).scheme == "http" and not any(os.environ.get(v) for v in _PROXY_VARS)


def _timeout(request: httpx.Request) -> Optional[float]:
    timeout = request.extensions.get("timeout") or {}
    values = [v for v in (timeout.get("connect"), timeout.get("read"), timeout.get("write")) if v is not None]
    return max(values) if values else None


def _stale(conn: http.client.HTTPConnection) -> bool:
    """An idle keep-alive socket that is readable has been closed (or poisoned) by the peer."""
    sock = conn.sock
    if sock is None:
        return False
    try:
        readable, _, _ = select.select([sock], [], [], 0)
    except (OSError, ValueError):
        return True
    return bool(readable)


class KeepAliveTransport(httpx.BaseTransport):
    def __init__(self) -> None:
        self._local = threading.local()
        self._lock = threading.Lock()
        self._conns: List[http.client.HTTPConnection] = []

    def _connection(self, request: httpx.Request, timeout: Optional[float]) -> http.client.HTTPConnection:
        url = request.url
        key = (url.host, url.port)
        conn: Optional[http.client.HTTPConnection] = getattr(self._local, "conn", None)
        if conn is not None and getattr(self._local, "key", None) != key:
            conn.close()
            with self._lock:
                self._conns.remove(conn)
            conn = None
        elif conn is not None and _stale(conn):
            # http.client reopens a closed connection object on the next request.
            conn.close()
        if conn is None:
            conn = http.client.HTTPConnection(url.host, url.port or 80, timeout=timeout)
            self._local.conn, self._local.key = conn, key
            with self._lock:
                self._conns.append(conn)
        elif conn.sock is not None:
            conn.sock.settimeout(timeout)
        conn.timeout = timeout
        return conn

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        timeout = _timeout(request)
        conn = self._connection(request, timeout)
        connecting = conn.sock is None
        try:
            conn.putrequest(request.method, request.url.raw_path.decode("ascii"), skip_host=True, skip_accept_encoding=True)
            chunked = False
            for name, value in request.headers.raw:
                conn.putheader(name, value)
                chunked = chunked or (name.lower() == b"transfer-encoding" and value.lower() == b"chunked")
            body = request.read()
            if chunked:
                conn.endheaders([body] if body else [], encode_chunked=True)
            else:
                conn.endheaders(body or None)
            res = conn.getresponse()
            body = res.read()
        except socket.timeout as exc:
            conn.close()
            raise (httpx.ConnectTimeout if connecting else httpx.ReadTimeout)(str(exc), request=request) from exc
        except (http.client.HTTPException, ConnectionError) as exc:
            conn.close()
            if connecting and isinstance(exc, ConnectionRefusedError):
                raise httpx.ConnectError(str(exc), request=request) from exc
            raise httpx.RemoteProtocolError(str(exc) or type(exc).__name__, request=request) from exc
        except OSError as exc:
            conn.close()
            raise (httpx.ConnectError if connecting else httpx.ReadError)(str(exc), request=request) from exc
        if res.will_close:
            conn.close()
        return httpx.Response(
            res.status,
            headers=res.getheaders(),
            stream=httpx.ByteStream(body),
            request=request,
            extensions={"http_version": b"HTTP/1.1", "reason_phrase": res.reason.encode("ascii", "replace")},
        )

    def close(self) -> None:
        with self._lock:
            conns, self._conns = self._conns, []
        for conn in conns:
            conn.close()
