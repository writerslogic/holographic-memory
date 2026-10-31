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
import weakref
from typing import Optional

import httpx

_PROXY_VARS = ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
_TIMEOUT_ERRORS = (httpx.ConnectTimeout, httpx.WriteTimeout, httpx.ReadTimeout)
_IO_ERRORS = (httpx.ConnectError, httpx.WriteError, httpx.ReadError)


def usable_for(url: str) -> bool:
    """True for a plain ``http`` URL with no proxy configured in the environment.

    Passing any transport to ``httpx.Client`` turns off its environment proxies, so a URL that
    a proxy variable could apply to keeps httpx's own transport.
    """
    return httpx.URL(url).scheme == "http" and not any(os.environ.get(v) for v in _PROXY_VARS)


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
        self._conns: weakref.WeakSet[http.client.HTTPConnection] = weakref.WeakSet()

    def _connection(self, request: httpx.Request, timeout: Optional[float]) -> http.client.HTTPConnection:
        url = request.url
        key = (url.host, url.port)
        conn: Optional[http.client.HTTPConnection] = getattr(self._local, "conn", None)
        if conn is not None and getattr(self._local, "key", None) != key:
            conn.close()
            with self._lock:
                self._conns.discard(conn)
            conn = None
        elif conn is not None and _stale(conn):
            # http.client reopens a closed connection object on the next request.
            conn.close()
        if conn is None:
            conn = http.client.HTTPConnection(url.host, url.port or 80, timeout=timeout)
            self._local.conn, self._local.key = conn, key
            with self._lock:
                self._conns.add(conn)
        conn.timeout = timeout
        return conn

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.scheme != "http":
            raise httpx.UnsupportedProtocol(
                "KeepAliveTransport supports only http URLs", request=request
            )
        body = request.read()
        timeouts = request.extensions.get("timeout") or {}
        conn = self._connection(request, timeouts.get("connect"))
        phase = 0
        try:
            if conn.sock is None:
                conn.connect()
            phase = 1
            conn.sock.settimeout(timeouts.get("write"))
            conn.putrequest(request.method, request.url.raw_path.decode("ascii"), skip_host=True, skip_accept_encoding=True)
            chunked = False
            for name, value in request.headers.raw:
                conn.putheader(name, value)
                chunked = chunked or (name.lower() == b"transfer-encoding" and value.lower() == b"chunked")
            if chunked:
                conn.endheaders([body] if body else [], encode_chunked=True)
            else:
                conn.endheaders(body or None)
            phase = 2
            conn.sock.settimeout(timeouts.get("read"))
            with conn.getresponse() as res:
                body = res.read()
        except socket.timeout as exc:
            conn.close()
            raise _TIMEOUT_ERRORS[phase](str(exc), request=request) from exc
        except http.client.HTTPException as exc:
            conn.close()
            raise httpx.RemoteProtocolError(str(exc) or type(exc).__name__, request=request) from exc
        except OSError as exc:
            conn.close()
            raise _IO_ERRORS[phase](str(exc), request=request) from exc
        except BaseException:
            conn.close()
            raise
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
            conns = list(self._conns)
            self._conns.clear()
        for conn in conns:
            conn.close()
