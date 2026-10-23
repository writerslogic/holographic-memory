import json
import math
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from holographic_sdk import HolographicClient
from holographic_sdk import client as client_module
from holographic_sdk._transport import KeepAliveTransport


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # Set per test: close the socket after each response without saying so, as a server
    # dropping idle keep-alive connections does.
    drop_after_response = False

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.seen.append((self.path, self.headers.get("Authorization"), body))
        reply = json.dumps({"matches": [], "port": self.client_address[1]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)
        if self.drop_after_response:
            self.close_connection = True

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    def start(drop_after_response=False):
        handler = type("Handler", (_Handler,), {"drop_after_response": drop_after_response})
        srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        srv.seen = []
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return srv, f"http://127.0.0.1:{srv.server_address[1]}"

    servers = []
    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def test_plain_http_url_uses_keep_alive_transport(server, monkeypatch):
    for var in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    srv, url = server()
    with HolographicClient(url, api_key="k") as c:
        assert isinstance(c.client._transport, KeepAliveTransport)
        ports = {c.query([0.5, 0.25], top_k=2)["port"] for _ in range(5)}
    assert len(ports) == 1, "requests should share one connection"
    path, auth, body = srv.seen[0]
    assert (path, auth, json.loads(body)) == ("/api/v1/query", "Bearer k", {"query_vector": [0.5, 0.25], "top_k": 2})


def test_proxy_environment_keeps_httpx_transport(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    with HolographicClient("http://127.0.0.1:1") as c:
        assert not isinstance(c.client._transport, KeepAliveTransport)


def test_reconnects_after_server_drops_idle_connection(server):
    srv, url = server(drop_after_response=True)
    with HolographicClient(url) as c:
        for _ in range(3):
            assert c.query([1.0], top_k=1)["matches"] == []
            time.sleep(0.05)  # let the close reach the idle connection
    assert len(srv.seen) == 3


def test_connection_errors_raise_httpx_errors():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with HolographicClient(f"http://127.0.0.1:{port}") as c:
        with pytest.raises(httpx.ConnectError):
            c.query([1.0])


@pytest.mark.parametrize("use_orjson", [True, False])
def test_json_body_rejects_non_finite_numbers(monkeypatch, use_orjson):
    if use_orjson:
        pytest.importorskip("orjson")
    else:
        monkeypatch.setattr(client_module, "orjson", None)
    assert json.loads(client_module._dumps({"v": [1.5, -2.0], "m": None})) == {"v": [1.5, -2.0], "m": None}
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError):
            client_module._dumps({"query_vector": [0.0, bad]})
