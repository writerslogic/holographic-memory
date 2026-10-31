import gc
import json
import math
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from writerslogic_warm_sdk import WarmClient
from writerslogic_warm_sdk import client as client_module
from writerslogic_warm_sdk._transport import KeepAliveTransport


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # Set per test: close the socket after each response without saying so, as a server
    # dropping idle keep-alive connections does.
    drop_after_response = False

    def setup(self):
        super().setup()
        with self.server.connection_lock:
            self.server.open_connections += 1
            self.server.no_connections.clear()

    def finish(self):
        try:
            super().finish()
        finally:
            with self.server.connection_lock:
                self.server.open_connections -= 1
                if self.server.open_connections == 0:
                    self.server.no_connections.set()

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.seen.append((self.path, self.headers.get("Authorization"), body))
        if self.path == "/slow":
            time.sleep(0.6)
        reply = json.dumps({"matches": [], "port": self.client_address[1]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        try:
            self.wfile.write(reply)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
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
        srv.connection_lock = threading.Lock()
        srv.open_connections = 0
        srv.no_connections = threading.Event()
        srv.no_connections.set()
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
    with WarmClient(url, api_key="k") as c:
        assert isinstance(c.client._transport, KeepAliveTransport)
        ports = {c.query([0.5, 0.25], top_k=2)["port"] for _ in range(5)}
    assert len(ports) == 1, "requests should share one connection"
    path, auth, body = srv.seen[0]
    assert (path, auth, json.loads(body)) == ("/api/v1/query", "Bearer k", {"query_vector": [0.5, 0.25], "top_k": 2})


def test_proxy_environment_keeps_httpx_transport(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    with WarmClient("http://127.0.0.1:1") as c:
        assert not isinstance(c.client._transport, KeepAliveTransport)


def test_reconnects_after_server_drops_idle_connection(server):
    srv, url = server(drop_after_response=True)
    with WarmClient(url) as c:
        for _ in range(3):
            assert c.query([1.0], top_k=1)["matches"] == []
            time.sleep(0.05)  # let the close reach the idle connection
    assert len(srv.seen) == 3


def test_connection_errors_raise_httpx_errors():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with WarmClient(f"http://127.0.0.1:{port}") as c:
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


def test_transport_rejects_https_before_sending_plaintext(server):
    srv, url = server()
    with httpx.Client(transport=KeepAliveTransport()) as client:
        with pytest.raises(httpx.UnsupportedProtocol):
            client.post(url.replace("http:", "https:", 1), content=b"private body")
    assert srv.seen == []


@pytest.mark.parametrize("reuse", [False, True])
def test_transport_read_timeout_is_not_a_connect_timeout_or_the_connect_budget(server, reuse):
    srv, url = server()
    timeout = httpx.Timeout(2.0, read=0.05)
    with httpx.Client(transport=KeepAliveTransport(), timeout=timeout) as client:
        if reuse:
            client.post(url + "/ready", content=b"ready").raise_for_status()
        with pytest.raises(httpx.ReadTimeout):
            client.post(url + "/slow", content=b"already sent")
    assert [path for path, _, _ in srv.seen] == (["/ready", "/slow"] if reuse else ["/slow"])


def test_transport_body_producer_failure_does_not_poison_the_connection(server):
    srv, url = server()

    def failed_body():
        yield b"prefix"
        raise RuntimeError("producer failed")

    with httpx.Client(transport=KeepAliveTransport()) as client:
        with pytest.raises(RuntimeError, match="producer failed"):
            client.post(url + "/failed", content=failed_body())
        response = client.post(url + "/healthy", content=b"valid")
        response.raise_for_status()
    assert [path for path, _, _ in srv.seen] == ["/healthy"]


def test_transport_blocked_write_raises_write_timeout_without_retrying():
    accepted = threading.Event()
    release = threading.Event()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(5)

        def hold_connection():
            with listener.accept()[0] as connection:
                connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
                accepted.set()
                release.wait(5)

        thread = threading.Thread(target=hold_connection, daemon=True)
        thread.start()
        try:
            timeout = httpx.Timeout(2.0, write=0.05)
            with httpx.Client(transport=KeepAliveTransport(), timeout=timeout) as client:
                with pytest.raises(httpx.WriteTimeout):
                    client.post(f"http://127.0.0.1:{listener.getsockname()[1]}/write", content=b"x" * (32 * 1024 * 1024))
            assert accepted.is_set()
            listener.settimeout(0.05)
            try:
                extra, _ = listener.accept()
            except socket.timeout:
                pass
            else:
                extra.close()
                pytest.fail("the failed write was retried on another connection")
        finally:
            release.set()
            thread.join(5)
            assert not thread.is_alive()


def test_transport_can_disable_only_the_read_timeout(server):
    srv, url = server()
    timeout = httpx.Timeout(0.05, read=None)
    with httpx.Client(transport=KeepAliveTransport(), timeout=timeout) as client:
        client.post(url + "/slow", content=b"valid").raise_for_status()
    assert len(srv.seen) == 1


def test_transport_releases_connections_when_worker_threads_exit(server):
    srv, url = server()
    errors = []
    with httpx.Client(transport=KeepAliveTransport()) as client:
        def send():
            try:
                client.post(url, content=b"valid").raise_for_status()
            except Exception as exc:
                errors.append(exc)

        workers = [threading.Thread(target=send) for _ in range(8)]
        for worker in workers:
            worker.start()
            worker.join(5)
            assert not worker.is_alive()
        assert errors == []
        assert len(srv.seen) == len(workers)
        gc.collect()
        assert srv.no_connections.wait(2), "exited workers retained open keep-alive sockets"
