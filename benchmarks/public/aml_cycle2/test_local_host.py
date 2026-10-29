# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

from contextlib import suppress
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import httpx

from adapter import tenant_for
from local_host import private_data_dir, private_token, verified_binary


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class LocalHostTests(unittest.TestCase):
    def test_private_token_and_data_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            token_file = root / "token.env"
            token_file.write_text("AML_TOKEN=synthetic-private-token\n")
            token_file.chmod(0o600)
            self.assertEqual(private_token(token_file), "synthetic-private-token")
            token_file.chmod(0o644)
            with self.assertRaises(ValueError):
                private_token(token_file)
            token_file.chmod(0o600)
            link = root / "link.env"
            link.symlink_to(token_file)
            with self.assertRaises(OSError):
                private_token(link)
            data = private_data_dir(root / "stores")
            self.assertEqual(data.stat().st_mode & 0o077, 0)
            data.chmod(0o755)
            with self.assertRaises(ValueError):
                private_data_dir(data)

    def test_pinned_binary_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "binary"
            binary.write_bytes(b"#!/bin/sh\nexit 0\n")
            binary.chmod(0o700)
            digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            self.assertEqual(verified_binary(binary, digest), binary)
            with self.assertRaises(ValueError):
                verified_binary(binary, "0" * 64)

    def test_add_reopen_search_and_auth(self) -> None:
        self._exercise_lifecycle(source_provenance=False)

    def test_provenance_persists_and_retries_after_reopen(self) -> None:
        self._exercise_lifecycle(source_provenance=True)

    def _exercise_lifecycle(self, *, source_provenance: bool) -> None:
        native = os.environ.get("HMS_QUERY_TEST_BINARY")
        if not native:
            self.skipTest("set HMS_QUERY_TEST_BINARY to run native lifecycle")
        binary = Path(native)
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            token_file = root / "token.env"
            token_file.write_text("AML_TOKEN=synthetic-private-token\n")
            token_file.chmod(0o600)
            stores = root / "stores"
            native_port, adapter_port = free_port(), free_port()
            self.assertNotEqual(native_port, adapter_port)
            command = [
                sys.executable, str(Path(__file__).with_name("local_host.py")),
                "--binary", str(binary), "--binary-sha256", digest,
                "--data-dir", str(stores), "--token-file", str(token_file),
                "--native-port", str(native_port), "--adapter-port", str(adapter_port),
            ]
            if source_provenance:
                command.append("--source-provenance")
            base = f"http://127.0.0.1:{adapter_port}"
            headers = {"Authorization": "Bearer synthetic-private-token"}
            add = {
                "request_id": "r1", "user_id": "alice", "session_id": "s1",
                "messages": [{"role": "user", "content": "The sapphire key is in the attic."}],
            }
            search = {"query": "sapphire key", "user_id": "alice", "top_k": 10}

            def launch() -> subprocess.Popen:
                process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                for _ in range(150):
                    if process.poll() is not None:
                        self.fail("local host exited before readiness")
                    with suppress(httpx.RequestError):
                        if httpx.get(base + "/healthz", timeout=1).status_code == 200:
                            return process
                    time.sleep(0.2)
                process.terminate()
                raise AssertionError("local host did not become ready")

            def assert_native_metadata() -> None:
                response = httpx.post(f"http://127.0.0.1:{native_port}/api/v1/memory/recall",
                    json={"query": "sapphire key", "k": 10},
                    headers={"X-Tenant-ID": tenant_for("alice")}, timeout=30)
                self.assertEqual(response.status_code, 200, response.text)
                rows = response.json()["results"]
                self.assertEqual(len(rows), 1)
                metadata = rows[0]["metadata"]
                if source_provenance:
                    self.assertEqual({key: metadata[key] for key in (
                        "aml_request_id", "aml_message_index", "aml_part_index")},
                        {"aml_request_id": "r1", "aml_message_index": 0, "aml_part_index": 0})
                else:
                    self.assertNotIn("aml_request_id", metadata)

            process = launch()
            try:
                self.assertEqual(httpx.post(base + "/add", json=add).status_code, 401)
                response = httpx.post(base + "/add", json=add, headers=headers, timeout=30)
                self.assertEqual(response.status_code, 200, response.text)
                response = httpx.post(base + "/search", json=search, headers=headers, timeout=30)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(len(response.json()["data"]), 1)
                assert_native_metadata()
            finally:
                process.terminate()
                self.assertEqual(process.wait(timeout=45), 0)

            process = launch()
            try:
                response = httpx.post(base + "/add", json=add, headers=headers, timeout=30)
                self.assertEqual(response.status_code, 200, response.text)
                response = httpx.post(base + "/search", json=search, headers=headers, timeout=30)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(len(response.json()["data"]), 1)
                self.assertIn("sapphire key", response.json()["data"][0]["content"])
                assert_native_metadata()
                other = httpx.post(base + "/search", json=search | {"user_id": "bob"}, headers=headers, timeout=30)
                self.assertEqual(other.json(), {"data": []})
            finally:
                process.terminate()
                self.assertEqual(process.wait(timeout=45), 0)


if __name__ == "__main__":
    unittest.main()
