# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import time
import unittest

import httpx

from adapter import FACT_BYTES, FACT_WORDS, add_facts, create_app, search_input, split_text, tenant_for


class ContractTests(unittest.TestCase):
    def test_fact_boundaries_preserve_utf8_and_whitespace(self) -> None:
        for size in (FACT_BYTES - 1, FACT_BYTES, FACT_BYTES + 1):
            original = "é" * (size // 2) + ("a" if size % 2 else "")
            pieces = split_text(original)
            self.assertEqual("".join(pieces), original)
            self.assertTrue(all(len(piece.encode()) <= FACT_BYTES for piece in pieces))
        original = "word  " * (FACT_WORDS + 1)
        pieces = split_text(original)
        self.assertEqual("".join(pieces), original)
        self.assertTrue(all(len(piece.split()) <= FACT_WORDS for piece in pieces))

    def test_add_validates_fields_and_stable_ids(self) -> None:
        payload = {"request_id": "r1", "user_id": "u1", "session_id": "s1", "messages": [{"role": "user", "content": "A  B", "timestamp": 1704067200000}]}
        parsed = add_facts(payload)
        self.assertEqual(parsed, add_facts(payload))
        self.assertEqual(parsed[3][0]["text"], "user: A  B")
        self.assertNotEqual(parsed[3][0]["id"], add_facts(payload | {"user_id": "u2"})[3][0]["id"])
        for bad in (payload | {"request_id": ""}, payload | {"messages": []}, payload | {"messages": [{"role": "system", "content": "x"}]}, payload | {"messages": [{"role": "user", "content": "x", "timestamp": True}]}):
            with self.assertRaises(ValueError):
                add_facts(bad)

    def test_search_top_k_and_query_boundaries(self) -> None:
        for size in (8191, 8192):
            self.assertEqual(search_input({"query": "x" * size, "user_id": "u", "top_k": 100})[2], 100)
        for bad in ({"query": "x" * 8193, "user_id": "u", "top_k": 100}, {"query": "x", "user_id": "u", "top_k": 101}, {"query": "x", "user_id": "u", "top_k": True}):
            with self.assertRaises(ValueError):
                search_input(bad)
        self.assertNotEqual(tenant_for("user-a"), tenant_for("user-b"))


class LiveServerTests(unittest.TestCase):
    def test_auth_idempotence_and_user_isolation(self) -> None:
        binary = os.environ.get("HMS_BIN")
        if not binary:
            self.skipTest("set HMS_BIN to run exact-S hms-server integration")
        with tempfile.TemporaryDirectory() as data:
            process = subprocess.Popen([binary, "--no-auth", "--bind", "127.0.0.1:18897", "--data-dir", data,
                                        "--max-top-k", "100", "--max-tenants", "4", "--log-level", "error"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                for _ in range(100):
                    try:
                        if httpx.get("http://127.0.0.1:18897/readyz", timeout=1).status_code == 200:
                            break
                    except httpx.RequestError:
                        pass
                    time.sleep(0.1)
                else:
                    self.fail("hms-server did not become ready")

                async def exercise() -> None:
                    app = create_app("http://127.0.0.1:18897", "secret")
                    transport = httpx.ASGITransport(app=app)
                    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                        add = {"request_id": "r", "user_id": "alice", "session_id": "s", "messages": [{"role": "user", "content": "The sapphire key is in the attic."}]}
                        self.assertEqual((await client.post("/add", json=add)).status_code, 401)
                        headers = {"Authorization": "Bearer secret"}
                        for _ in range(2):
                            response = await client.post("/add", json=add, headers=headers)
                            self.assertEqual(response.status_code, 200, response.text)
                            self.assertEqual(response.json(), {"success": True, "request_id": "r", "user_id": "alice", "session_id": "s"})
                        response = await client.post("/search", json={"query": "sapphire key", "user_id": "alice", "top_k": 100}, headers=headers)
                        self.assertEqual(response.status_code, 200, response.text)
                        self.assertEqual(len(response.json()["data"]), 1)
                        self.assertIn("sapphire key", response.json()["data"][0]["content"])
                        other = await client.post("/search", json={"query": "sapphire key", "user_id": "bob", "top_k": 100}, headers=headers)
                        self.assertEqual(other.json(), {"data": []})
                    await app.router.on_shutdown[0]()

                asyncio.run(exercise())
            finally:
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
