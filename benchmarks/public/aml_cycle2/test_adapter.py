# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import suppress
import asyncio
import json
import os
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx

from adapter import FACT_BYTES, FACT_WORDS, MAX_ID, add_facts, create_app, search_input, split_text, tenant_for


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
        for size in (8191, 8192, 8193):
            self.assertEqual(search_input({"query": "x" * size, "user_id": "u", "top_k": 100})[2], 100)
        for bad in ({"query": "x" * (1024 * 1024 + 1), "user_id": "u", "top_k": 100}, {"query": "x", "user_id": "u", "top_k": 101}, {"query": "x", "user_id": "u", "top_k": True}):
            with self.assertRaises(ValueError):
                search_input(bad)
        self.assertNotEqual(tenant_for("user-a"), tenant_for("user-b"))

    def test_provenance_preserves_legacy_ids_text_and_distinct_repeated_messages(self) -> None:
        for size in (FACT_BYTES - 1, FACT_BYTES, FACT_BYTES + 1):
            content = "é" * (size // 2) + ("a" if size % 2 else "")
            payload = {"request_id": "r", "user_id": "u", "session_id": "s", "messages": [
                {"role": "user", "content": content, "timestamp": 100},
                {"role": "assistant", "content": content, "timestamp": 100},
                {"role": "user", "content": content, "timestamp": 100},
            ]}
            legacy = add_facts(payload)[3]
            facts = add_facts(payload, source_provenance=True)[3]
            self.assertEqual([{k: v for k, v in fact.items() if k != "metadata"} for fact in facts], legacy)
            self.assertEqual(facts, add_facts(payload, source_provenance=True)[3])
            self.assertEqual(len({fact["id"] for fact in facts}), len(facts))
            for index, message in enumerate(payload["messages"]):
                parts = [fact for fact in facts if fact["metadata"]["aml_message_index"] == index]
                self.assertEqual("".join(fact["text"] for fact in parts), f"{message['role']}: {content}")
                self.assertEqual([fact["metadata"] for fact in parts], [
                    {"aml_request_id": "r", "aml_message_index": index, "aml_part_index": part}
                    for part in range(len(parts))
                ])

    def test_provenance_preserves_word_chunks_and_request_id_byte_boundaries(self) -> None:
        for words in (FACT_WORDS - 1, FACT_WORDS, FACT_WORDS + 1):
            content = "word  " * words
            for size in (MAX_ID - 1, MAX_ID, MAX_ID + 1):
                request = "é" * (size // 2) + ("a" if size % 2 else "")
                payload = {"request_id": request, "user_id": "u", "session_id": "s",
                           "messages": [{"role": "user", "content": content}]}
                if size > MAX_ID:
                    with self.assertRaises(ValueError):
                        add_facts(payload, source_provenance=True)
                    continue
                facts = add_facts(payload, source_provenance=True)[3]
                self.assertEqual("".join(fact["text"] for fact in facts), "user: " + content)
                self.assertTrue(all(len(fact["text"].split()) <= FACT_WORDS for fact in facts))
                self.assertTrue(all(fact["metadata"]["aml_request_id"] == request for fact in facts))

    def test_provenance_configuration_requires_explicit_booleans(self) -> None:
        for value in (None, 0, 1, "false", "true"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                add_facts({}, source_provenance=value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                create_app("http://native", "secret", source_provenance=value)

    def test_multimodal_caption_path_and_media_validation(self) -> None:
        payload = {"request_id": "r1", "user_id": "u1", "session_id": "s1", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "blue coat"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            {"type": "text", "text": "at the station"},
        ]}]}
        self.assertEqual(add_facts(payload)[3][0]["text"], "user: blue coat\nat the station")
        self.assertEqual(search_input({"query": [{"type": "text", "text": "coat"}], "user_id": "u1", "top_k": 100})[3], True)
        with self.assertRaises(ValueError):
            add_facts(payload | {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/pic.png"}}]}]})
        with self.assertRaises(ValueError):
            add_facts(payload | {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%"}}]}]})


class ProvenanceRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_add_forwards_opt_in_provenance_with_the_existing_tenant_and_text(self) -> None:
        real = httpx.AsyncClient
        for enabled in (False, True):
            requests = []

            async def native(request):
                requests.append(request)
                return httpx.Response(200, json={})

            with patch("adapter.httpx.AsyncClient", side_effect=lambda **kw: real(
                    transport=httpx.MockTransport(native), **kw)):
                app = create_app("http://native", "secret", source_provenance=enabled)
            async with app.router.lifespan_context(app), real(
                    transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                payload = {"request_id": "r", "user_id": "alice", "session_id": "s",
                           "messages": [{"role": "user", "content": "literal  text"}]}
                response = await client.post("/add", json=payload, headers={"Authorization": "Bearer secret"})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(len(requests), 1)
                self.assertEqual(requests[0].headers["X-Tenant-ID"], tenant_for("alice"))
                self.assertEqual(requests[0].url.path, "/api/v1/memory/remember")
                fact = json.loads(requests[0].content)["facts"][0]
                self.assertEqual(fact["text"], "user: literal  text")
                self.assertEqual("metadata" in fact, enabled)
                if enabled:
                    self.assertEqual(fact["metadata"], {
                        "aml_request_id": "r", "aml_message_index": 0, "aml_part_index": 0,
                    })


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
                    with suppress(httpx.RequestError):
                        if httpx.get("http://127.0.0.1:18897/readyz", timeout=1).status_code == 200:
                            break
                    time.sleep(0.1)
                else:
                    self.fail("hms-server did not become ready")

                async def exercise() -> None:
                    app = create_app("http://127.0.0.1:18897", "secret")
                    transport = httpx.ASGITransport(app=app)
                    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=transport, base_url="http://test") as client:
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

                    provenance = create_app("http://127.0.0.1:18897", "secret", source_provenance=True)
                    async with provenance.router.lifespan_context(provenance), httpx.AsyncClient(
                            transport=httpx.ASGITransport(app=provenance), base_url="http://test") as client:
                        conflict = await client.post("/add", json=add, headers=headers)
                        self.assertEqual(conflict.status_code, 409, conflict.text)
                        fresh = add | {"user_id": "provenance-user"}
                        for _ in range(2):
                            response = await client.post("/add", json=fresh, headers=headers)
                            self.assertEqual(response.status_code, 200, response.text)
                        async with httpx.AsyncClient(base_url="http://127.0.0.1:18897") as native:
                            response = await native.post("/api/v1/memory/recall",
                                json={"query": "sapphire key", "k": 100},
                                headers={"X-Tenant-ID": tenant_for("provenance-user")})
                            self.assertEqual(response.status_code, 200, response.text)
                            rows = response.json()["results"]
                            self.assertEqual(len(rows), 1)
                            self.assertEqual(rows[0]["text"], "user: The sapphire key is in the attic.")
                            self.assertEqual({key: rows[0]["metadata"][key] for key in (
                                "aml_request_id", "aml_message_index", "aml_part_index")},
                                {"aml_request_id": "r", "aml_message_index": 0, "aml_part_index": 0})

                asyncio.run(exercise())
            finally:
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
