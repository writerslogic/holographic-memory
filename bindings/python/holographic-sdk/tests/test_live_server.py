"""Runs the SDK against a real hms-server process.

Set HMS_SERVER_BIN to the path of a binary built with
`cargo build --features server --bin hms-server`; the module is skipped otherwise.
"""

import asyncio
import math
import os
import random
import subprocess

import httpx
import pytest

from holographic_sdk import AsyncHolographicClient, HolographicClient

BIN = os.environ.get("HMS_SERVER_BIN")
pytestmark = pytest.mark.skipif(not BIN, reason="HMS_SERVER_BIN not set")

DIM = 24
API_KEY = "live-test-key"
SALT = b"0123456789abcdef"


def embedding(seed: int):
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(DIM)]


def noisy(vec, seed: int, amount=0.02):
    rng = random.Random(seed)
    return [v + rng.uniform(-amount, amount) for v in vec]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    data = tmp_path_factory.mktemp("hms-data")
    env = {**os.environ, "HMS_API_KEY": API_KEY}
    proc = subprocess.Popen(
        [BIN, "--bind", "127.0.0.1:0", "--input-dim", str(DIM), "--dim", "4096", "--data-dir", str(data)],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        line = proc.stdout.readline().strip()
        assert line.startswith("hms-server listening on "), line
        yield f"http://{line.rsplit(' ', 1)[1]}"
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def docs(prefix: str):
    return [
        {"id": f"{prefix}{i}", "text": f"document {i}", "vector": embedding(i), "metadata": {"n": i, "even": i % 2 == 0}}
        for i in range(6)
    ]


def test_sync_client_end_to_end(server):
    with HolographicClient(url=server, api_key=API_KEY, tenant_id="sync") as client:
        client.add_documents(docs("s")[:1])
        client.add_documents(docs("s")[1:])

        for i in range(6):
            res = client.query(noisy(embedding(i), 100 + i), top_k=3)
            assert res["matches"][0]["id"] == f"s{i}"
            assert res["matches"][0]["text"] == f"document {i}"
            assert res["matches"][0]["metadata"]["n"] == i
            scores = [m["score"] for m in res["matches"]]
            assert scores == sorted(scores, reverse=True) and len(scores) == 3

        res = client.query(embedding(1), top_k=6, filter={"even": True})
        assert {m["id"] for m in res["matches"]} == {"s0", "s2", "s4"}

        client.delete_document("s2")
        res = client.query(embedding(2), top_k=6)
        assert "s2" not in {m["id"] for m in res["matches"]}
        with pytest.raises(httpx.HTTPStatusError) as err:
            client.delete_document("s2")
        assert err.value.response.status_code == 404


def test_ids_needing_escaping_can_be_deleted(server):
    with HolographicClient(url=server, api_key=API_KEY, tenant_id="escape") as client:
        doc_id = "folder/doc #1?x=y%20z"
        client.add_documents([{"id": doc_id, "vector": embedding(7)}])
        client.delete_document(doc_id)
        assert client.query(embedding(7))["matches"] == []


def test_async_client_end_to_end(server):
    async def run():
        async with AsyncHolographicClient(url=server, api_key=API_KEY, tenant_id="async") as client:
            await client.add_documents(docs("a"))
            for i in range(6):
                res = await client.query(noisy(embedding(i), 200 + i), top_k=2)
                assert res["matches"][0]["id"] == f"a{i}"
            await client.delete_document("a3")
            res = await client.query(embedding(3), top_k=6)
            assert "a3" not in {m["id"] for m in res["matches"]}

    asyncio.run(run())


def test_masked_vectors_still_rank_nearest_first(server):
    kwargs = dict(url=server, api_key=API_KEY, tenant_id="masked", zero_trust_key="pass", mask_salt=SALT)
    with HolographicClient(**kwargs) as client:
        client.add_documents(docs("m"))
        for i in range(6):
            res = client.query(noisy(embedding(i), 300 + i), top_k=3)
            assert res["matches"][0]["id"] == f"m{i}"


def test_tenants_are_isolated_and_auth_required(server):
    with HolographicClient(url=server, api_key=API_KEY, tenant_id="one") as one:
        one.add_documents(docs("t"))
    with HolographicClient(url=server, api_key=API_KEY, tenant_id="two") as two:
        assert two.query(embedding(0))["matches"] == []
    with HolographicClient(url=server, tenant_id="one") as anonymous:
        with pytest.raises(httpx.HTTPStatusError) as err:
            anonymous.query(embedding(0))
        assert err.value.response.status_code == 401
    with HolographicClient(url=server, api_key=API_KEY, tenant_id="one") as client:
        with pytest.raises(httpx.HTTPStatusError) as err:
            client.add_documents([{"id": "bad", "vector": [1.0, 2.0]}])
        assert err.value.response.status_code == 422
        assert not math.isnan(client.query(embedding(0))["matches"][0]["score"])
