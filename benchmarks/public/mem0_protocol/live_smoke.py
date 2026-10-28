# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the configured adapter with synthetic facts against a real server."""

import argparse
import asyncio
import hashlib
import json
import subprocess
from pathlib import Path

from aiohttp import ClientSession, web

from shim import Shim, build_app, load_config


async def smoke(binary: Path, config_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(str(config_path))
    with (output / "server.log").open("wb") as log:
        server = subprocess.Popen(
            [str(binary), "--no-auth", "--bind", "127.0.0.1:0", "--data-dir", str(output / "data"),
             "--input-dim", str(config.get("embedding", {}).get("dimensions", 384))],
            stdout=subprocess.PIPE, stderr=log,
        )
        runner = None
        try:
            line = server.stdout.readline().decode().strip()
            if "listening on " not in line:
                raise RuntimeError("server did not announce its listening address")
            base = "http://" + line.split("listening on ", 1)[1]
            shim = Shim(base, config, str(output / "embedding_calls.jsonl"))
            runner = web.AppRunner(build_app(shim))
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
            async with ClientSession() as client:
                ids = []
                for text in (
                    "Mira: I commute on a bicycle every weekday.",
                    "Mira: My favorite dessert is chocolate cake.",
                    "Mira: I adopted a terrier named Pebble.",
                ):
                    async with client.post(url + "/memories", json={
                        "messages": [{"role": "user", "content": text}],
                        "user_id": "synthetic-smoke", "timestamp": 1704067200,
                    }) as response:
                        assert response.status == 200, f"add HTTP {response.status}"
                        body = await response.json()
                        assert len(body["results"]) == 1
                        assert body["results"][0]["created_at"] == "2024-01-01T00:00:00"
                        ids.append(body["results"][0]["id"])
                async with client.post(url + "/memories", json={
                    "messages": [{"role": "user", "content": "Mira: I commute on a bicycle every weekday."}],
                    "user_id": "synthetic-smoke", "timestamp": 1704067200,
                }) as response:
                    assert response.status == 200, f"retry HTTP {response.status}"
                    assert (await response.json())["results"][0]["id"] == ids[0]
                async with client.get(base + "/api/v1/stats", headers={"X-Tenant-ID": "synthetic-smoke"}) as response:
                    assert response.status == 200
                    assert (await response.json())["documents"] == 3
                searches = []
                for query in ("How does Mira travel to work?", "What transport gets her to the office?"):
                    async with client.post(url + "/search", json={
                        "query": query, "user_id": "synthetic-smoke", "limit": 3,
                    }) as response:
                        assert response.status == 200, f"search HTTP {response.status}"
                        body = await response.json()
                        searches.append({"query": query, "response": body})
                (output / "searches.json").write_text(json.dumps(searches, indent=2) + "\n")
                assert all(len(row["response"]["results"]) == 3 for row in searches)
                assert "bicycle" in searches[1]["response"]["results"][0]["memory"]
                async with client.post(url + "/search", json={
                    "query": "How does Mira travel to work?", "user_id": "synthetic-other", "limit": 1,
                }) as response:
                    assert response.status == 200, f"isolated search HTTP {response.status}"
                    assert (await response.json())["results"] == []
            receipt = {
                "passed": True, "synthetic": True,
                "checks": ["add", "timestamp", "retry-identity", "semantic-recall", "tenant-isolation"],
                "server_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "shim_sha256": hashlib.sha256(Path(__file__).with_name("shim.py").read_bytes()).hexdigest(),
            }
            (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps(receipt))
        finally:
            if runner is not None:
                await runner.cleanup()
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(smoke(args.binary, args.config, args.output))
