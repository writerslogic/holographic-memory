# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = ["jsonschema>=4.23,<5"]
# ///

"""Validate actual HTTP memory requests/replies against the generated OpenAPI schemas."""

import argparse
import json
import os
import subprocess
import tempfile
import urllib.request

from jsonschema import Draft202012Validator


def check(binary):
    with tempfile.TemporaryDirectory(prefix="hms-memory-schema-") as directory:
        process = subprocess.Popen(
            [binary, "--bind", "127.0.0.1:0", "--data-dir", directory, "--dim", "1024", "--input-dim", "2"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            env={**os.environ, "HMS_API_KEY": "memory-schema-test-key"},
        )
        try:
            line = process.stdout.readline().strip()
            assert line.startswith("hms-server listening on "), "server did not announce its listener"
            base = "http://" + line.rsplit(" ", 1)[1]

            def request(path, payload=None, method="POST"):
                body = None if method == "GET" else (b"" if payload is None else json.dumps(payload).encode())
                req = urllib.request.Request(base + path, body, method=method, headers={
                    "Authorization": "Bearer memory-schema-test-key", "Content-Type": "application/json",
                })
                with urllib.request.urlopen(req, timeout=30) as response:
                    return json.load(response)

            spec = request("/openapi.json", method="GET")

            def validator(schema):
                document = {"components": spec["components"], **schema}
                Draft202012Validator.check_schema(document)
                return Draft202012Validator(document)

            def operation(name):
                return spec["paths"]["/api/v1/memory/" + name]["post"]

            cases = [
                ("remember", {"id": "a", "text": "alpha memory"}),
                ("remember", {"facts": [{"id": "b", "text": "beta memory", "subject": "topic",
                                          "entities": ["e"], "metadata": {"role": "test"}}]}),
                ("recall", {"query": "memory", "history": "all", "explain": True, "k": 10}),
                ("explain", {"query": "alpha", "id": "a", "history": "all"}),
                ("get", {"id": "a"}),
                ("get", {"id": "missing"}),
                ("history", {"id": "a"}),
                ("forget", {"scope": {"kind": "ids", "ids": ["b"]}, "dryRun": True}),
                ("applyForgetting", {}),
                ("consolidate", {}),
                ("applyForgetting", None),
                ("consolidate", None),
            ]
            for name, payload in cases:
                op = operation(name)
                body = op["requestBody"]
                if payload is None:
                    assert body["required"] is False
                else:
                    validator(body["content"]["application/json"]["schema"]).validate(payload)
                reply = request("/api/v1/memory/" + name, payload)
                response_validator = validator(op["responses"]["200"]["content"]["application/json"]["schema"])
                response_validator.validate(reply)
                assert not response_validator.is_valid({}), f"{name} reply lost its typed envelope"
                if name in ("applyForgetting", "consolidate"):
                    assert reply["dryRun"] is True
                    assert body["content"]["application/json"]["schema"]["properties"]["dryRun"]["default"] is True

            invalid = [
                ("remember", {}),
                ("remember", {"facts": []}),
                ("remember", {"facts": [], "text": "mixed request"}),
                ("recall", {"k": 1}),
                ("recall", {"query": "memory", "k": 0}),
                ("recall", {"query": "memory", "typo": True}),
                ("explain", {"query": "memory"}),
                ("forget", {"scope": {"kind": "subject"}}),
            ]
            for name, payload in invalid:
                schema = operation(name)["requestBody"]["content"]["application/json"]["schema"]
                assert not validator(schema).is_valid(payload), f"{name} accepted a malformed schema fixture"
            print(f"validated {len(cases)} actual memory request/reply pairs and {len(invalid)} invalid shapes")
        finally:
            process.terminate()
            process.wait(timeout=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-binary", default=os.environ.get("HMS_SERVER_BIN"))
    arguments = parser.parse_args()
    if not arguments.server_binary:
        parser.error("pass --server-binary or set HMS_SERVER_BIN")
    check(os.path.abspath(arguments.server_binary))
