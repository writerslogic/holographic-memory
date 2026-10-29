# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["fastapi==0.162.0", "httpx==0.28.1", "uvicorn==0.54.0"]
# ///
"""Serve the AML adapter with a pinned native binary on a persistent local disk."""

from __future__ import annotations

from contextlib import suppress
import argparse
import asyncio
import hashlib
import os
import re
import signal
import stat
import subprocess
from pathlib import Path

import httpx
import uvicorn

from adapter import create_app


def private_token(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077 or metadata.st_size > 4096:
            raise ValueError("token file must be a private regular file of at most 4096 bytes")
        content = os.read(descriptor, 4097).decode()
    finally:
        os.close(descriptor)
    values = {}
    for line in content.splitlines():
        if line and not line.startswith("#"):
            name, separator, value = line.partition("=")
            if not separator or name in values:
                raise ValueError("token file has an invalid entry")
            values[name] = value.strip().strip('"').strip("'")
    token = values.get("AML_TOKEN")
    if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{16,1024}", token):
        raise ValueError("token file has no valid AML_TOKEN")
    return token


def private_data_dir(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError("data directory must not be a symlink")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.stat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
        raise ValueError("data directory must be owned by this user and private")
    return path


def verified_binary(path: Path, expected_sha256: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected binary SHA-256 must be lowercase hexadecimal")
    if path.is_symlink() or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("native binary must be an executable regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("native binary SHA-256 differs from the pinned build")
    return path


async def serve(args: argparse.Namespace) -> None:
    token = private_token(args.token_file)
    data_dir = private_data_dir(args.data_dir)
    binary = verified_binary(args.binary, args.binary_sha256)
    command = [
        str(binary), "--no-auth", "--bind", f"127.0.0.1:{args.native_port}",
        "--data-dir", str(data_dir), "--max-top-k", "100", "--max-tenants", "64",
        "--tenant-idle-seconds", "30", "--max-concurrent-work", "8",
        "--max-queue", "512", "--queue-timeout-ms", "600000",
        "--max-body-bytes", "67108864", "--log-level", "error",
    ]
    shutdown = asyncio.Event()
    server: uvicorn.Server | None = None

    def request_shutdown(_number: int, _frame: object) -> None:
        shutdown.set()
        if server is not None:
            server.should_exit = True

    previous_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, request_shutdown)
    native = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            for _ in range(300):
                if shutdown.is_set():
                    raise RuntimeError("shutdown requested during startup")
                if native.poll() is not None:
                    raise RuntimeError("native memory engine exited during startup")
                with suppress(httpx.RequestError):
                    response = await client.get(f"http://127.0.0.1:{args.native_port}/readyz")
                    if response.status_code == 200:
                        break
                await asyncio.sleep(0.2)
            else:
                raise RuntimeError("native memory engine did not become ready")
        app = create_app(f"http://127.0.0.1:{args.native_port}", token,
                         source_provenance=args.source_provenance)
        server = uvicorn.Server(uvicorn.Config(
            app, host="127.0.0.1", port=args.adapter_port,
            access_log=False, log_level="warning", timeout_graceful_shutdown=1800,
        ))

        async def monitor_native() -> None:
            await asyncio.to_thread(native.wait)
            server.should_exit = True

        monitor = asyncio.create_task(monitor_native())
        try:
            await server.serve()
            if not server.started:
                raise RuntimeError("AML adapter did not start")
            if native.poll() is not None:
                raise RuntimeError("native memory engine exited while serving")
        finally:
            if native.poll() is None:
                native.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(native.wait), timeout=45)
            except asyncio.TimeoutError:
                native.kill()
                await asyncio.to_thread(native.wait)
            await asyncio.wait_for(monitor, timeout=5)
    finally:
        if native.poll() is None:
            native.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(native.wait), timeout=45)
            except asyncio.TimeoutError:
                native.kill()
                await asyncio.to_thread(native.wait)
        signal.signal(signal.SIGTERM, previous_term)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--source-provenance", action="store_true",
                        help="store request/message/chunk provenance; use only with a fresh data directory")
    parser.add_argument("--native-port", type=int, default=8777)
    parser.add_argument("--adapter-port", type=int, default=8000)
    args = parser.parse_args()
    if not all(1024 <= port <= 65535 for port in (args.native_port, args.adapter_port)) or args.native_port == args.adapter_port:
        parser.error("native and adapter ports must be distinct integers from 1024 to 65535")
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
