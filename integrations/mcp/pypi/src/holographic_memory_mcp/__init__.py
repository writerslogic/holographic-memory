# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Launcher for the hms-mcp binary.

Finds, or downloads and verifies, the prebuilt binary for this platform, then runs it with stdio
inherited so the MCP stdio transport passes straight through. Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

__version__ = "0.1.3"

TARGETS = {
    ("darwin", "arm64"): "aarch64-apple-darwin",
    ("darwin", "x86_64"): "x86_64-apple-darwin",
    ("linux", "x86_64"): "x86_64-unknown-linux-gnu",
    ("linux", "aarch64"): "aarch64-unknown-linux-gnu",
    ("windows", "amd64"): "x86_64-pc-windows-msvc",
    ("windows", "x86_64"): "x86_64-pc-windows-msvc",
}
MAX_BYTES = 256 * 1024 * 1024
_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


def _fail(message: str) -> "NoReturn":  # noqa: F821
    sys.stderr.write(f"holographic-memory-mcp: {message}\n")
    raise SystemExit(1)


def _triple() -> str:
    key = (platform.system().lower(), platform.machine().lower())
    triple = TARGETS.get(key)
    if triple is None:
        _fail(
            f"no prebuilt hms-mcp for {key[0]}-{key[1]}; build it with cargo "
            "(--features mcp) and set HMS_MCP_BINARY"
        )
    return triple


def _cache_dir() -> Path:
    override = os.environ.get("HMS_MCP_CACHE_DIR")
    if override:
        return Path(override)
    home = Path.home()
    system = platform.system().lower()
    if system == "darwin":
        return home / "Library" / "Caches" / "holographic-memory-mcp"
    if system == "windows":
        base = os.environ.get("LOCALAPPDATA") or str(home / "AppData" / "Local")
        return Path(base) / "holographic-memory-mcp"
    return Path(os.environ.get("XDG_CACHE_HOME") or home / ".cache") / "holographic-memory-mcp"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_checksum(triple: str) -> str:
    override = os.environ.get("HMS_MCP_CHECKSUMS")
    table_path = Path(override) if override else Path(__file__).with_name("checksums.json")
    try:
        table = json.loads(table_path.read_text())
    except (OSError, ValueError):
        _fail(
            f"no checksum table at {table_path}; this package was not built for release. "
            "Set HMS_MCP_BINARY to a binary you trust."
        )
    digest = (table.get("assets") or {}).get(triple, "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        _fail(f"checksum table has no valid entry for {triple}")
    return digest


def _download(url: str, destination: Path) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in _LOOPBACK):
        raise RuntimeError("refusing a non-HTTPS download from a non-loopback host")
    request = urllib.request.Request(url, headers={"User-Agent": f"holographic-memory-mcp/{__version__}"})
    size = 0
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as out:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            size += len(block)
            if size > MAX_BYTES:
                raise RuntimeError("download exceeds the size limit")
            out.write(block)


def ensure_binary() -> str:
    override = os.environ.get("HMS_MCP_BINARY")
    if override:
        return override
    triple = _triple()
    exe = ".exe" if platform.system().lower() == "windows" else ""
    directory = _cache_dir() / __version__ / triple
    binary = directory / f"hms-mcp{exe}"
    expected = _expected_checksum(triple)
    if binary.exists() and _sha256(binary) == expected:
        return str(binary)
    base = os.environ.get("HMS_MCP_RELEASE_BASE") or (
        f"https://github.com/writerslogic/holographic-memory/releases/download/mcp-v{__version__}"
    )
    url = f"{base.rstrip('/')}/hms-mcp-{triple}{exe}"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    sys.stderr.write(f"holographic-memory-mcp: downloading {url}\n")
    handle, partial_name = tempfile.mkstemp(prefix=".download-", dir=directory)
    os.close(handle)
    partial = Path(partial_name)
    try:
        _download(url, partial)
        actual = _sha256(partial)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch: expected {expected}, got {actual}")
        partial.chmod(partial.stat().st_mode | stat.S_IXUSR)
        os.replace(partial, binary)
    except Exception as error:  # noqa: BLE001 - report any failure once, then exit
        partial.unlink(missing_ok=True)
        _fail(f"download failed: {error}")
    return str(binary)


def main() -> None:
    binary = ensure_binary()
    arguments = [binary, *sys.argv[1:]]
    if os.name == "nt":
        raise SystemExit(subprocess.call(arguments))
    os.execv(binary, arguments)
