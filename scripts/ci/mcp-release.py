# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate MCP release assets and write the checksum tables packed by both launchers."""

import hashlib
import json
import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]
TRIPLES = (
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "x86_64-pc-windows-msvc",
)


def version() -> str:
    npm = json.loads((ROOT / "integrations/mcp/npm/package.json").read_text())["version"]
    pyproject = tomllib.loads((ROOT / "integrations/mcp/pypi/pyproject.toml").read_text())
    pypi = pyproject["project"]["version"]
    source = (ROOT / "integrations/mcp/pypi/src/holographic_memory_mcp/__init__.py").read_text()
    match = re.search(r'^__version__ = "([^"]+)"$', source, re.MULTILINE)
    if not match or not (npm == pypi == match.group(1)):
        raise ValueError(f"launcher versions disagree: npm={npm}, PyPI={pypi}, source={match.group(1) if match else 'missing'}")
    return npm


def checksums(assets: pathlib.Path) -> dict[str, str]:
    expected = {f"hms-mcp-{triple}{'.exe' if 'windows' in triple else ''}": triple for triple in TRIPLES}
    actual = {entry.name for entry in assets.iterdir() if entry.is_file()}
    if actual != set(expected):
        raise ValueError(f"MCP assets differ: missing={sorted(set(expected) - actual)}, unexpected={sorted(actual - set(expected))}")
    table = {}
    for name, triple in expected.items():
        with (assets / name).open("rb") as binary:
            table[triple] = hashlib.file_digest(binary, "sha256").hexdigest()
    return table


def main() -> None:
    release_version = version()
    tag = sys.argv[1]
    if tag != f"mcp-v{release_version}":
        raise ValueError(f"expected tag mcp-v{release_version}, got {tag}")
    if len(sys.argv) == 2:
        return
    if len(sys.argv) != 3:
        raise ValueError("usage: mcp-release.py mcp-vX.Y.Z [assets-directory]")
    table = {"assets": checksums(pathlib.Path(sys.argv[2]))}
    encoded = json.dumps(table, indent=2) + "\n"
    for target in (
        ROOT / "integrations/mcp/npm/checksums.json",
        ROOT / "integrations/mcp/pypi/src/holographic_memory_mcp/checksums.json",
    ):
        target.write_text(encoded)


if __name__ == "__main__":
    main()
