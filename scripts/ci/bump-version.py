# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Set the release version everywhere it must agree, then verify it.

usage: bump-version.py X.Y.Z[-pre]

Updates Cargo.toml, Cargo.lock (offline), package.json with its optionalDependencies, and
package-lock.json, then runs check-versions.py. The wheel version comes from Cargo.toml.
"""
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SEMVER = re.compile(r"^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$")


def dump(path: pathlib.Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def main(argv: list[str]) -> int:
    if len(argv) != 1 or not SEMVER.match(argv[0]):
        print(__doc__, file=sys.stderr)
        return 2
    version = argv[0]

    cargo = ROOT / "Cargo.toml"
    text = cargo.read_text()
    text, count = re.subn(r'(?m)^version = "[^"]+"$', f'version = "{version}"', text, count=1)
    if count != 1:
        print("no package version found in Cargo.toml", file=sys.stderr)
        return 1
    cargo.write_text(text)

    package_path = ROOT / "package.json"
    package = json.loads(package_path.read_text())
    package["version"] = version
    for name in package.get("optionalDependencies", {}):
        package["optionalDependencies"][name] = version
    dump(package_path, package)

    lock_path = ROOT / "package-lock.json"
    lock = json.loads(lock_path.read_text())
    lock["version"] = version
    root = lock["packages"][""]
    root["version"] = version
    for name in root.get("optionalDependencies", {}):
        root["optionalDependencies"][name] = version
        lock["packages"][f"node_modules/{name}"]["version"] = version
    dump(lock_path, lock)

    subprocess.run(["cargo", "update", "--workspace", "--offline"], cwd=ROOT, check=True)
    return subprocess.run([sys.executable, str(ROOT / "scripts/ci/check-versions.py")]).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
