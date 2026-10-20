# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fail unless the crate, the wheel and the npm packages agree on one version.

usage: check-versions.py [--tag v0.7.0 | --tag py-v0.7.0]

Cargo.toml is the source of truth (maturin reads the wheel version from it). package.json
and every optionalDependency must equal it, and a release tag must name it.
"""
import argparse
import json
import pathlib
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag")
    args = parser.parse_args()

    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text())["package"]["version"]
    package = json.loads((ROOT / "package.json").read_text())
    problems = []
    if package["version"] != cargo:
        problems.append(f"package.json is {package['version']}, Cargo.toml is {cargo}")
    for name, version in package.get("optionalDependencies", {}).items():
        if version != cargo:
            problems.append(f"optionalDependency {name} is {version}, Cargo.toml is {cargo}")
    if args.tag:
        tag_version = args.tag.removeprefix("refs/tags/").removeprefix("py-").removeprefix("v")
        if tag_version != cargo:
            problems.append(f"tag {args.tag} does not name version {cargo}")
    print(json.dumps({"version": cargo, "problems": len(problems)}))
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
