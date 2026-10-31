#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Install the built wheel into an empty virtualenv (no source tree on the path, no network)
# and run a real search through it.
# usage: scripts/ci/smoke-wheel.sh <python> <dist-dir>     (run from the repository root)
set -euo pipefail

python=${1:?usage: smoke-wheel.sh <python> <dist-dir>}
dist=${2:?usage: smoke-wheel.sh <python> <dist-dir>}
version=$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n 1)

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
"$python" -m venv "$work/venv"
bin="$work/venv/bin"
[ -d "$work/venv/Scripts" ] && bin="$work/venv/Scripts"

"$bin/python" -m pip install --quiet --no-index --find-links "$dist" warm-python
# -I: ignore the working directory, so the installed wheel is what gets imported.
"$bin/python" -I scripts/ci/smoke_python.py --expect-version "$version"
