#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
scratch_dir=$(mktemp -d)
trap 'rm -rf "$scratch_dir"' EXIT

if [[ ! -d "$repo_root/interop/w3c/node_modules" ]]; then
  npm ci --ignore-scripts --prefix "$repo_root/interop/w3c"
fi
cd "$repo_root"
cargo run --quiet --locked --features provenance --example emit_w3c_credential > "$scratch_dir/credential.json"
node interop/w3c/verify.mjs "$scratch_dir/credential.json"
