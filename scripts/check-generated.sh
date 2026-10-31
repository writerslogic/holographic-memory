#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
TS_RS_EXPORT_DIR="$scratch/bindings" WARM_SCHEMA_EXPORT_DIR="$scratch/schemas" \
  cargo test --locked --lib --features node-api,server export
if ! diff -ru schemas "$scratch/schemas" || ! diff -ru bindings/bindings "$scratch/bindings/bindings"; then
  echo 'Generated artifacts are stale. Regenerate with: cargo test --locked --lib --features node-api,server export' >&2
  exit 1
fi
