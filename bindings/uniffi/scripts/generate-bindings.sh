#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Generates Swift or Kotlin sources from a compiled hms_mobile library's UniFFI metadata.
#
#   generate-bindings.sh swift  <out-dir>
#   generate-bindings.sh kotlin <out-dir> [global-config.toml]
#
# LIBRARY=<path> names an already built libhms_mobile.{dylib,so} (any target built from this
# source: the metadata is identical). Without it, a debug library is built for the host.
# CARGO_TARGET_DIR defaults to <crate>/target.
set -euo pipefail

language="${1:?usage: generate-bindings.sh swift|kotlin <out-dir> [global-config.toml]}"
out="${2:?usage: generate-bindings.sh swift|kotlin <out-dir> [global-config.toml]}"
crate="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="${3:-}"
target="${CARGO_TARGET_DIR:-$crate/target}"

case "$language" in
  swift | kotlin) ;;
  *) echo "language must be swift or kotlin" >&2; exit 2 ;;
esac

library="${LIBRARY:-}"
if [ -z "$library" ]; then
  (cd "$crate" && cargo build --lib)
  case "$(uname -s)" in
    Darwin) library="$target/debug/libhms_mobile.dylib" ;;
    *) library="$target/debug/libhms_mobile.so" ;;
  esac
fi

mkdir -p "$out"
cargo run --quiet --manifest-path "$crate/bindgen/Cargo.toml" -- generate \
  --library "$library" \
  ${config:+--config "$config"} \
  --language "$language" \
  --no-format \
  --out-dir "$out"
echo "generated $language bindings in $out"
