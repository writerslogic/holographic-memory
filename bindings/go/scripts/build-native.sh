#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
#
# From a clean clone of the repository: builds the static C ABI library with cargo and
# installs it at bindings/go/lib/<GOOS>_<GOARCH>/ where the cgo directives look first.
# Needs a Rust toolchain (MSRV 1.89) and a C compiler. CARGO_TARGET_DIR is honoured.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
c_crate="$here/../c"
goos="${GOOS:-$(go env GOOS)}"
goarch="${GOARCH:-$(go env GOARCH)}"
triple=""
case "$goos/$goarch" in
  windows/amd64) triple="x86_64-pc-windows-gnu" ;;
esac
args=(build --release --manifest-path "$c_crate/Cargo.toml")
[[ -n "$triple" ]] && args+=(--target "$triple")
cargo "${args[@]}"
target_dir="${CARGO_TARGET_DIR:-$c_crate/target}"
[[ -n "$triple" ]] && target_dir="$target_dir/$triple"
install -d "$here/lib/${goos}_${goarch}"
install -m 0644 "$target_dir/release/libholographic_memory_c.a" "$here/lib/${goos}_${goarch}/"
cp "$c_crate/include/holographic_memory.h" "$here/include/holographic_memory.h"
echo "installed $here/lib/${goos}_${goarch}/libholographic_memory_c.a"
