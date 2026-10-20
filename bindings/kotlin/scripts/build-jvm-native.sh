#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Builds libhms_mobile for the host and stages it where the :jvm module's resources and the
# JNA resource-path convention (<os>-<arch>/) expect it: jvm/build/natives/.
# Env: CARGO_TARGET_DIR (default <crate>/target), PROFILE=release|debug (default release).
set -euo pipefail

kotlin_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
crate="$(cd "$kotlin_dir/../uniffi" && pwd)"
target="${CARGO_TARGET_DIR:-$crate/target}"
profile="${PROFILE:-release}"

flags=()
if [ "$profile" = release ]; then flags=(--release); fi
host="$(rustc -vV | sed -n 's/^host: //p')"
# `strip = true` makes the macOS 27 linker's strip emit a dylib dyld refuses to load
# (mis-aligned LINKEDIT), so the host library keeps its symbol table; it has its own target dir.
host_target="$target/jvm-host"
(cd "$crate" && CARGO_PROFILE_RELEASE_STRIP=debuginfo cargo build ${flags[@]+"${flags[@]}"} --lib --target "$host" --target-dir "$host_target")

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) platform=darwin-aarch64; file=libhms_mobile.dylib ;;
  Darwin-x86_64) platform=darwin-x86-64; file=libhms_mobile.dylib ;;
  Linux-x86_64) platform=linux-x86-64; file=libhms_mobile.so ;;
  Linux-aarch64) platform=linux-aarch64; file=libhms_mobile.so ;;
  *) echo "unsupported host $(uname -sm)" >&2; exit 1 ;;
esac

out="$kotlin_dir/jvm/build/natives/$platform"
mkdir -p "$out"
cp "$host_target/$host/$profile/$file" "$out/$file"
echo "staged $out/$file"
