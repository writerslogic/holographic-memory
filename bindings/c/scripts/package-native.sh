#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Builds the C ABI library in release mode and packs one platform archive:
#   writerslogic-warm-c-<version>-<rid>.tar.gz  (+ .sha256)
# holding lib/ (static and shared libraries), include/writerslogic_warm.h and LICENSE.
#
# Usage: package-native.sh [<rid> [<rust-target-triple> [<out-dir>]]]
#   rid  one of osx-arm64 osx-x64 linux-x64 linux-arm64 win-x64 win-arm64 win-x64-gnu;
#        defaults to the host. `win-x64-gnu` is the MinGW static library cgo needs (Go on
#        Windows); `win-*` MSVC archives serve Java and .NET.
# The default output directory is bindings/c/dist. CARGO_TARGET_DIR is honoured.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="$(cd "$here/../.." && pwd)"

host_rid() {
  local os arch
  case "$(uname -s)" in Darwin) os=osx ;; Linux) os=linux ;; MINGW* | MSYS* | CYGWIN*) os=win ;; *) echo "unsupported OS" >&2; exit 1 ;; esac
  case "$(uname -m)" in arm64 | aarch64) arch=arm64 ;; x86_64 | AMD64) arch=x64 ;; *) echo "unsupported CPU" >&2; exit 1 ;; esac
  echo "$os-$arch"
}

rid="${1:-$(host_rid)}"
triple="${2:-}"
out="${3:-$here/dist}"
version="$(sed -n 's/^version = "\(.*\)"/\1/p' "$here/Cargo.toml" | head -1)"

args=(build --release --manifest-path "$here/Cargo.toml")
[[ -n "$triple" ]] && args+=(--target "$triple")
cargo "${args[@]}"

target_dir="${CARGO_TARGET_DIR:-$here/target}"
[[ -n "$triple" ]] && target_dir="$target_dir/$triple"
build="$target_dir/release"

name="writerslogic-warm-c-$version-$rid"
stage="$(mktemp -d)/$name"
mkdir -p "$stage/lib" "$stage/include" "$out"
cp "$here/include/writerslogic_warm.h" "$here/include/holographic_memory.h" "$stage/include/"
cp "$root/LICENSE" "$stage/LICENSE"
case "$rid" in
  osx-*) cp "$build/libwriterslogic_warm_c.a" "$build/libwriterslogic_warm_c.dylib" "$stage/lib/" ;;
  linux-*) cp "$build/libwriterslogic_warm_c.a" "$build/libwriterslogic_warm_c.so" "$stage/lib/" ;;
  win-x64-gnu) cp "$build/libwriterslogic_warm_c.a" "$build/writerslogic_warm_c.dll" "$stage/lib/" ;;
  win-*) cp "$build/writerslogic_warm_c.lib" "$build/writerslogic_warm_c.dll" "$build/writerslogic_warm_c.dll.lib" "$stage/lib/" ;;
  *) echo "unknown rid $rid" >&2; exit 1 ;;
esac
for library in "$stage"/lib/*; do
  legacy="${library/writerslogic_warm_c/holographic_memory_c}"
  cp "$library" "$legacy"
done
tar -C "$(dirname "$stage")" -czf "$out/$name.tar.gz" "$name"
(cd "$out" && { shasum -a 256 "$name.tar.gz" 2>/dev/null || sha256sum "$name.tar.gz"; } >"$name.tar.gz.sha256")
rm -rf "$(dirname "$stage")"
echo "$out/$name.tar.gz"
