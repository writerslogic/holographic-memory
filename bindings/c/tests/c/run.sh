#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Builds the library and the C test program and runs it. MODE=leaks (macOS `leaks`) or
# MODE=valgrind additionally checks for leaks at exit. CARGO_TARGET_DIR selects the build dir.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
profile="${PROFILE:-debug}"
flags=()
[[ "$profile" == "release" ]] && flags+=(--release)
cargo build ${flags[@]+"${flags[@]}"} --manifest-path "$here/Cargo.toml"
target="${CARGO_TARGET_DIR:-$here/target}/$profile"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
case "$(uname -s)" in
  Darwin) static=(-framework CoreFoundation -framework Security -liconv) ;;
  *) static=(-ldl) ;;
esac
cflags=(-std=c11 -D_DEFAULT_SOURCE -Wall -Wextra -Werror -g -I"$here/include")
cc "${cflags[@]}" "$here/tests/c/test_hms.c" -o "$out/test_hms" \
  -L"$target" -lholographic_memory_c -lpthread -lm
# The static archive must be self-contained too.
cc "${cflags[@]}" "$here/tests/c/test_hms.c" "$target/libholographic_memory_c.a" \
  -o "$out/test_hms_static" "${static[@]}" -lpthread -lm
export DYLD_LIBRARY_PATH="$target" LD_LIBRARY_PATH="$target"
case "${MODE:-}" in
  leaks) leaks --atExit -- "$out/test_hms" ;;
  valgrind) valgrind --leak-check=full --error-exitcode=1 "$out/test_hms" ;;
  *)
    "$out/test_hms"
    "$out/test_hms_static"
    ;;
esac
