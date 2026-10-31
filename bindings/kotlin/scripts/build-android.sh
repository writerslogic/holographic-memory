#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Builds libwarm_mobile.so for the four Android ABIs into android/src/main/jniLibs with cargo-ndk.
#
# Requires: rustup, `cargo install cargo-ndk`, and the Android NDK (r26+; r27.3 tested):
# ANDROID_NDK_HOME, or ANDROID_HOME/ndk/<version> (the highest version is used).
# Rust targets are added with `rustup target add`.
# Env: ABIS="arm64-v8a armeabi-v7a x86_64 x86", ANDROID_PLATFORM=24, CARGO_TARGET_DIR.
set -euo pipefail

kotlin_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
crate="$(cd "$kotlin_dir/../uniffi" && pwd)"
abis="${ABIS:-arm64-v8a armeabi-v7a x86_64 x86}"
platform="${ANDROID_PLATFORM:-24}"

command -v cargo-ndk >/dev/null || { echo "cargo-ndk not found: cargo install cargo-ndk --locked" >&2; exit 1; }

if [ -z "${ANDROID_NDK_HOME:-}" ]; then
  sdk="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}"
  ANDROID_NDK_HOME="$(ls -d "$sdk"/ndk/* 2>/dev/null | sort -V | tail -1 || true)"
fi
[ -d "${ANDROID_NDK_HOME:-}" ] || { echo "Android NDK not found: set ANDROID_NDK_HOME" >&2; exit 1; }
export ANDROID_NDK_HOME

triples=()
args=()
for abi in $abis; do
  args+=(-t "$abi")
  case "$abi" in
    arm64-v8a) triples+=(aarch64-linux-android) ;;
    armeabi-v7a) triples+=(armv7-linux-androideabi) ;;
    x86_64) triples+=(x86_64-linux-android) ;;
    x86) triples+=(i686-linux-android) ;;
    *) echo "unknown ABI $abi" >&2; exit 2 ;;
  esac
done
rustup target add "${triples[@]}"

cd "$crate"
cargo ndk "${args[@]}" --platform "$platform" -o "$kotlin_dir/android/src/main/jniLibs" build --release --lib
ls -l "$kotlin_dir"/android/src/main/jniLibs/*/libwarm_mobile.so
