#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Builds Frameworks/hms_mobileFFI.xcframework and regenerates Sources/HolographicMemory.
#
# Requires macOS with Xcode (command line tools alone cannot build the iOS slices) and rustup.
# Rust targets are added with `rustup target add` (user-level). Slices:
#   iOS device            aarch64-apple-ios
#   iOS simulator         aarch64-apple-ios-sim + x86_64-apple-ios
#   macOS                 aarch64-apple-darwin  + x86_64-apple-darwin
#   Mac Catalyst          aarch64-apple-ios-macabi + x86_64-apple-ios-macabi
#   visionOS (opt-in)     aarch64-apple-visionos + aarch64-apple-visionos-sim
#                         VISIONOS=1; tier-3 targets, built with nightly -Zbuild-std and the
#                         visionOS SDK from Xcode 15+.
#
# Environment: CARGO_TARGET_DIR, IPHONEOS_DEPLOYMENT_TARGET (15.0), MACOSX_DEPLOYMENT_TARGET
# (12.0), XROS_DEPLOYMENT_TARGET (1.0), SLICES="ios ios-sim macos catalyst" to build a subset.
set -euo pipefail

swift_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
crate="$(cd "$swift_dir/../uniffi" && pwd)"
target="${CARGO_TARGET_DIR:-$crate/target}"
work="$target/xcframework"
slices="${SLICES:-ios ios-sim macos catalyst}"
if [ "${VISIONOS:-0}" = 1 ]; then slices="$slices visionos visionos-sim"; fi

export IPHONEOS_DEPLOYMENT_TARGET="${IPHONEOS_DEPLOYMENT_TARGET:-15.0}"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-12.0}"
export XROS_DEPLOYMENT_TARGET="${XROS_DEPLOYMENT_TARGET:-1.0}"

command -v xcodebuild >/dev/null || { echo "xcodebuild not found: install Xcode" >&2; exit 1; }
command -v rustup >/dev/null || { echo "rustup not found" >&2; exit 1; }

triples_for() {
  case "$1" in
    ios) echo aarch64-apple-ios ;;
    ios-sim) echo aarch64-apple-ios-sim x86_64-apple-ios ;;
    macos) echo aarch64-apple-darwin x86_64-apple-darwin ;;
    catalyst) echo aarch64-apple-ios-macabi x86_64-apple-ios-macabi ;;
    visionos) echo aarch64-apple-visionos ;;
    visionos-sim) echo aarch64-apple-visionos-sim ;;
    *) echo "unknown slice $1" >&2; exit 2 ;;
  esac
}

tier2=()
tier3=()
for slice in $slices; do
  for triple in $(triples_for "$slice"); do
    case "$slice" in
      visionos*) tier3+=("$triple") ;;
      *) tier2+=("$triple") ;;
    esac
  done
done

cd "$crate"
if [ "${#tier2[@]}" -gt 0 ]; then
  rustup target add "${tier2[@]}"
  args=()
  for triple in "${tier2[@]}"; do args+=(--target "$triple"); done
  cargo build --release --lib "${args[@]}"
fi
if [ "${#tier3[@]}" -gt 0 ]; then
  rustup toolchain install nightly --profile minimal -c rust-src
  for triple in "${tier3[@]}"; do
    cargo +nightly build --release --lib -Zbuild-std=std,panic_abort --target "$triple"
  done
fi

# Bindings come from a library built from this exact source.
if [ "${#tier2[@]}" -gt 0 ]; then
  host="$(rustc -vV | sed -n 's/^host: //p')"
  library="$target/$host/release/libhms_mobile.dylib"
  [ -f "$library" ] || { cargo build --release --lib --target "$host"; }
  LIBRARY="$library" "$crate/scripts/generate-bindings.sh" swift "$work/generated"
else
  "$crate/scripts/generate-bindings.sh" swift "$work/generated"
fi

headers="$work/headers"
rm -rf "$headers"
mkdir -p "$headers" "$swift_dir/Sources/HolographicMemory"
cp "$work/generated/hms_mobileFFI.h" "$headers/"
cp "$work/generated/hms_mobileFFI.modulemap" "$headers/module.modulemap"
cp "$work/generated/hms_mobile.swift" "$swift_dir/Sources/HolographicMemory/HmsMobile.swift"

lib() { echo "$target/$1/release/libhms_mobile.a"; }
fat() { # fat <slice> <triple>...  -> prints the merged library path
  local out="$work/libs/$1/libhms_mobile.a"
  shift
  mkdir -p "$(dirname "$out")"
  if [ "$#" -eq 1 ]; then cp "$(lib "$1")" "$out"; else
    lipo -create $(for t in "$@"; do lib "$t"; done) -output "$out"
  fi
  echo "$out"
}

xc=()
for slice in $slices; do
  merged="$(fat "$slice" $(triples_for "$slice"))"
  xc+=(-library "$merged" -headers "$headers")
done

out="$swift_dir/Frameworks/hms_mobileFFI.xcframework"
rm -rf "$out"
mkdir -p "$swift_dir/Frameworks"
xcodebuild -create-xcframework "${xc[@]}" -output "$out"
echo "built $out"
