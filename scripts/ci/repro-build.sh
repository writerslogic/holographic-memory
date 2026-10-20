#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Build the release artefacts that must be bit-for-bit reproducible and record their hashes.
# usage: scripts/ci/repro-build.sh <out-dir>      (run from the repository root)
#
# The reproducible-build workflow runs this on two independent runners from two different
# checkout paths and compares the resulting SHA256SUMS. What makes the output independent of
# the machine: the lockfile (--locked), a fixed SOURCE_DATE_EPOCH (the commit time), no
# incremental compilation, and --remap-path-prefix for the checkout, cargo home and rustup home.
set -euo pipefail

out=${1:?usage: repro-build.sh <out-dir>}
mkdir -p "$out"
out=$(cd "$out" && pwd)
root=$(pwd)

SOURCE_DATE_EPOCH=$(git log -1 --format=%ct)
export SOURCE_DATE_EPOCH
export CARGO_INCREMENTAL=0
export CARGO_TARGET_DIR="$root/target"
cargo_home=${CARGO_HOME:-$HOME/.cargo}
rustup_home=${RUSTUP_HOME:-$HOME/.rustup}
export RUSTFLAGS="--remap-path-prefix=$root=/build --remap-path-prefix=$cargo_home=/cargo --remap-path-prefix=$rustup_home=/rustup"

echo "rustc: $(rustc --version --verbose | tr '\n' ' ')"
echo "SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH root=$root"

# 1. Command line tools.
cargo build --locked --release --features server --bin hms-eval --bin hms-server
cp target/release/hms-eval target/release/hms-server "$out/"

# 2. The napi cdylib (what becomes the .node binding), without the Node loader step.
cargo build --locked --release --lib --features node-api,security
case "$(uname -s)" in
  Linux) lib=libholographic_memory.so ;;
  Darwin) lib=libholographic_memory.dylib ;;
  *) echo "unsupported host" >&2; exit 1 ;;
esac
cp "target/release/$lib" "$out/node-api-$lib"

# 3. The Python wheel.
python3 -m venv "$out/.venv"
"$out/.venv/bin/pip" install --quiet 'maturin==1.15.0'
"$out/.venv/bin/maturin" build --locked --release --out "$out/wheel"
cp "$out"/wheel/*.whl "$out/"

# 4. The crate as published.
cargo package --locked --no-verify --no-default-features --allow-dirty
cp target/package/*.crate "$out/"

rm -rf "$out/.venv" "$out/wheel"
(cd "$out" && sha256sum -- * | sort -k2 > SHA256SUMS)
cat "$out/SHA256SUMS"
