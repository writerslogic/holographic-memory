#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Builds the crate's rustdoc from one source tree and copies it to an output directory, which
# scripts/docs_site.py embeds in the site under api/rust/doc/ (WARM_RUSTDOC_DIR).
#
# Usage: scripts/build-rustdoc.sh <source-dir> <out-dir>
# The `provenance` feature pulls in `security`, so both appear. `experimental` is left out on
# purpose: it is documented as not part of the published API. Set CARGO_TARGET_DIR to share
# compiled dependencies between the versions of the site.
set -euo pipefail

src="$1"
out="$2"
target="${CARGO_TARGET_DIR:-$src/target}"
cd "$src"
cargo doc --no-deps --lib --features provenance --locked --quiet --target-dir "$target"
rm -rf "${out:?}"
mkdir -p "$out"
cp -R "$target/doc/." "$out/"
test -f "$out/writerslogic_warm/index.html"
