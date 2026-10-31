#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Regenerates include/writerslogic_warm.h. Pass --check to fail if the committed header is stale.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cbindgen="${CBINDGEN:-cbindgen}"
out="$here/include/writerslogic_warm.h"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
(cd "$here" && "$cbindgen" --config cbindgen.toml --crate writerslogic-warm-c --output "$tmp")
if [[ "${1:-}" == "--check" ]]; then
  diff -u "$out" "$tmp"
else
  mv "$tmp" "$out"
  trap - EXIT
fi
