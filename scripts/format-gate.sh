#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# Golden stores are append-only: fails if a pull request modifies or deletes any file that
# already exists under tests/golden/ at the base revision. Usage: format-gate.sh <base-ref>
set -euo pipefail
base="${1:?base ref}"
bad=$(git diff --name-status --no-renames "$base"...HEAD -- tests/golden |
  awk '$1 != "A" && $2 != "tests/golden/MANIFEST.json" {print}')
if [ -n "$bad" ]; then
  echo "Existing golden files changed or were deleted (golden stores are append-only):" >&2
  echo "$bad" >&2
  echo "Keep reading the old format, bump the version, and add a NEW golden store." >&2
  exit 1
fi
# The manifest may only gain entries: no line of it may be removed or altered.
if git diff "$base"...HEAD -- tests/golden/MANIFEST.json | grep -E '^-[^-]' | grep -vE '^-\s*$' >/dev/null; then
  removed=$(git diff "$base"...HEAD -- tests/golden/MANIFEST.json | grep -E '^-[^-]' || true)
  # Closing braces and commas move when entries are appended; real entries contain a quote.
  if echo "$removed" | grep -E '"[^"]+": \[|"(v|rejected)[^"]*": \{' >/dev/null; then
    echo "MANIFEST.json entries were removed or altered:" >&2
    echo "$removed" >&2
    exit 1
  fi
fi
echo "golden stores unchanged"
