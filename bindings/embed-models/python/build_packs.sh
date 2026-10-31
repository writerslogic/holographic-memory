#!/bin/bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# Fetches the pinned upstream files for registry models, converts them and writes the packs next
# to the Python package and the Node package. Needs a binary built with
# `--features local-embed-download`.
set -euo pipefail
BIN=${1:?usage: build_packs.sh /path/to/warm-embed [model-id ...]}
shift
HERE=$(cd "$(dirname "$0")" && pwd)
IDS=("$@")
if [ ${#IDS[@]} -eq 0 ]; then
  while IFS=$'\t' read -r id _; do IDS+=("$id"); done < <("$BIN" list)
fi
mkdir -p "$HERE/writerslogic_warm_models/packs" "$HERE/../node/packs"
for id in "${IDS[@]}"; do
  "$BIN" fetch "$id" --out "$HERE/writerslogic_warm_models/packs"
done
cp "$HERE"/writerslogic_warm_models/packs/*.hmsemb "$HERE/../node/packs/"
