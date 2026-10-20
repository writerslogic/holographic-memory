#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# Build the hms-mcp release binary for the host target, name it as the launchers expect, and
# record its sha256 in both launcher checksum tables. Run once per target (CI matrix), merging
# the produced checksums.json files, then upload the binaries to the `mcp-v<version>` release.
#
#   integrations/mcp/scripts/build-release.sh [out-dir]
#
# Env: FEATURES (default "mcp,security"), CARGO_TARGET_DIR.
set -euo pipefail

root="$(cd "$(dirname "$0")/../../.." && pwd)"
out="${1:-$root/integrations/mcp/dist}"
features="${FEATURES:-mcp,security}"
mkdir -p "$out"

triple="$(rustc -vV | sed -n 's/^host: //p')"
exe=""
case "$triple" in *windows*) exe=".exe" ;; esac

(cd "$root" && cargo build --release --locked --features "$features" --bin hms-mcp)
target_dir="${CARGO_TARGET_DIR:-$root/target}"
asset="hms-mcp-${triple}${exe}"
cp "$target_dir/release/hms-mcp${exe}" "$out/$asset"

if command -v sha256sum >/dev/null 2>&1; then
  sum="$(sha256sum "$out/$asset" | cut -d' ' -f1)"
else
  sum="$(shasum -a 256 "$out/$asset" | cut -d' ' -f1)"
fi
printf '%s  %s\n' "$sum" "$asset" >"$out/$asset.sha256"

# Merge this target's checksum into the table both launchers read.
node - "$out/checksums.json" "$triple" "$sum" <<'JS'
const fs = require("fs");
const [file, triple, sum] = process.argv.slice(2);
let table = { assets: {} };
try { table = JSON.parse(fs.readFileSync(file, "utf8")); } catch {}
table.assets[triple] = sum;
fs.writeFileSync(file, JSON.stringify(table, null, 2) + "\n");
JS
cp "$out/checksums.json" "$root/integrations/mcp/npm/checksums.json"
cp "$out/checksums.json" "$root/integrations/mcp/pypi/src/holographic_memory_mcp/checksums.json"
echo "built $out/$asset ($sum)"
