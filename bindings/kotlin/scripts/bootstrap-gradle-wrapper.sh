#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Fetch gradle-wrapper.jar, which is not committed (a binary in the tree). The jar comes from
# the Gradle release tag that matches gradle-wrapper.properties and must match the pinned
# SHA-256 below; the distribution it then downloads is pinned by distributionSha256Sum.
set -euo pipefail

VERSION=8.14.6
JAR_SHA256=7d3a4ac4de1c32b59bc6a4eb8ecb8e612ccd0cf1ae1e99f66902da64df296172

dest="$(cd "$(dirname "$0")/.." && pwd)/gradle/wrapper/gradle-wrapper.jar"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
  "https://raw.githubusercontent.com/gradle/gradle/v${VERSION}/gradle/wrapper/gradle-wrapper.jar" \
  --output "$tmp"
actual="$(shasum -a 256 "$tmp" | cut -d' ' -f1)"
if [ "$actual" != "$JAR_SHA256" ]; then
  echo "gradle-wrapper.jar checksum mismatch: got $actual" >&2
  exit 1
fi
mkdir -p "$(dirname "$dest")"
mv "$tmp" "$dest"
echo "wrote $dest"
