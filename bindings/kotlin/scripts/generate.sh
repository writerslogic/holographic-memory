#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Regenerates the committed Kotlin sources for the JVM module and the Android module.
# LIBRARY=<libhms_mobile.dylib|.so> reuses a built library; otherwise a host debug build is made.
set -euo pipefail

kotlin_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gen="$kotlin_dir/../uniffi/scripts/generate-bindings.sh"

rm -rf "$kotlin_dir/jvm/src/main/kotlin" "$kotlin_dir/android/src/main/kotlin"
"$gen" kotlin "$kotlin_dir/jvm/src/main/kotlin" "$kotlin_dir/uniffi-jvm.toml"
"$gen" kotlin "$kotlin_dir/android/src/main/kotlin" "$kotlin_dir/uniffi-android.toml"
