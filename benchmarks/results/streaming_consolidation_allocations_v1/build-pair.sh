#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$HOME/hms-streaming-alloc"
mkdir -p parent-source candidate-source evidence
tar -xf parent-minimal.tar -C parent-source
tar -xf candidate-minimal.tar -C candidate-source
for arm in parent candidate; do
    if [[ ! -d "$arm-target" ]]; then cp -a --reflink=auto "$HOME/hms-streaming-quantized/target" "$arm-target"; fi
    (
      cd "$arm-source"
      taskset -c 18,19 nice -n 19 env CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR="$HOME/hms-streaming-alloc/$arm-target" \
        cargo build --locked --release --example qgraph_consolidation
    ) >"evidence/$arm-build.log" 2>&1
    cp "$arm-target/release/examples/qgraph_consolidation" "$arm-run"
done
(
  cd parent-source
  taskset -c 18,19 nice -n 19 env CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR="$HOME/hms-streaming-alloc/parent-target" \
    cargo build --locked --release --features experimental --example qgraph_build_profile
) > evidence/profile-build.log 2>&1
cp parent-target/release/examples/qgraph_build_profile parent-profile
sha256sum parent-run candidate-run parent-profile parent-minimal.tar candidate-minimal.tar > evidence/source-binary-sha256.txt
date -u > build-complete
