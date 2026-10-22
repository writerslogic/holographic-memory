#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
qag_task="$HOME/sol-qag"
qag_out="$qag_task/v2"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1
export CARGO_TARGET_DIR="$qag_task/target" CARGO_BUILD_JOBS=2 RUSTFLAGS='-C target-cpu=native'
unset HMS_QGRAPH_PREFETCH HMS_QGRAPH_BATCH_EST HMS_QGRAPH_FRONTIER HMS_QGRAPH_BLOCK_SCREEN HMS_QGRAPH_KERNEL
export HMS_QGRAPH_HUGEPAGES=0
cd "$qag_task/repo-v2"
test ! -e "$qag_out/profile-build.json"
sha256sum Cargo.toml Cargo.lock src/bin/qag-diagnose.rs src/core/qgraph/{mod.rs,vertex.rs,projection.rs,projection_diagnostic.rs,vertex/file.rs} > "$qag_out/profile-source.sha256"
rustc +1.96.0 --version --verbose > "$qag_out/compiler.txt"
cp -n "$qag_task/target/release/qag-diagnose" "$qag_task/artifacts/qag-diagnose-v1-patience"
cargo +1.96.0 build --release --locked --features experimental --bin qag-diagnose > "$qag_out/logs/profile-build.log" 2>&1
sha256sum "$qag_task/target/release/qag-diagnose" > "$qag_out/profile-binary.sha256"
for qag_dataset in llama yi; do
  qag_ef=256
  if [ "$qag_dataset" = yi ]; then qag_ef=512; fi
  "$qag_task/target/release/qag-diagnose" --data "$qag_task/data/$qag_dataset-full" --graph "$qag_task/data/$qag_dataset-full.graph" --out "$qag_out/profiles/$qag_dataset-high.json" --bits 8 --ef "$qag_ef" --rerank 400 --training "$qag_task/data/$qag_dataset-full-training" --training-count 32768 --projected-degree 32 --profile --persist-dir "$qag_out/profiles/$qag_dataset-persisted" > "$qag_out/logs/profile-$qag_dataset.log" 2>&1
done
printf '{"status":"complete","profile_only":true,"compiler":"1.96.0","rustflags":"-C target-cpu=native","timing_lock":"~/hms-timing.lock","affinity":"unpinned diagnostic; not QPS evidence"}\n' > "$qag_out/profile-build.json"
