#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
qag_task="$HOME/sol-qag"
qag_out="$qag_task/v4"
export CARGO_TARGET_DIR="$qag_task/target-v4" CARGO_BUILD_JOBS=2 CARGO_TERM_COLOR=never
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1
cd "$qag_task/repo-v4"
cp "$qag_out/holographic_memory.so" "$qag_out/diagnostic-holographic_memory.so"
cp "$CARGO_TARGET_DIR/release/qag-diagnose" "$qag_out/diagnostic-qag-diagnose"
sha256sum Cargo.toml Cargo.lock src/bin/{qag-diagnose,public-bench}.rs src/python/vector_index.rs src/core/qgraph/{mod.rs,vertex.rs,vertex/file.rs,update.rs,diagnostics.rs,projection.rs,kernels.rs} > "$qag_out/final-source.sha256"
export RUSTFLAGS='-C target-cpu=native'
cargo +1.96.0 build --release --locked --features experimental,python --lib --bin qag-diagnose > "$qag_out/logs/final-build.log" 2>&1
cp "$CARGO_TARGET_DIR/release/libholographic_memory.so" "$qag_out/holographic_memory.so"
sha256sum "$qag_out/holographic_memory.so" "$CARGO_TARGET_DIR/release/qag-diagnose" > "$qag_out/final-binaries.sha256"
unset RUSTFLAGS
export CARGO_PROFILE_DEV_OPT_LEVEL=1 CARGO_PROFILE_TEST_OPT_LEVEL=1 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
cargo +1.96.0 test --locked --all-features > "$qag_out/logs/all-features-tests.log" 2>&1
uv run --no-project --python "$HOME/vibe-hms/mamba/envs/vibe/bin/python" python "$PWD/benchmarks/results/query_aware_graph_v4/verify-final.py" > "$qag_out/logs/final-python-tests.log" 2>&1
