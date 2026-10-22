#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 HMS_QAG_OUTER_LOCK=1
qag_task="$HOME/sol-qag"
cd "$qag_task/repo"
for qag_patience in 20 24 32; do
  "$qag_task/target/release/qag-diagnose" --data "$qag_task/data/llama-prefix" --graph "$qag_task/data/llama-prefix.graph" --out "$qag_task/results/llama-prefix-patience-$qag_patience.json" --bits 8 --ef 160 --rerank 200 --training "$qag_task/data/llama-prefix-training" --training-count 32768 --projected-degree 32 --patience "$qag_patience" > "$qag_task/logs/patience-$qag_patience.log" 2>&1
done
RUSTFLAGS='-C target-cpu=native' CARGO_TARGET_DIR="$qag_task/target-python" cargo +1.96.0 build --release --features python --lib > "$qag_task/logs/build-python-patience.log" 2>&1
cp "$qag_task/target-python/release/libholographic_memory.so" "$qag_task/artifacts/patience-holographic_memory.so"
sha256sum "$qag_task/artifacts/patience-holographic_memory.so" > "$qag_task/results/patience-python.sha256"
HMS_QAG_BASELINE_SO="$qag_task/artifacts/baseline-holographic_memory.so" HMS_QAG_CANDIDATE_SO="$qag_task/artifacts/patience-holographic_memory.so" uv run --no-project --python "$HOME/vibe-hms/mamba/envs/vibe/bin/python" --with pytest python "$qag_task/qag_smoke.py" --vibe "$HOME/vibe-hms/vibe" --python-tests "$qag_task/repo/tests/python/test_vector_index.py" --reserved "$qag_task/data/patience-smoke-reserved.u32" > "$qag_task/logs/python-patience-smoke.log" 2>&1
