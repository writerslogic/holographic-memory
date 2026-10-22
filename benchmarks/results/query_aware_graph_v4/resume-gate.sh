#!/usr/bin/env bash
set -euo pipefail
qag_task="$HOME/sol-qag"
qag_out="$qag_task/v4"
export CARGO_TARGET_DIR="$qag_task/target-v4" CARGO_BUILD_JOBS=2 CARGO_TERM_COLOR=never
export CARGO_PROFILE_DEV_OPT_LEVEL=1 CARGO_PROFILE_TEST_OPT_LEVEL=1 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1
unset RUSTFLAGS
cd "$qag_task/repo-v4"
cp "$qag_out/logs/all-features-tests.log" "$qag_out/logs/all-features-compile-second.log"
sha256sum src/bin/public-bench.rs tests/qgraph_prefetch.rs > "$qag_out/final-callers.sha256"
cargo +1.96.0 test --locked --all-features --no-run > "$qag_out/logs/all-features-compile.log" 2>&1
cargo +1.96.0 test --locked --all-features > "$qag_out/logs/all-features-tests.log" 2>&1
uv run --no-project --python "$HOME/vibe-hms/mamba/envs/vibe/bin/python" python "$PWD/benchmarks/results/query_aware_graph_v4/verify-final.py" > "$qag_out/logs/final-python-tests.log" 2>&1
bash "$PWD/benchmarks/results/query_aware_graph_v4/timing.sh"
