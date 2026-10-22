#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
qag_task="$HOME/sol-qag"
qag_out="$qag_task/v3"
qag_python="$HOME/vibe-hms/mamba/envs/vibe/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 HMS_QAG_OUTER_LOCK=1
export CARGO_TARGET_DIR="$qag_task/target-python" CARGO_BUILD_JOBS=2 RUSTFLAGS='-C target-cpu=native'
unset HMS_QGRAPH_PREFETCH HMS_QGRAPH_BATCH_EST HMS_QGRAPH_FRONTIER HMS_QGRAPH_BLOCK_SCREEN HMS_QGRAPH_KERNEL
export HMS_QGRAPH_HUGEPAGES=0
test -f "$qag_task/v2/runs/yi-layout-v2/completion.json"
test ! -e "$qag_out/candidate-holographic_memory.so"
cd "$qag_task/repo-v3"
sha256sum Cargo.toml Cargo.lock src/core/qgraph/kernels.rs > "$qag_out/source.sha256"
rustc +1.96.0 --version --verbose > "$qag_out/compiler.txt"
cargo +1.96.0 build --release --locked --features python --lib > "$qag_out/logs/python-build.log" 2>&1
cp "$CARGO_TARGET_DIR/release/libholographic_memory.so" "$qag_out/candidate-holographic_memory.so"
sha256sum "$qag_out/candidate-holographic_memory.so" "$qag_task/v2/candidate-holographic_memory.so" > "$qag_out/binaries.sha256"
uv run --no-project --python "$qag_python" python - "$qag_out/candidate-holographic_memory.so" "$qag_task/repo-v3/tests/python/test_vector_index.py" > "$qag_out/logs/native-python.txt" 2>&1 <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / 'vibe-hms/vibe'))
from vibe.algorithms.qag_pair.module import _load_extension
sys.modules['holographic_memory'] = _load_extension(Path(sys.argv[1]), '_hms_qag_v3_native')
import pytest
raise SystemExit(pytest.main(['-q', sys.argv[2]]))
PY
uv run --no-project --python "$qag_python" python "$qag_out/sources/qag_hms_smoke.py" --vibe "$HOME/vibe-hms/vibe" --extension "$qag_out/candidate-holographic_memory.so" --wrapper "$qag_out/sources/hms_wrapper.py" --baseline "$qag_task/artifacts/baseline-holographic_memory.so" > "$qag_out/logs/wrapper-native.txt" 2>&1
for qag_dataset in llama yi; do
  uv run --no-project --python "$qag_python" python "$qag_out/sources/qag_layout_execute.py" --revision --view "$qag_task/runs/$qag_dataset-full-validation-v1/evaluation.hdf5" --training "$qag_task/data/$qag_dataset-full-training/queries.f32" --baseline "$qag_task/v2/candidate-holographic_memory.so" --candidate "$qag_out/candidate-holographic_memory.so" --out "$qag_out/runs/$qag_dataset-kernel-v3" --run-id "$qag_dataset-kernel-v3"
done
printf '{"status":"complete","phase":"validation-only exact byte-dot comparison","compiler":"1.96.0","rustflags":"-C target-cpu=native"}\n' > "$qag_out/completion.json"
