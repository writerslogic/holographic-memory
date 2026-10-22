#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
qag_task="$HOME/sol-qag"
qag_out="$qag_task/v2"
qag_python="$HOME/vibe-hms/mamba/envs/vibe/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 HMS_QAG_OUTER_LOCK=1
export CARGO_TARGET_DIR="$qag_task/target-python" CARGO_BUILD_JOBS=2 RUSTFLAGS='-C target-cpu=native'
cd "$qag_task/repo-v2"
test -f "$qag_out/profile-build.json"
uv run --no-project --python "$qag_python" python - "$qag_out" <<'PY'
import json, sys
from pathlib import Path
for name in ('llama', 'yi'):
    profile = json.loads((Path(sys.argv[1]) / 'profiles' / (name + '-high.json')).read_text())
    assert profile['returned_ids_identical'] and profile['persisted_bytes_identical']
    a, b = profile['layouts']
    assert b['index_bytes'] < 0.9 * a['index_bytes'], 'insufficient memory saving to justify a layout timing run'
PY
cargo +1.96.0 build --release --locked --features python --lib > "$qag_out/logs/python-build.log" 2>&1
cp "$CARGO_TARGET_DIR/release/libholographic_memory.so" "$qag_out/candidate-holographic_memory.so"
sha256sum "$qag_out/candidate-holographic_memory.so" > "$qag_out/python-binary.sha256"
for qag_dataset in llama yi; do
  uv run --no-project --python "$qag_python" python "$qag_out/sources/qag_layout_execute.py" --view "$qag_task/runs/$qag_dataset-full-validation-v1/evaluation.hdf5" --training "$qag_task/data/$qag_dataset-full-training/queries.f32" --baseline "$qag_out/candidate-holographic_memory.so" --candidate "$qag_out/candidate-holographic_memory.so" --out "$qag_out/runs/$qag_dataset-layout-v2" --run-id "$qag_dataset-layout-v2"
done
