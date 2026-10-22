#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
qag_task="$HOME/sol-qag"
qag_out="$qag_task/v2"
qag_python="$HOME/vibe-hms/mamba/envs/vibe/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 HMS_QAG_OUTER_LOCK=1
test -f "$qag_out/candidate-holographic_memory.so"
test ! -e "$qag_out/runs/llama-layout-v2"
test ! -e "$qag_out/runs/yi-layout-v2"
uv pip install --python "$qag_python" pytest > "$qag_out/logs/install-pytest.txt" 2>&1
uv run --no-project --python "$qag_python" python - "$qag_out/candidate-holographic_memory.so" "$qag_task/repo-v2/tests/python/test_vector_index.py" > "$qag_out/logs/native-python-resumed.txt" 2>&1 <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / 'vibe-hms/vibe'))
from vibe.algorithms.qag_pair.module import _load_extension
sys.modules['holographic_memory'] = _load_extension(Path(sys.argv[1]), '_hms_qag_v2_native')
import pytest
raise SystemExit(pytest.main(['-q', sys.argv[2]]))
PY
uv run --no-project --python "$qag_python" python "$qag_out/sources/qag_hms_smoke.py" --vibe "$HOME/vibe-hms/vibe" --extension "$qag_out/candidate-holographic_memory.so" --wrapper "$qag_out/sources/hms_wrapper.py" --baseline "$qag_task/artifacts/baseline-holographic_memory.so" > "$qag_out/logs/wrapper-native.json" 2>&1
for qag_dataset in llama yi; do
  uv run --no-project --python "$qag_python" python "$qag_out/sources/qag_layout_execute.py" --view "$qag_task/runs/$qag_dataset-full-validation-v1/evaluation.hdf5" --training "$qag_task/data/$qag_dataset-full-training/queries.f32" --baseline "$qag_out/candidate-holographic_memory.so" --candidate "$qag_out/candidate-holographic_memory.so" --out "$qag_out/runs/$qag_dataset-layout-v2" --run-id "$qag_dataset-layout-v2"
done
printf '{"status":"complete","resumed_from":"missing pytest, before native checks; no rebuild"}\n' > "$qag_out/layout-completion.json"
bash "$qag_task/v3/kernel-v3.sh"
