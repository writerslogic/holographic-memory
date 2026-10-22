#!/usr/bin/env bash
set -euo pipefail
qag_task="$HOME/sol-qag"
qag_out="$qag_task/v4"
export HMS_QAG_OUTER_LOCK=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1
test -f "$qag_out/final-gate.json"
for qag_dataset in llama yi; do
  uv run --no-project --python "$HOME/vibe-hms/mamba/envs/vibe/bin/python" python "$qag_task/repo-v4/benchmarks/public/qag_serving_execute.py" \
    --view "$qag_task/runs/$qag_dataset-full-validation-v1/evaluation.hdf5" \
    --training "$qag_task/data/$qag_dataset-full-training/queries.f32" \
    --extension "$qag_out/holographic_memory.so" --diagnostic "$qag_out/diagnostics/$qag_dataset.json" \
    --baseline "$qag_task/v2/runs/$qag_dataset-layout-v2" --out "$qag_out/runs/$qag_dataset" --run-id "$qag_dataset-v4"
done
