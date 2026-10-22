#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 HMS_QAG_OUTER_LOCK=1
qag_task="$HOME/sol-qag"
qag_python="$HOME/vibe-hms/mamba/envs/vibe/bin/python"
for qag_dataset in llama yi; do
  uv run --no-project --python "$qag_python" python "$qag_task/qag_execute.py" freeze --phase validation --dataset "$qag_dataset-128-ip" --source "$HOME/vibe-hms/vibe/data/$qag_dataset-128-ip.hdf5" --native "$qag_task/data/$qag_dataset-full" --training "$qag_task/data/$qag_dataset-full-training" --config "$qag_task/configs/final-candidates/qag_pair/config.yml" --baseline "$qag_task/artifacts/baseline-holographic_memory.so" --candidate "$qag_task/artifacts/patience-holographic_memory.so" --out "$qag_task/freezes/$qag_dataset-full-validation-v1"
  uv run --no-project --python "$qag_python" python "$qag_task/qag_execute.py" run --freeze "$qag_task/freezes/$qag_dataset-full-validation-v1/frozen.json" --run-id "$qag_dataset-full-validation-v1"
done
for qag_dataset in llama yi; do
  uv run --no-project --python "$qag_python" python "$qag_task/qag_execute.py" freeze --phase official --dataset "$qag_dataset-128-ip" --source "$HOME/vibe-hms/vibe/data/$qag_dataset-128-ip.hdf5" --native "$qag_task/data/$qag_dataset-full" --training "$qag_task/data/$qag_dataset-full-training" --config "$qag_task/configs/final-candidates/qag_pair/config.yml" --baseline "$qag_task/artifacts/baseline-holographic_memory.so" --candidate "$qag_task/artifacts/patience-holographic_memory.so" --validation-report "$qag_task/runs/$qag_dataset-full-validation-v1/report.json" --out "$qag_task/freezes/$qag_dataset-official-v1"
  uv run --no-project --python "$qag_python" python "$qag_task/qag_execute.py" run --freeze "$qag_task/freezes/$qag_dataset-official-v1/frozen.json" --run-id "$qag_dataset-official-v1"
done
uv run --no-project --python "$qag_python" python "$qag_task/qag_regression.py" --run-id default4-v1 --candidate "$qag_task/artifacts/patience-holographic_memory.so"
uv run --no-project --python "$qag_python" python "$qag_task/qag_report.py" --results "$HOME/vibe-hms/vibe/results/qag-llama-128-ip/100/qag-projection-validation-v1/ip_64_500.hdf5" --data "$qag_task/data/qag-llama-128-ip.hdf5" --receipt "$qag_task/results/projection-validation-receipt.json" --out "$qag_task/results/projection-validation-summary-v2.json"
