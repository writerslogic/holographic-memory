#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 HMS_QAG_OUTER_LOCK=1
qag_task="$HOME/sol-qag"
for qag_dataset in llama yi; do
  uv run --no-project --python "$HOME/vibe-hms/mamba/envs/vibe/bin/python" python "$qag_task/qag_mlann_execute.py" --view "$qag_task/runs/$qag_dataset-official-v1/evaluation.hdf5" --training-queries "$qag_task/data/$qag_dataset-full-training/queries.f32" --baseline "$qag_task/artifacts/baseline-holographic_memory.so" --candidate "$qag_task/artifacts/patience-holographic_memory.so" --config "$qag_task/mlann_pair/config.yml" --freeze "$qag_task/mlann_pair/frozen.json" --run-id "$qag_dataset-rf80-v1" --out "$qag_task/competitors/mlann-rf80/$qag_dataset-128-ip"
done
