#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

qag_mlann_script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 HMS_QAG_OUTER_LOCK=1
exec flock "$HOME/hms-timing.lock" uv run --no-project \
  --python "$HOME/vibe-hms/mamba/envs/vibe/bin/python" -- \
  python "$qag_mlann_script_dir/qag_mlann_execute.py" "$@"
