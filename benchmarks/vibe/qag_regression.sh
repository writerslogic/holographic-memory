#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

qag_regression_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1
export WARM_QAG_OUTER_LOCK=1
exec flock "$HOME/warm-timing.lock" uv run --no-project \
  --python "$HOME/vibe-warm/mamba/envs/vibe/bin/python" -- \
  python "$qag_regression_dir/qag_regression.py" "$@"
