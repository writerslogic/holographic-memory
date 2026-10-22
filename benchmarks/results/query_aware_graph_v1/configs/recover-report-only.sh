#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
qag_task="$HOME/sol-qag"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1
test ! -e "$qag_task/results/projection-validation-summary-v2.json"
uv run --no-project --python "$HOME/vibe-hms/mamba/envs/vibe/bin/python" python "$qag_task/qag_report.py" --results "$HOME/vibe-hms/vibe/results/qag-llama-128-ip/100/qag-projection-validation-v1/ip_64_500.hdf5" --data "$qag_task/data/qag-llama-128-ip.hdf5" --receipt "$qag_task/results/projection-validation-receipt.json" --out "$qag_task/results/projection-validation-summary-v2.json"
sha256sum "$qag_task/v2/profiles/llama-persisted/fixed.hms" "$qag_task/v2/profiles/llama-persisted/compact.hms" "$qag_task/v2/profiles/yi-persisted/fixed.hms" "$qag_task/v2/profiles/yi-persisted/compact.hms" > "$qag_task/v2/persisted.sha256"
