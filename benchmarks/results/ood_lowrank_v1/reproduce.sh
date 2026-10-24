#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
: "${HMS_SOURCE:?Path to an isolated checkout of ac235724967cf6c7046ba3c02882a6046d37d075}"
: "${HMS_YI_HDF5:?Path to the yi-128-ip HDF5 dataset}"
: "${HMS_OUTPUT:?Path to a new result JSON file}"
: "${CARGO_TARGET_DIR:?Dedicated target directory}"
artifacts=$(cd "$(dirname "$0")" && pwd)
test "$(git -C "$HMS_SOURCE" rev-parse HEAD)" = ac235724967cf6c7046ba3c02882a6046d37d075
test "$(git -C "$HMS_SOURCE" branch --show-current)" != main
git -C "$HMS_SOURCE" diff --quiet
git -C "$HMS_SOURCE" apply --unidiff-zero --check "$artifacts/prototype.patch"
git -C "$HMS_SOURCE" apply --unidiff-zero "$artifacts/prototype.patch"
cd "$HMS_SOURCE"
export RUSTFLAGS='-C target-cpu=x86-64-v3'
export CARGO_BUILD_JOBS=2
uv run --no-project --python 3.11.16 --with maturin maturin build --locked --release --features python,experimental --out "$CARGO_TARGET_DIR/wheels"
uv venv --python 3.11.16 "$CARGO_TARGET_DIR/experiment-env"
uv pip install --python "$CARGO_TARGET_DIR/experiment-env/bin/python" "$CARGO_TARGET_DIR"/wheels/holographic_vsa-*.whl numpy==2.4.4 h5py==3.16.0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=2
nice -n 19 taskset -c 0,1 uv run --no-project "$CARGO_TARGET_DIR/experiment-env/bin/python" "$artifacts/../../vibe/ood_ivf_ab.py" "$HMS_YI_HDF5" "$HMS_OUTPUT" --source eb32e65267b40419d13d6f75eee5465e937ca95d --build-threads 2
