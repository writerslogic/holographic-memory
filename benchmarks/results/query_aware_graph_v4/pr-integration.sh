#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
qag_repo="$1"
qag_out="$2"
export CARGO_TARGET_DIR="$3"
qag_python="$4"
mkdir -p "$qag_out"
cd "$qag_repo"
export CARGO_BUILD_JOBS=2 CARGO_TERM_COLOR=never
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1
export CARGO_PROFILE_DEV_OPT_LEVEL=1 CARGO_PROFILE_TEST_OPT_LEVEL=1 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
cargo +1.96.0 test --locked --all-features > "$qag_out/all-features-tests.log" 2>&1
cargo +1.96.0 build --locked --features experimental,python --lib > "$qag_out/python-build.log" 2>&1
cp "$CARGO_TARGET_DIR/debug/libholographic_memory.so" "$qag_out/holographic_memory.so"
export QAG_PR_REPO="$qag_repo" QAG_PR_EXTENSION="$qag_out/holographic_memory.so"
uv run --no-project --python "$qag_python" python - <<'PY' > "$qag_out/python-tests.log" 2>&1
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path.home() / "vibe-hms/vibe"))
from vibe.algorithms.qag_pair.module import _load_extension

sys.modules["holographic_memory"] = _load_extension(
    Path(os.environ["QAG_PR_EXTENSION"]), "_hms_qag_pr_tests"
)
import pytest

raise SystemExit(pytest.main([
    "-q", str(Path(os.environ["QAG_PR_REPO"]) / "tests/python/test_vector_index.py")
]))
PY
sha256sum "$qag_out/holographic_memory.so" > "$qag_out/python-extension.sha256"
