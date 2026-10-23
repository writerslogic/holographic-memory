#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
qag_task="$HOME/sol-qag"
qag_out="$qag_task/v4"
qag_python="$HOME/vibe-hms/mamba/envs/vibe/bin/python"
mkdir -p "$qag_out/logs" "$qag_out/diagnostics"
export CARGO_TARGET_DIR="$qag_task/target-v4" CARGO_BUILD_JOBS=2 RUSTFLAGS='-C target-cpu=native'
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1
unset HMS_QGRAPH_PREFETCH HMS_QGRAPH_BATCH_EST HMS_QGRAPH_FRONTIER HMS_QGRAPH_BLOCK_SCREEN HMS_QGRAPH_KERNEL
export HMS_QGRAPH_HUGEPAGES=0
cd "$qag_task/repo-v4"
test ! -e "$qag_out/compiler.txt"
rustc +1.96.0 --version --verbose > "$qag_out/compiler.txt"
sha256sum Cargo.toml Cargo.lock src/bin/qag-diagnose.rs src/python/vector_index.rs src/core/qgraph/{mod.rs,vertex.rs,vertex/file.rs,update.rs,diagnostics.rs,projection.rs,kernels.rs} > "$qag_out/source.sha256"
cargo +1.96.0 build --release --locked --features experimental,python --lib --bin qag-diagnose > "$qag_out/logs/build.log" 2>&1
cp "$CARGO_TARGET_DIR/release/libholographic_memory.so" "$qag_out/holographic_memory.so"
sha256sum "$qag_out/holographic_memory.so" "$CARGO_TARGET_DIR/release/qag-diagnose" > "$qag_out/binaries.sha256"
uv run --no-project --python "$qag_python" python - "$qag_out/holographic_memory.so" "$PWD/tests/python/test_vector_index.py" > "$qag_out/logs/python-tests.log" 2>&1 <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / 'vibe-hms/vibe'))
from vibe.algorithms.qag_pair.module import _load_extension
sys.modules['holographic_memory'] = _load_extension(Path(sys.argv[1]), '_hms_qag_v4_tests')
import pytest
raise SystemExit(pytest.main(['-q', sys.argv[2]]))
PY
uv run --no-project --python "$qag_python" python - "$qag_task" > "$qag_out/logs/input-verification.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
verified = []
for dataset in ('llama', 'yi'):
    for name in (dataset + '-full', dataset + '-full-training'):
        folder = root / 'data' / name
        meta = json.loads((folder / 'meta.json').read_text())
        assert meta['official_test_read'] is False
        for file, expected in meta['files'].items():
            assert Path(file).name == file
            digest = hashlib.file_digest((folder / file).open('rb'), 'sha256').hexdigest()
            assert digest == expected, (name, file)
        verified.append({'dataset': name, 'files': meta['files']})
print(json.dumps(verified, indent=2))
PY
for qag_dataset in llama yi; do
  "$CARGO_TARGET_DIR/release/qag-diagnose" --data "$qag_task/data/$qag_dataset-full" --graph "$qag_task/data/$qag_dataset-full.graph" --out "$qag_out/diagnostics/$qag_dataset.json" --bits 8 --training "$qag_task/data/$qag_dataset-full-training" --training-count 32768 --projected-degree 32 --compare-serving > "$qag_out/logs/diagnose-$qag_dataset.log" 2>&1
done
