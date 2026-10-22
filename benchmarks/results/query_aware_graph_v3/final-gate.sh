#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
qag_task="$HOME/sol-qag"
qag_out="$qag_task/final-gate"
qag_python="$HOME/vibe-hms/mamba/envs/vibe/bin/python"
test -f "$qag_task/v3/completion.json"
test ! -e "$qag_out"
test ! -e "$qag_task/repo-final-gate"
mkdir "$qag_out"
cp -a "$qag_task/repo-v3" "$qag_task/repo-final-gate"
uv run --no-project --python "$qag_python" python - "$qag_task" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
root = Path(sys.argv[1])
points = []
for dataset in ('llama', 'yi'):
    directory = root / 'v3/runs' / (dataset + '-kernel-v3')
    report_path = directory / 'report.json'
    report = json.loads(report_path.read_text())
    completion = json.loads((directory / 'completion.json').read_text())
    gate = json.loads((directory / 'gate.json').read_text())
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == completion['report']['sha256']
    assert gate['identical_returned_ids'] and report['source']['scope'] == 'validation'
    assert report['source']['queries'] == 256 and not report['source']['official_test']
    rows = {(r['arm'], r['ef'], r['rerank']): r for r in report['rows']}
    assert len(rows) == 6
    for ef, rr in ((100, 100), (256, 400), (512, 400)):
        a, b = rows[(0, ef, rr)], rows[(1, ef, rr)]
        aa, bb = [{r['repetition']: r for r in row['repetitions']} for row in (a, b)]
        assert set(aa) == set(bb) == {0, 1, 2}
        assert all(aa[i]['ids_sha256'] == bb[i]['ids_sha256'] for i in range(3))
        assert a['recall_ids_min'] == b['recall_ids_min']
        ratios = [bb[i]['qps'] / aa[i]['qps'] for i in range(3)]
        points.append({'dataset': dataset, 'ef': ef, 'rerank': rr,
                       'paired_ratios': ratios,
                       'separated_ranges': b['qps_min'] > a['qps_max'],
                       'accepted': min(ratios) > 1 and b['qps_min'] > a['qps_max']})
accept = all(point['accepted'] for point in points)
selected = 'repo-v3' if accept else 'repo-v2'
source = root / selected / 'src/core/qgraph/kernels.rs'
if not accept:
    shutil.copyfile(source, root / 'repo-final-gate/src/core/qgraph/kernels.rs')
decision = {'kernel_accepted': accept, 'selected_kernel_source': selected,
            'criterion': 'All six predeclared validation points faster in every observed paired block with separated QPS ranges; no new official evaluation',
            'kernel_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'points': points}
with (root / 'v3/decision.json').open('x') as out:
    json.dump(decision, out, indent=2); out.write('\n')
PY
export CARGO_TARGET_DIR="$qag_task/target-final-189" CARGO_BUILD_JOBS=2 RUSTUP_TOOLCHAIN=1.89.0 CARGO_TERM_COLOR=never
export CARGO_PROFILE_DEV_OPT_LEVEL=1 CARGO_PROFILE_TEST_OPT_LEVEL=1 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1
unset RUSTFLAGS HMS_QGRAPH_KERNEL HMS_QGRAPH_PREFETCH HMS_QGRAPH_BATCH_EST HMS_QGRAPH_FRONTIER HMS_QGRAPH_BLOCK_SCREEN
cd "$qag_task/repo-final-gate"
find src tests examples bindings assets docs -type f -print0 | sort -z | xargs -0 sha256sum > "$qag_out/source.sha256"
sha256sum Cargo.toml Cargo.lock build.rs > "$qag_out/manifests-before.sha256"
rustc +1.89.0 --version --verbose > "$qag_out/compiler.txt"
set +e
rustup component add rustfmt --toolchain 1.89.0 > "$qag_out/rustfmt-install.txt" 2>&1
cargo fmt --check > "$qag_out/fmt.txt" 2>&1
qag_fmt_rc=$?
printf '{"attempts":1,"command":"cargo +1.89.0 test --all-features"}\n' > "$qag_out/cargo-attempt.json"
cargo +1.89.0 test --all-features > "$qag_out/cargo.txt" 2>&1
qag_cargo_rc=$?
set -e
sha256sum Cargo.toml Cargo.lock build.rs > "$qag_out/manifests-after.sha256"
uv run --no-project --python "$qag_python" python - "$qag_out" "$qag_fmt_rc" "$qag_cargo_rc" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
checks = {}
for key, command, status in (('fmt', 'cargo fmt --check', int(sys.argv[2])),
                             ('cargo', 'cargo +1.89.0 test --all-features', int(sys.argv[3]))):
    checks[key] = {'command': command, 'status': 'pass' if status == 0 else 'fail',
                   'exit_code': status, 'log': 'final-gate/' + key + '.txt'}
checks['cargo']['attempts'] = 1
checks['environment'] = {'toolchain': '1.89.0', 'CARGO_BUILD_JOBS': '2',
                         'dev_test_opt_level': '1 (repository CI)', 'dev_test_debug': '0 (repository CI)',
                         'RUSTFLAGS': 'unset', 'target': str(Path.home() / 'sol-qag/target-final-189')}
with (root / 'checks.json').open('x') as out:
    json.dump(checks, out, indent=2); out.write('\n')
PY
exit $((qag_fmt_rc != 0 || qag_cargo_rc != 0))
