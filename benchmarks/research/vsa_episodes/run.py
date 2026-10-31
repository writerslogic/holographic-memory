"""Run the frozen, local-only admission experiment after offline compilation."""
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / 'benchmarks/results/vsa_episodes_v1'
DEV = '--dev' in sys.argv
DEST = OUT / ('development' if DEV else 'evaluation')
DEST.mkdir(parents=True, exist_ok=True)
if (DEST / 'results.json').exists():
    raise SystemExit('Results already exist; preserve them instead of silently rerunning.')
files = [
    'benchmarks/results/vsa_episodes_v1/PLAN.json',
    'benchmarks/research/vsa_episodes/src/main.rs',
    'benchmarks/research/vsa_episodes/Cargo.toml',
    'benchmarks/research/vsa_episodes/Cargo.lock',
    'benchmarks/research/vsa_episodes/run.py',
    'target/vsa-episodes-eval/release/hms-vsa-episodes-eval',
    'src/core/phase_hvec.rs',
    'benchmarks/research/contrast/src/hv.rs',
    'benchmarks/research/contrast/src/bm25.rs',
]
manifest = {
    'sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
    'platform': platform.platform(),
    'machine': platform.machine(),
    'hardware': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
    'memory_bytes': int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)),
    'rustc': subprocess.check_output(['rustc', '+1.99.0', '-Vv'], text=True),
    'development': DEV,
    'execution_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'isolation': 'Single process/thread; other host workloads not controlled. Microsecond timings are indicative, not production SLAs.',
    'started_unix': time.time(),
}
(DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
cmd = ['/usr/bin/time', '-l', str(ROOT / 'target/vsa-episodes-eval/release/hms-vsa-episodes-eval')]
if DEV:
    cmd.append('--dev')
cmd.append(str(DEST))
with (DEST / 'execution.log').open('w') as log:
    try:
        result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=log, timeout=600, check=False)
        manifest['exit_code'] = result.returncode
    except subprocess.TimeoutExpired:
        manifest['timed_out'] = True
        manifest['exit_code'] = 124
manifest['elapsed_seconds'] = time.time() - manifest['started_unix']
(DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
raise SystemExit(manifest['exit_code'])
