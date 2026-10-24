#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$HOME/vibe-image-ab/verification-v1"
if [[ ${1:-} != --locked ]]; then
  exec 9>"$HOME/hms-timing.lock"
  flock -n 9
  date -u +LOCK_ACQUIRED=%FT%TZ
  set +e
  timeout --signal=TERM --kill-after=2s 238s bash run.sh --locked
  result=$?
  set -e
  date -u +LOCK_RELEASE=%FT%TZ
  echo "EXIT_CODE=$result"
  exit "$result"
fi
date -u > host.txt
uname -a >> host.txt
lscpu >> host.txt
cat /proc/loadavg >> host.txt
ps -eo pid,ppid,comm,args --sort=-pcpu > processes-before.txt
uv run --offline --no-project python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

if os.getloadavg()[0] >= 3:
    raise RuntimeError("load gate failed before source verification")
root = Path.home() / "vibe-image-ab"
expected = json.loads(Path("source-manifest.json").read_text())
actual = {}
for name, digest in expected["rust_source_sha256"].items():
    actual[name] = hashlib.sha256((root / "src" / name).read_bytes()).hexdigest()
    assert actual[name] == digest, name
files = ("src/Cargo.toml", "src/Cargo.lock", "merged.profdata", "train.json")
provenance = dict(source=expected["source"], source_files_checked=len(actual),
                  source_matches=True, source_manifest_sha256=hashlib.sha256(
                      Path("source-manifest.json").read_bytes()).hexdigest(),
                  file_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in files})
Path("provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
PY
uv run --offline --no-project python paired.py \
  --root "$HOME/vibe-image-ab" \
  --glove "$HOME/.cache/hms-bench/glove-100-angular" \
  --sift "$HOME/hms-bench-decisions/data/sift-128-euclidean" \
  --out "$HOME/vibe-image-ab/verification-v1/run"
ps -eo pid,ppid,comm,args --sort=-pcpu > processes-after.txt
