#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$HOME/hms-streaming-alloc"
step=${1:?build, prepare or profile required}
case "$step" in build|prepare|profile) ;; *) exit 2 ;; esac
mkdir -p evidence
exec 9>"$HOME/hms-timing.lock"
flock -w 600 9
unit="hms-alloc-$step.scope"
cleanup() {
    status=$?
    if systemctl --user --quiet is-active "$unit"; then systemctl --user stop "$unit"; fi
    printf 'exit_status=%s\n' "$status"
    date -u
}
trap cleanup EXIT
printf 'lock_acquired step=%s\n' "$step"
date -u
cat /proc/loadavg
if pgrep -x rustc >/dev/null || pgrep -x cargo >/dev/null; then
    printf 'Unexpected compiler process while holding timing lock.\n' >&2
    exit 75
fi
available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
minimum=6291456
if [[ "$step" == build ]]; then minimum=10485760; fi
(( available >= minimum ))
export RAYON_NUM_THREADS=8
case "$step" in
build)
    timeout --kill-after=5s 800 systemd-run --user --scope -q --unit="$unit" \
        -p RuntimeMaxSec=790s -p MemoryMax=8G -p MemorySwapMax=0 bash build-pair.sh
    ;;
prepare)
    test ! -e fixture
    /usr/bin/time -v timeout --kill-after=5s 800 systemd-run --user --scope -q --unit="$unit" \
        -p RuntimeMaxSec=790s -p MemoryMax=4G -p MemorySwapMax=0 \
        taskset -c 0-7 ./parent-profile --synthetic 240000 --dim 100 --metric euclidean \
        --degree 32 --build-ef 64 --prepare-streaming fixture
    timeout --kill-after=2s 20 uv run --no-project python hash-files.py fixture > evidence/fixture-sha256.json
    ;;
profile)
    test ! -e profile-work
    /usr/bin/time -v timeout --kill-after=5s 800 systemd-run --user --scope -q --unit="$unit" \
        -p RuntimeMaxSec=790s -p MemoryMax=4G -p MemorySwapMax=0 \
        bash -c 'set -euo pipefail; cp -a --reflink=never fixture profile-work; taskset -c 0-7 ./parent-profile --consolidate profile-work'
    timeout --kill-after=2s 20 uv run --no-project python hash-files.py profile-work > evidence/profile-files-sha256.json
    ;;
esac
