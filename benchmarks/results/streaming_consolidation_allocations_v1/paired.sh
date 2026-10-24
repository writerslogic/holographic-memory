#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$HOME/hms-streaming-alloc"
round=${1:?round required}
[[ "$round" == 1 || "$round" == 2 || "$round" == 3 ]]
exec 9>"$HOME/hms-timing.lock"
flock -w 600 9
started=$SECONDS
export RAYON_NUM_THREADS=8
arms=(parent candidate)
if [[ "$round" == 2 ]]; then arms=(candidate parent); fi
for arm in "${arms[@]}"; do
    label="$round-$arm"
    test ! -e "work-$label"
    available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    (( available >= 6291456 ))
    remaining=$((820 - (SECONDS - started)))
    (( remaining > 20 ))
    limit=400
    if (( remaining < limit )); then limit=$remaining; fi
    unit="hms-alloc-$label.scope"
    cleanup() { if systemctl --user --quiet is-active "$unit"; then systemctl --user stop "$unit"; fi; }
    trap cleanup EXIT
    {
      date -u
      cat /proc/loadavg
      /usr/bin/time -v timeout --kill-after=10s "$limit" systemd-run --user --scope -q \
        --unit="$unit" -p RuntimeMaxSec="$((limit-10))s" -p MemoryMax=4G -p MemorySwapMax=0 \
        bash -c 'set -euo pipefail
          cp -a --reflink=never fixture "work-$1"
          taskset -c 0-7 "./$2-run" "work-$1"
          uv run --no-project python hash-files.py "work-$1"' _ "$label" "$arm"
      cat /proc/loadavg
      date -u
    } > "evidence/$label.log" 2>&1
done
