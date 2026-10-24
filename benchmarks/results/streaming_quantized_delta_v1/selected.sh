#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/hms-streaming-quantized"
token=${1:?unique token required}
[[ "$token" =~ ^[a-zA-Z0-9_-]+$ ]]
rows=${2:?row count required}
first=${3:?query offset required}
[[ "$rows" == 1000000 || "$rows" == 3000000 ]]
[[ "$first" == 256 || "$first" == 384 ]]
[[ "$rows:$first" != 3000000:256 ]]
cap=3G
minimum_kib=4194304
if [[ "$rows" == 3000000 ]]; then cap=6G; minimum_kib=8388608; fi
exec 9>"$HOME/hms-timing.lock"
flock -w 600 9 || { printf 'LOCK_TIMEOUT %s\n' "$token"; exit 75; }
available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_kib < minimum_kib )); then
  printf 'MEMORY_LOW %s %s\n' "$token" "$available_kib"
  exit 76
fi
printf 'START %s available_kib=%s\n' "$token" "$available_kib"
export RAYON_NUM_THREADS=8
export HMS_DELTA_ROWS="$rows"
export HMS_DELTA_QUERY_FIRST="$first"
export HMS_DELTA_DATA="$HOME/hms-streaming-peak/datasets/msturing-30m-prefix-3m.fbin"
export HMS_DELTA_QUERIES="$HOME/hms-streaming-peak/datasets/msturing-30m-queries.fbin"
unit="hms-quantized-$token.scope"
set +e
{
  date -u
  cat /proc/loadavg
  cat production-commit.txt
  sha256sum grid-test source/src/core/qgraph/update/delta.rs source/src/core/qgraph/update/delta_codes.rs source/src/core/qgraph/update/delta_quantized_tests.rs
  /usr/bin/time -v timeout --kill-after=5s 225 systemd-run --user --scope --unit="$unit" -q \
    -p MemoryMax="$cap" -p MemorySwapMax=0 taskset -c 0-7 \
    ./grid-test quantized_delta_selected --ignored --nocapture --test-threads=1
  status=$?
  scope_state=$(systemctl --user show "$unit" -p ActiveState --value 2>/dev/null)
  if [[ -n "$scope_state" && "$scope_state" != inactive && "$scope_state" != failed ]]; then
    systemctl --user stop "$unit" || status=77
  fi
  group=$(systemctl --user show "$unit" -p ControlGroup --value 2>/dev/null)
  if [[ -n "$group" && -f "/sys/fs/cgroup$group/cgroup.procs" ]] && [[ -n $(cat "/sys/fs/cgroup$group/cgroup.procs") ]]; then
    printf 'SCOPE_NOT_EMPTY %s\n' "$unit"
    status=78
  else
    printf 'SCOPE_EMPTY %s\n' "$unit"
  fi
  cat /proc/loadavg
  date -u
  printf 'EXIT_STATUS %s\n' "$status"
} >"$token.tune.log" 2>&1
printf 'DONE %s %s\n' "$token" "$status" >"$token.done"
exit "$status"
