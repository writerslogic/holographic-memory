#!/bin/bash
# One locked VIBE run (local mode) per call: bench.sh <dataset> <parallelism> <env> <defs-folder> [lock|nolock]
D=$1; P=$2; E=$3; F=$4; L=${5:-lock}
export MAMBA_ROOT_PREFIX=~/vibe-hms/mamba NUMBA_CACHE_DIR=/tmp/numba-vibe
for v in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS TBB_NUM_THREADS RAYON_NUM_THREADS NUMBA_NUM_THREADS VECLIB_MAXIMUM_THREADS NUMEXPR_NUM_THREADS; do export $v=1; done
export OMP_DYNAMIC=false MKL_DYNAMIC=false
cd ~/vibe-hms/vibe
LOG=~/vibe-hms/logs/$D-$(basename $F).log
s=$(date +%s)
CMD=(~/vibe-hms/bin/micromamba run -n $E python run.py --dataset $D --definitions $F --local --parallelism $P --timeout 7200)
if [ "$L" = lock ]; then
  W=$(( $(date -d 03:10 +%s) - $(date +%s) ))
  [ $W -le 0 ] && { echo "$D $F skipped: lock deadline passed"; exit 0; }
  flock -w $W ~/hms-timing.lock bash -c 'echo "acquired $(date +%T) $(uptime)"; "$@"; r=$?; echo "released $(date +%T) $(uptime)"; exit $r' _ "${CMD[@]}" > $LOG 2>&1
else
  "${CMD[@]}" > $LOG 2>&1
fi
echo "$D $F rc=$? $(( $(date +%s)-s ))s $(date +%T)"
