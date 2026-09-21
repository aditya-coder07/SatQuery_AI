#!/usr/bin/env bash
# Shared preamble for per-job cluster units (source it after setting LOG and
# MIN_FREE_GB). Sizes every launch from FREE VRAM on the shared L40S, never
# total, and serialises launches with a lock: a job holds
# logs/.gpu_launch.lock for LAUNCH_HOLD_S seconds after it starts, so a
# second unit cannot pass its own VRAM check while the first is still
# loading weights (three VLM jobs passed the check within 30 s of each other
# on 2026-09-14 10:12 and two of them OOM'd).
set -u
# Overridable (2026-09-21): the dev01 account rebuilt the tree under
# ~/satquery/repo with its own env and a Hub copy of the base model.
cd "${SATQUERY_ROOT:-/scratch/home/adi01/satquery}"
PY=${SATQUERY_PY:-/scratch/home/adi01/satquery/.conda-env/bin/python}
BASE=${SATQUERY_BASE:-models/qwen25_vl_3b}
mkdir -p logs
LAUNCH_HOLD_S=${LAUNCH_HOLD_S:-300}
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
free_gb() {
  local used total
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  echo $(( (total - used) / 1024 ))
}
run() {  # run <name> <command...>: wait for VRAM under the launch lock, hold it while the job ramps
  local name=$1; shift
  exec 9>logs/.gpu_launch.lock
  flock 9
  while [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; do
    log "waiting for VRAM: $(free_gb) GB free < ${MIN_FREE_GB} GB"; flock -u 9; sleep 120; flock 9
  done
  log "START $name"
  "$@" >> "logs/$name.log" 2>&1 &
  local pid=$!
  sleep "$LAUNCH_HOLD_S"
  flock -u 9
  if wait "$pid"; then log "DONE $name (exit 0)"; else log "FAILED $name (exit $?)"; fi
}
