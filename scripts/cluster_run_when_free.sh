#!/usr/bin/env bash
# Run one command once at least $1 GB of VRAM is free on the shared L40S.
#   bash scripts/cluster_run_when_free.sh <min_free_gb> <log_name> <command...>
# Sizes from free VRAM, never total; never touches other users' processes.
set -u
cd /scratch/home/adi01/satquery
min=$1; name=$2; shift 2
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  free=$(( (total - used) / 1024 ))
  [ "$free" -ge "$min" ] && break
  echo "$(date '+%F %T') waiting: ${free} GB free < ${min}" >> "logs/$name.log"
  sleep 120
done
echo "$(date '+%F %T') START $*" >> "logs/$name.log"
"$@" >> "logs/$name.log" 2>&1
rc=$?  # capture before the date substitution below resets $?
echo "$(date '+%F %T') EXIT $rc" >> "logs/$name.log"
