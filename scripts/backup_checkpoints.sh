#!/usr/bin/env bash
# Keep every Phase 6 checkpoint in more than one place, verified by sha256.
#
#   bash scripts/backup_checkpoints.sh [--local] [run ...]   # default: every run under checkpoints/v3
#
# Two tiers, because the college wifi moves ~0.1-0.5 MB/s downstream:
#   1. always: a second copy on the cluster's own filesystem
#      (/scratch/home/adi01/satquery_backup), best/final artefacts only;
#   2. with --local: the same artefacts mirrored to this workstation
#      (adapters are ~60 MB; CNN best.pt 100-300 MB), one file at a time.
# Digests of every backed-up file are appended to
# artifacts/best_models/checksums.tsv (tracked in git; weights never are).
set -euo pipefail
HOST=adi01@172.16.1.161
KEY=~/.ssh/ailab
SSH="ssh -q -i $KEY -o BatchMode=yes"
REMOTE=/scratch/home/adi01/satquery/checkpoints/v3
BACKUP=/scratch/home/adi01/satquery_backup/checkpoints/v3
DEST=checkpoints/v3
LOCAL=0
if [ "${1:-}" = "--local" ]; then LOCAL=1; shift; fi
runs=("$@")
if [ ${#runs[@]} -eq 0 ]; then runs=($($SSH $HOST "ls $REMOTE")); fi
mkdir -p "$DEST" artifacts/best_models

for r in "${runs[@]}"; do
  echo "== $r"
  # Tier 1: on-cluster copy + digest list of the kept artefacts.
  $SSH $HOST "set -e; mkdir -p $BACKUP/$r; cd $REMOTE/$r
    files=\$(find . -maxdepth 2 -type f \( -name 'best.pt' -o -name 'final.pt' -o -name 'last.pt' -o -name '*.json' -o -path './adapter_best/*' -o -path './adapter_final/*' \) | sort)
    for f in \$files; do mkdir -p $BACKUP/$r/\$(dirname \$f); cp -u \$f $BACKUP/$r/\$f; done
    (cd $BACKUP/$r && sha256sum \$files)" > "/tmp/digests_$r.sha"
  n=$(wc -l < "/tmp/digests_$r.sha")
  echo "  cluster copy: $n files"
  while read -r sum path; do
    printf '%s\t%s\t%s\t%s\n' "$(date +%F)" "$r" "${path#./}" "$sum" >> artifacts/best_models/checksums.tsv
  done < "/tmp/digests_$r.sha"
  # Tier 2: local mirror, verified.
  if [ "$LOCAL" = 1 ]; then
    ok=0; bad=0
    while read -r sum path; do
      rel=${path#./}
      mkdir -p "$DEST/$r/$(dirname "$rel")"
      if [ ! -f "$DEST/$r/$rel" ] || [ "$(sha256sum "$DEST/$r/$rel" | cut -d' ' -f1)" != "$sum" ]; then
        scp -q -i $KEY -o BatchMode=yes "$HOST:$REMOTE/$r/$rel" "$DEST/$r/$rel"
      fi
      if [ "$(sha256sum "$DEST/$r/$rel" | cut -d' ' -f1)" = "$sum" ]; then ok=$((ok+1)); else echo "  MISMATCH $rel"; bad=$((bad+1)); fi
    done < "/tmp/digests_$r.sha"
    echo "  local mirror: $ok verified, $bad problems"
  fi
done
sort -u -o artifacts/best_models/checksums.tsv artifacts/best_models/checksums.tsv
