#!/usr/bin/env bash
# Pull Phase 6 results from the cluster into the repo: per-run metrics and
# metadata (never weights), benchmark reports, and the cluster's registry
# records (merged by experiment_id, append-only).
#
#   bash scripts/sync_cluster_results.sh [run ...]
set -euo pipefail
HOST=adi01@172.16.1.161
KEY=~/.ssh/ailab
SSH="ssh -q -i $KEY -o BatchMode=yes"
SCP="scp -q -i $KEY -o BatchMode=yes"
DEST=docs/assets/phase6
mkdir -p "$DEST" artifacts/benchmark_reports artifacts/experiment_registry

runs=("$@")
if [ ${#runs[@]} -eq 0 ]; then
  runs=($($SSH $HOST 'ls ~/satquery/checkpoints/v3'))
fi
for r in "${runs[@]}"; do
  mkdir -p "$DEST/$r"
  for f in metrics.json run_metadata.json val_history.json; do
    $SCP "$HOST:~/satquery/checkpoints/v3/$r/$f" "$DEST/$r/$f" 2>/dev/null || true
  done
  echo "synced $r: $(ls "$DEST/$r" | tr '\n' ' ')"
done

# Benchmark reports (json only; prediction dumps stay on the cluster).
$SSH $HOST 'ls ~/satquery/artifacts/benchmark_reports/*.json 2>/dev/null' | while read -r f; do
  $SCP "$HOST:$f" artifacts/benchmark_reports/
done

# Registry merge.
$SCP "$HOST:~/satquery/artifacts/experiment_registry/registry.jsonl" artifacts/experiment_registry/.cluster_registry.jsonl 2>/dev/null || touch artifacts/experiment_registry/.cluster_registry.jsonl
python - <<'PY'
import json
from pathlib import Path
local = Path("artifacts/experiment_registry/registry.jsonl")
have = set()
if local.exists():
    for line in local.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            have.add((rec["experiment_id"], rec.get("status"), rec.get("timestamp")))
new = 0
with local.open("a", encoding="utf-8") as out:
    for line in Path("artifacts/experiment_registry/.cluster_registry.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        key = (rec["experiment_id"], rec.get("status"), rec.get("timestamp"))
        if key not in have:
            out.write(json.dumps(rec, sort_keys=True) + "\n")
            have.add(key)
            new += 1
print(f"registry: {new} new records merged")
PY
