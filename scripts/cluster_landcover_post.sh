#!/usr/bin/env bash
# After the full-BigEarthNet land-cover run: verify the run, then fit the
# v3 calibration for SINGLE_LANDCOVER on the official VAL split (never
# test), derive the assertion threshold from those calibrated val logits,
# and load the checkpoint through the deployed tool loader. Read-only with
# respect to the checkpoint; writes docs/assets/calibration_v3/*,
# configs/calibration.v3.json (merged), artifacts/calibration_v3/logits/
# landcover.npz and docs/assets/calibration_v3/assertion_threshold_landcover.json.
# Logged in docs/research/queue_ledger.md as job #17.
set -u
cd /scratch/home/adi01/satquery
PY=/scratch/home/adi01/satquery/.conda-env/bin/python
LOG=logs/landcover_post.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

while systemctl --user is-active --quiet sq-train-landcover-v3-full; do sleep 120; done
log "landcover full unit finished"
grep -q "EXIT 0" logs/train_landcover_v3_full.log || { log "GATE FAIL: train_landcover_v3_full did not exit 0"; exit 1; }
CK=checkpoints/v3/landcover_full
for f in best.pt final.pt metrics.json band_stats.json; do [ -f "$CK/$f" ] || { log "GATE FAIL: $CK/$f missing"; exit 1; }; done
$PY - <<'PYEOF' >> "$LOG" 2>&1 || { log "GATE FAIL: metrics.json incomplete"; exit 1; }
import json
m = json.load(open("checkpoints/v3/landcover_full/metrics.json"))
assert m["n_test"] == 125866 and m["test_at_best_val"] is not None, (m["n_test"], m.get("test_at_best_val"))
t = m["test_at_best_val"]
print(f"GATE OK: test@best-val macro mAP {t['map_all_bands']:.4f} micro {t['map_micro_all_bands']:.4f} "
      f"retention {t['retention']:.4f} (epoch {t['epoch']}); final macro {m['map_all_bands']:.4f}")
PYEOF

# Wait for a little VRAM (inference only, ~3 GB).
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  [ $(( (total - used) / 1024 )) -ge 6 ] && break
  log "waiting for VRAM"; sleep 120
done

log "START calibrate landcover (val)"
$PY evaluation/calibrate.py --heads landcover --ben-data data/ben_v1_full --ben-split val --ben-limit 20000 \
  --track-a-ckpt $CK/best.pt --batch-size 128 --out-dir docs/assets/calibration_v3 \
  --registry configs/calibration.v3.json --cache-dir artifacts/calibration_v3/logits --refresh-cache >> "$LOG" 2>&1 \
  && log "DONE calibrate (exit 0)" || { log "FAILED calibrate (exit $?)"; exit 1; }

log "START assertion threshold"
$PY evaluation/assertion_threshold.py --logits artifacts/calibration_v3/logits/landcover.npz \
  --calibration configs/calibration.v3.json --target-precision 0.90 --min-asserted 50 \
  --out docs/assets/calibration_v3/assertion_threshold_landcover.json >> "$LOG" 2>&1 \
  && log "DONE assertion threshold (exit 0)" || { log "FAILED assertion threshold (exit $?)"; exit 1; }

log "START verify_deploy landcover"
$PY scripts/verify_deploy.py --map configs/deploy.v3.yaml --only landcover >> "$LOG" 2>&1 \
  && log "DONE verify_deploy (exit 0)" || { log "FAILED verify_deploy (exit $?)"; exit 1; }
log "landcover post-processing finished"
