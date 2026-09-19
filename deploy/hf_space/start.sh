#!/usr/bin/env bash
# Download the CPU weight set, then serve the API on the Space port.
set -euo pipefail
cd /app
: "${SATQUERY_WEIGHTS_REPO:?set the SATQUERY_WEIGHTS_REPO Space variable, e.g. <user>/satquery-cpu-weights}"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id=os.environ["SATQUERY_WEIGHTS_REPO"], repo_type="model",
    token=os.environ.get("HF_TOKEN"), local_dir="/app", allow_patterns=["checkpoints/**"])
print("weights at", path)
PY
find /app/checkpoints -type f | sed 's#^/app/##' | sort
exec python scripts/serve_local.py --map configs/deploy.cpu.yaml --profile cpu --host 0.0.0.0 --port 7860
