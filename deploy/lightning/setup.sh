#!/usr/bin/env bash
# One-time setup of a Lightning AI Studio (or any Linux box with >= 12 GB RAM)
# as the SatQuery API host. CPU is enough: the full Phase 6 stack runs on
# CPU with the VLM in bf16 (docs/deploy-free.md has the measured times); on a
# GPU Studio the same script serves the 4-bit path automatically.
#
#   export HF_TOKEN=hf_...            # read token for the private weights repo
#   bash deploy/lightning/setup.sh    # from the repo root, once
#   bash deploy/lightning/serve.sh    # each start (or as the Studio's startup command)
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${HF_TOKEN:?export HF_TOKEN (a Hugging Face read token) first}"
WEIGHTS_REPO="${SATQUERY_WEIGHTS_REPO:-DeepakShivhareEe/satquery-cpu-weights}"

echo "== python packages"
pip install -q --timeout 120 --retries 5 -r requirements.txt \
  "transformers==5.15.1" "peft==0.20.0" "accelerate==1.14.0" "timm==1.0.29" "huggingface_hub>=0.30"
if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  pip install -q "bitsandbytes==0.50.2"          # 4-bit path on a GPU Studio
else
  python -c "import torch" 2>/dev/null || pip install -q torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
fi

echo "== weights (base 7 GB from the Hub, checkpoints 1.9 GB from $WEIGHTS_REPO)"
python - <<PY
import os
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct", local_dir="models/qwen25_vl_3b", token=os.environ["HF_TOKEN"],
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
snapshot_download(os.environ.get("SATQUERY_WEIGHTS_REPO", "$WEIGHTS_REPO"), repo_type="model", local_dir=".",
                  token=os.environ["HF_TOKEN"], allow_patterns=["checkpoints/**"])
PY
find checkpoints -type f | wc -l | xargs echo "checkpoint files:"

echo "== load check (all 8 tools)"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/verify_deploy.py --map configs/deploy.v3.yaml
echo "setup complete - start with: bash deploy/lightning/serve.sh"
