#!/usr/bin/env bash
# Start the SatQuery API on this machine (Lightning Studio: expose port 8000
# publicly in the Studio's port panel; the URL is stable per Studio).
#
#   SATQUERY_CORS_ORIGINS   browser origins allowed (default: the Vercel site + localhost)
#   SATQUERY_THREADS        torch CPU threads (default: all cores)
#   SATQUERY_GROUNDING_MIN_PIXELS  1048576 is the measured budget; 262144 is ~4x faster on CPU at some accuracy cost
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SATQUERY_CORS_ORIGINS="${SATQUERY_CORS_ORIGINS:-https://satquery-ai-self.vercel.app,https://satquery-ai.vercel.app,http://localhost:3000}"
export OMP_NUM_THREADS="${SATQUERY_THREADS:-$(nproc)}"
exec python scripts/serve_local.py --map configs/deploy.v3.yaml --profile full --host 0.0.0.0 --port "${PORT:-8000}"
