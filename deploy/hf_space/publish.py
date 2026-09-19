"""Publish the SatQuery API to Hugging Face in one command.

Creates (or updates) a private model repo with the CPU weight set, then a
Docker Space that serves the API from it, with the Space's variables and
secret set. Idempotent: re-running updates in place.

Prerequisite: `hf auth login` with a WRITE token (huggingface.co/settings/tokens).

Usage (from the repo root)::

    python deploy/hf_space/publish.py --weights C:/Users/dk231/Desktop/SatQuery_AI/hf_stage/checkpoints
    python deploy/hf_space/publish.py --weights ~/satquery/.hf_stage/checkpoints --cors https://satquery-ai.vercel.app

The Space reads the weights with the same token it is given as the
HF_TOKEN secret; pass `--read-token` to hand it a separate read-only token
instead of the login token.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, get_token

HERE = Path(__file__).resolve().parent
EXPECTED = [
    "v2/grounding_pre", "v2/caption_pre", "v2/change_vqa", "change_caption",
    "v3/change_mask", "v3/optsar_fusion", "v3/landcover_full",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", type=Path, required=True, help="staged checkpoints/ directory (1.4 GB)")
    ap.add_argument("--weights-repo", default="satquery-cpu-weights")
    ap.add_argument("--space", default="satquery-api")
    ap.add_argument("--cors", default="https://satquery-ai-self.vercel.app", help="browser origin allowed to call the API")
    ap.add_argument("--git-ref", default="main", help="branch/tag of the GitHub repo the Space builds from")
    ap.add_argument("--read-token", default=None, help="read-only token for the Space secret (default: the login token)")
    ap.add_argument("--skip-weights", action="store_true", help="do not re-upload the weights")
    args = ap.parse_args()

    missing = [d for d in EXPECTED if not (args.weights / d).is_dir()]
    if missing:
        print("staged weights incomplete, missing: " + ", ".join(missing), file=sys.stderr)
        return 1
    token = get_token()
    if not token:
        print("not logged in: run `hf auth login` with a write token first", file=sys.stderr)
        return 1
    api = HfApi()
    user = api.whoami()["name"]
    weights_id = f"{user}/{args.weights_repo}"
    space_id = f"{user}/{args.space}"

    api.create_repo(weights_id, repo_type="model", private=True, exist_ok=True)
    if not args.skip_weights:
        print(f"uploading {args.weights} -> {weights_id}/checkpoints (1.4 GB, a few minutes)")
        api.upload_folder(repo_id=weights_id, repo_type="model", folder_path=str(args.weights),
                          path_in_repo="checkpoints", commit_message="SatQuery CPU weight set")
    files = api.list_repo_files(weights_id, repo_type="model")
    n_pt = sum(f.endswith(".pt") for f in files)
    print(f"weights repo {weights_id}: {len(files)} files, {n_pt} checkpoints")
    if n_pt < 7:
        print("expected 7 .pt files on the weights repo", file=sys.stderr)
        return 1

    api.create_repo(space_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    api.upload_folder(repo_id=space_id, repo_type="space", folder_path=str(HERE),
                      allow_patterns=["Dockerfile", "start.sh", "README.md"],
                      commit_message=f"SatQuery API, cpu profile ({args.git_ref})")
    api.add_space_secret(space_id, "HF_TOKEN", args.read_token or token)
    api.add_space_variable(space_id, "SATQUERY_WEIGHTS_REPO", weights_id)
    api.add_space_variable(space_id, "SATQUERY_CORS_ORIGINS", args.cors)
    api.add_space_variable(space_id, "SATQUERY_GIT_REF", args.git_ref)
    api.restart_space(space_id)
    print(f"space: https://huggingface.co/spaces/{space_id}")
    print(f"api:   https://{user}-{args.space}.hf.space   (builds ~10 min; then /health)")
    print(f"vercel NEXT_PUBLIC_API_URL = https://{user}-{args.space}.hf.space")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
