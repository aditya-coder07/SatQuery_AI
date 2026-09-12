"""Load every tool's checkpoint through the real tool loader, per a deploy map.

The config-driven successor of `verify_v2_deploy.py`: the map lives in
`configs/deploy.<tag>.yaml` so a deployment change is a config diff, not a
code change, and the same script verifies v2 and v3 maps.

For each tool: set its environment variables for this process only, call
`is_available()`, construct the loader, report the architecture, parameter
count and load time. VLM tools (rs_vqa, grounding via adapter) share one
base; the base is loaded once and adapters are attached to it.

Read-only. Nothing is written.

Usage (on the machine that holds the checkpoints)::

    python scripts/verify_deploy.py --map configs/deploy.v3.yaml [--root ~/satquery] [--only change_mask ...]
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOADERS = {
    "landcover": "_Handle", "change_mask": "_Handle", "grounding": "_Handle", "caption": "_Handle",
    "change_caption": "_Handle", "optsar_fusion": "_Handle", "change_vqa": "_SemanticHandle",
}


def check_tool(tool: str, spec: dict, root: Path) -> tuple[bool, str]:
    env = dict(spec.get("env") or {})
    for k, v in env.items():
        p = root / v
        if not p.exists():
            return False, f"{k}={v} does not exist"
        os.environ[k] = str(p)
    module = importlib.import_module(f"satquery.tools.{tool}")
    if hasattr(module, "is_available"):
        ok, reason = module.is_available()
        if not ok:
            return False, f"is_available(): {reason}"
    started = time.time()
    if tool == "rs_vqa":
        handle = module._ModelHandle.get(Path(os.environ["SATQUERY_VQA_BASE"]), Path(os.environ["SATQUERY_VQA_ADAPTER"]))
        return True, f"{type(handle.model).__name__}, adapters {list(handle.adapters)}, {time.time() - started:.1f}s"
    if tool == "grounding" and "SATQUERY_GROUNDING_ADAPTER" in env:
        from satquery.tools import rs_vqa

        handle = rs_vqa._ModelHandle.get(Path(os.environ["SATQUERY_VQA_BASE"]), Path(os.environ["SATQUERY_VQA_ADAPTER"]))
        handle.ensure_adapter(module.VLM_ADAPTER_NAME, Path(os.environ["SATQUERY_GROUNDING_ADAPTER"]))
        return True, f"adapter '{module.VLM_ADAPTER_NAME}' attached to the shared base, {time.time() - started:.1f}s"
    path = root / next(iter(env.values()))
    handle = getattr(module, LOADERS[tool])(path)
    model = getattr(handle, "model", None)
    n = sum(p.numel() for p in model.parameters()) / 1e6 if model is not None else 0.0
    arch = type(model).__name__ if model is not None else "?"
    return True, f"{arch}, {n:.1f}M params, {time.time() - started:.1f}s"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--map", type=Path, default=Path("configs/deploy.v3.yaml"))
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--only", nargs="*", default=None)
    args = p.parse_args()
    spec = yaml.safe_load(args.map.read_text(encoding="utf-8"))
    tools = spec["tools"]
    order = ["rs_vqa"] + [t for t in tools if t != "rs_vqa"]  # base first, adapters after
    ok_n = 0
    for tool in order:
        if args.only and tool not in args.only:
            continue
        s = tools[tool]
        try:
            ok, msg = check_tool(tool, s, args.root)
        except Exception:  # noqa: BLE001 - report every failure, keep going
            ok, msg = False, traceback.format_exc().strip().splitlines()[-1]
        ok_n += ok
        print(f"{'OK ' if ok else 'FAIL'} {tool:<15} [{s.get('status', '?'):<8}] {msg}")
    total = len(args.only) if args.only else len(tools)
    print(f"\n{ok_n}/{total} loaded")
    return 0 if ok_n == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
