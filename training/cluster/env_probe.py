"""What GPU are we actually on, and what does that permit?

Every training default in this repository was chosen for a free-tier T4
(`docs/03` section 1.1): fp16 rather than bf16 because Turing is sm75, `sdpa`
rather than FlashAttention-2 because FA2 needs sm80, batch size 1 with
accumulation because 16 GB is the ceiling. Those are not preferences. They
are consequences of one specific card, and on a college cluster card they are
very likely wrong in the expensive direction - an A100 run left on fp16 and
batch size 1 wastes most of the hardware it was given.

The fix is not to hardcode a second set of constants for "the cluster". A
shared cluster hands out whatever is free, and a notebook session gets a
different card on Tuesday than it got on Monday. So the recipe is derived
from the card at run time and **recorded in the run metadata**, which means a
number can later be tied to the hardware that produced it rather than to an
assumption about it.

Three things are decided here and nowhere else:

* **precision** - bf16 needs compute capability 8.0. Below that, fp16 with a
  GradScaler, which is what the existing scripts already do.
* **attention** - FlashAttention-2 needs sm80 *and* the `flash-attn` package
  actually importable. Both are checked, because a config that names an
  attention implementation the environment cannot provide fails at model
  load, several minutes into a job.
* **batch shape** - a per-job memory budget scaled by observed VRAM, with
  the effective batch held constant by adjusting gradient accumulation. That
  last part matters: changing the effective batch silently changes the
  optimisation problem, so a bigger card should make a run *faster*, not
  different.

This module imports torch lazily and returns a fully-populated profile with
`available=False` when there is none, so the campaign configuration can be
validated, printed and tested on a laptop with no GPU.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Compute capability 8.0 (Ampere) is the line for both bf16 tensor-core
# support and FlashAttention-2. Turing (7.5) and Pascal (6.0) sit below it.
AMPERE = (8, 0)

# VRAM below this and the T4-shaped defaults are the right defaults, because
# the machine is T4-shaped whatever its name is.
SMALL_CARD_GB = 20.0


@dataclass
class GpuProfile:
    """What the running machine can do. Serialised into every run's metadata."""

    available: bool = False
    device_count: int = 0
    name: str = "cpu"
    capability: tuple[int, int] | None = None
    vram_gb: float = 0.0
    torch_version: str | None = None
    cuda_version: str | None = None
    bf16: bool = False
    flash_attn_2: bool = False
    host: str = field(default_factory=platform.node)
    free_disk_gb: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def dtype_name(self) -> str:
        """The compute dtype to train in. fp16 needs a GradScaler; bf16 does not."""
        return "bfloat16" if self.bf16 else "float16"

    @property
    def needs_grad_scaler(self) -> bool:
        return not self.bf16

    @property
    def attn_implementation(self) -> str:
        return "flash_attention_2" if self.flash_attn_2 else "sdpa"

    @property
    def is_small_card(self) -> bool:
        return not self.available or self.vram_gb < SMALL_CARD_GB

    def to_json(self) -> dict:
        data = asdict(self)
        data["capability"] = list(self.capability) if self.capability else None
        data["dtype"] = self.dtype_name
        data["attn_implementation"] = self.attn_implementation
        return data


def _flash_attn_importable() -> bool:
    """Is FlashAttention-2 actually installed?

    Checked by import rather than by version string: a cluster module system
    can leave a `flash_attn` on the path that was built for a different torch
    and raises on import. That failure belongs here, at probe time, not
    fifteen minutes into a job at `from_pretrained`.
    """
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    return True


def probe(data_root: str | Path = ".") -> GpuProfile:
    """Inspect the current machine. Never raises; degrades to a CPU profile."""
    profile = GpuProfile()

    try:
        profile.free_disk_gb = shutil.disk_usage(Path(data_root)).free / 1e9
    except OSError as exc:  # pragma: no cover - depends on the mount
        profile.notes.append(f"disk usage unavailable: {exc}")

    try:
        import torch
    except ImportError:
        profile.notes.append("torch not installed; CPU profile")
        return profile

    profile.torch_version = torch.__version__
    profile.cuda_version = getattr(torch.version, "cuda", None)

    if not torch.cuda.is_available():
        profile.notes.append("torch present but CUDA unavailable; CPU profile")
        return profile

    profile.available = True
    profile.device_count = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0)
    profile.name = props.name
    profile.capability = (props.major, props.minor)
    profile.vram_gb = props.total_memory / 1e9

    profile.bf16 = profile.capability >= AMPERE
    if not profile.bf16:
        profile.notes.append(
            f"compute capability {props.major}.{props.minor} < 8.0: "
            "fp16 + GradScaler, no FA2"
        )

    profile.flash_attn_2 = profile.bf16 and _flash_attn_importable()
    if profile.bf16 and not profile.flash_attn_2:
        profile.notes.append("sm80+ but flash_attn not importable; using sdpa")

    if profile.device_count > 1:
        profile.notes.append(
            f"{profile.device_count} GPUs visible; single-process scripts use "
            "device 0 only. Set CUDA_VISIBLE_DEVICES or launch under accelerate "
            "to use the rest."
        )

    return profile


# --- Recipe derivation ------------------------------------------------------
#
# One entry per job family. `micro_batch` is what a T4-class card takes; the
# scaling below multiplies it by the VRAM ratio and divides accumulation by
# the same factor, so the EFFECTIVE batch is invariant to the hardware. A run
# on an A100 and a run on a T4 then solve the same optimisation problem at
# different speeds, which is the only way two runs' numbers stay comparable.

BASE_RECIPES: dict[str, dict] = {
    "encoder": {"micro_batch": 32, "grad_accum": 1, "workers": 4},
    "vlm_qlora": {"micro_batch": 1, "grad_accum": 16, "workers": 2},
    "grounding": {"micro_batch": 32, "grad_accum": 1, "workers": 4},
    "change": {"micro_batch": 16, "grad_accum": 1, "workers": 4},
    "head": {"micro_batch": 64, "grad_accum": 1, "workers": 2},
}

# Reference card the base recipes were measured on.
REFERENCE_VRAM_GB = 16.0
MAX_SCALE = 8


def recipe_for(job_family: str, profile: GpuProfile) -> dict:
    """Batch shape for `job_family` on this machine, effective batch preserved.

    Raises `KeyError` on an unknown family rather than falling back to a
    default, because a typo in `configs/campaign.yaml` that silently trains
    with someone else's batch size is a bug that only shows up in the metric.
    """
    base = dict(BASE_RECIPES[job_family])

    if not profile.available:
        base.update(scale=1, dtype="float32", attn_implementation="sdpa",
                    grad_scaler=False, device="cpu")
        return base

    # The quantity that must not move. Everything below rearranges how this
    # total is reached, never what it is.
    effective = base["micro_batch"] * base["grad_accum"]

    # Claim only 80% of the headroom: VRAM is not the only limit (activation
    # memory grows with sequence length too), and a run that OOMs at hour
    # three has cost more than a run that used 70% of the card.
    ratio = profile.vram_gb / REFERENCE_VRAM_GB * 0.8

    if ratio >= 1.0:
        scale = float(min(MAX_SCALE, int(ratio)))
        micro = base["micro_batch"] * int(scale)
    else:
        # Smaller than the reference card. Scaling only upwards would hand a
        # 6 GB laptop the 16 GB batch size and OOM on the first step, so the
        # ratio divides instead. `scale` is reported as a fraction to make
        # that visible in the run metadata.
        divisor = max(1, int(round(1 / ratio)))
        scale = 1 / divisor
        micro = max(1, base["micro_batch"] // divisor)

    # Accumulation is DERIVED from the target rather than scaled alongside
    # micro_batch. Scaling both independently drifts: 32x1 shrunk by three
    # gives 10x3 = 30, a 6% smaller effective batch that nobody asked for and
    # that would quietly make the run non-comparable to the reference run.
    # Rounding up rather than down keeps the batch at or above the target,
    # never below it - an extra fraction of a batch is cheap, a silently
    # weaker optimisation problem is not.
    micro = min(micro, effective)
    base["micro_batch"] = micro
    base["grad_accum"] = max(1, -(-effective // micro))
    base["effective_batch"] = micro * base["grad_accum"]
    base["scale"] = scale
    base["dtype"] = profile.dtype_name
    base["attn_implementation"] = profile.attn_implementation
    base["grad_scaler"] = profile.needs_grad_scaler
    base["device"] = "cuda"
    return base


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Probe the GPU and derive batch shapes.")
    p.add_argument("--data-root", default=os.environ.get("SATQUERY_DATA_ROOT", "."))
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--out", type=Path, help="also write the profile to this path")
    args = p.parse_args()

    profile = probe(args.data_root)
    payload = profile.to_json()
    payload["recipes"] = {k: recipe_for(k, profile) for k in BASE_RECIPES}

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        cap = ".".join(str(x) for x in profile.capability) if profile.capability else "-"
        print(f"host          {profile.host}")
        print(f"gpu           {profile.name} x{profile.device_count}")
        print(f"capability    {cap}")
        print(f"vram          {profile.vram_gb:.1f} GB")
        print(f"free disk     {profile.free_disk_gb:.1f} GB")
        print(f"torch / cuda  {profile.torch_version} / {profile.cuda_version}")
        print(f"dtype         {profile.dtype_name}")
        print(f"attention     {profile.attn_implementation}")
        for note in profile.notes:
            print(f"  note: {note}")
        print()
        for family, recipe in payload["recipes"].items():
            print(f"  {family:<12} micro_batch={recipe['micro_batch']:<4} "
                  f"grad_accum={recipe['grad_accum']:<3} scale x{recipe.get("scale", 1):g} eff={recipe.get("effective_batch", "-")}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
