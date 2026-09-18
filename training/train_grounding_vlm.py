"""Fine-tune Qwen2.5-VL for referring-expression grounding (LoRA).

Why a second grounding trainer
------------------------------
`train_grounding.py` regresses a box from a small from-scratch (or ImageNet)
CNN and reached 16.0% Acc@0.5 - on a self-made split of the official test
set, because that was the only data on disk. Published DIOR-RSVG results sit
at 77-83%, all from models with a pretrained vision-language alignment. The
cheapest fair path to that regime is the VLM this project already deploys
for VQA: Qwen2.5-VL emits boxes natively, so grounding is a text target and
the whole QLoRA/LoRA stack carries over.

Design
------
* Data: the unified JSONL manifests from
  `training/prepare/dior_rsvg_official.py` - OFFICIAL train for training,
  OFFICIAL val for model selection, and the test split is never read here.
* Target: the pretraining answer format (see `training/common/vlm_grounding`)
  with boxes in the frame the model actually sees.
* Batched: examples are right-padded together; labels cover the assistant
  reply and nothing else, located by token search for the assistant marker.
* Selection metric is **val Acc@0.5 from real generation**, not loss. Phase 5
  showed val loss does not predict RSVQA accuracy; a box is a worse case,
  since a one-token slip is a 30-pixel miss.
* Adapter only. The base model is untouched, so the deployed VQA adapter is
  untouched: this is a task-specific adapter (option B of the unified-VLM
  study), and RSVQA cannot regress by construction.
* Precision: bf16 LoRA when the SHARED GPU currently has room, 4-bit NF4
  otherwise - decided from `mem_get_info`, never from total VRAM.

Usage::

    python training/train_grounding_vlm.py --model models/qwen25_vl_3b \
        --data data/dior_rsvg_official --ckpt-dir checkpoints/v3/grounding_vlm \
        --epochs 3 --batch-size 8 --grad-accum 2 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common import registry  # noqa: E402
from training.common.vlm_grounding import (  # noqa: E402
    assistant_start, build_chat, iou_xyxy, marker_ids, predict_batch,
    render_answer, scale_box, smart_resized_hw,
)

DEFAULT_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]
BF16_MIN_FREE_GB = 20.0


def load_manifest(path: Path, limit: int | None = None, seed: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit is not None and limit < len(rows):
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:limit]
    return rows


class GroundingSet:
    def __init__(self, rows: list[dict], root: Path):
        self.rows, self.root = rows, root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        from PIL import Image

        r = self.rows[i]
        img = Image.open(self.root / r["image"]).convert("RGB")
        return img, r


class Collate:
    """Runs the processor in DataLoader workers so the GPU never waits on JPEG decode."""

    def __init__(self, processor):
        self.processor = processor
        self.marker = marker_ids(processor.tokenizer)

    def __call__(self, items):
        images, texts = [], []
        for img, r in items:
            rh, rw = smart_resized_hw(self.processor, img.width, img.height)
            box = scale_box(r["target"]["bbox_xyxy"], rw / img.width, rh / img.height)
            answer = render_answer(box, r["target"].get("category") or "object")
            chat = build_chat(img, r["question"], answer)
            texts.append(self.processor.apply_chat_template(chat, tokenize=False))
            images.append(img)
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        pad = self.processor.tokenizer.pad_token_id
        for i in range(labels.shape[0]):
            start = assistant_start(batch["input_ids"][i], self.marker)
            labels[i, :start] = -100
        labels[labels == pad] = -100
        batch["labels"] = labels
        return batch


def choose_quant(mode: str, torch) -> str:
    if mode != "auto":
        return mode
    free, _ = torch.cuda.mem_get_info()
    return "none" if free / 2**30 >= BF16_MIN_FREE_GB else "4bit"


def build_model(args, torch, peft):
    from transformers import AutoProcessor, BitsAndBytesConfig

    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:  # transformers < 5
        from transformers import AutoModelForVision2Seq as AutoVLM

    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    if args.max_pixels:
        processor.image_processor.size["longest_edge"] = args.max_pixels
        processor.image_processor.max_pixels = args.max_pixels

    quant = choose_quant(args.quant, torch)
    kwargs = dict(device_map={"": 0}, local_files_only=True, trust_remote_code=False)
    if quant == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        kwargs["dtype"] = torch.bfloat16
    model = AutoVLM.from_pretrained(str(args.model), **kwargs)
    model.config.use_cache = False

    if quant == "4bit":
        model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    lora = peft.LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=args.lora_targets,
        # The vision tower is frozen; only the language tower's linears adapt.
        exclude_modules=r".*visual.*",
    )
    model = peft.get_peft_model(model, lora)
    if args.train_merger:
        # The patch-merger is the only bridge from vision to language; letting
        # it move is what GeoGround/GeoChat do and is ~15M params.
        for name, p in model.named_parameters():
            if "visual.merger" in name:
                p.requires_grad_(True)
    model.print_trainable_parameters()
    return model, processor, quant


def evaluate(model, processor, rows: list[dict], root: Path, batch_size: int) -> dict:
    """Generation-based Acc@0.5 / mIoU on `rows`; the selection metric."""
    from PIL import Image

    model.eval()
    hits25 = hits5 = hits7 = 0
    ious: list[float] = []
    n_parsed = 0
    for s in range(0, len(rows), batch_size):
        chunk = rows[s:s + batch_size]
        images = [Image.open(root / r["image"]).convert("RGB") for r in chunk]
        preds, _, parsed = predict_batch(model, processor, images, [r["question"] for r in chunk])
        for r, p, ok in zip(chunk, preds, parsed):
            n_parsed += int(ok)
            iou = iou_xyxy(p, r["target"]["bbox_xyxy"]) if p else 0.0
            ious.append(iou)
            hits25 += iou >= 0.25
            hits5 += iou >= 0.5
            hits7 += iou >= 0.7
    model.train()
    n = max(1, len(rows))
    return {"n": len(rows), "acc@0.25": hits25 / n, "acc@0.5": hits5 / n,
            "acc@0.7": hits7 / n, "miou": sum(ious) / n, "parse_rate": n_parsed / n}


def save_adapter(model, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(tmp)
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True, help="dir with manifests/{train,val}.jsonl")
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--quant", choices=["auto", "none", "4bit"], default="auto")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", nargs="*", default=DEFAULT_LORA_TARGETS)
    p.add_argument("--train-merger", action="store_true")
    p.add_argument("--max-pixels", type=int, default=None,
                   help="cap processor longest_edge pixels (tokens); default keeps native 800x800 -> 812x812")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--val-limit", type=int, default=400)
    p.add_argument("--val-every", type=int, default=500, help="optimizer steps")
    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--notes", default="")
    args = p.parse_args()

    import torch
    import peft
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    train_rows = load_manifest(args.data / "manifests" / "train.jsonl", args.limit_train, args.seed)
    val_rows = load_manifest(args.data / "manifests" / "val.jsonl", args.val_limit, args.seed)
    print(f"train {len(train_rows)} | val(selection) {len(val_rows)}")

    model, processor, quant = build_model(args, torch, peft)
    print(f"quant: {quant}")

    loader = DataLoader(GroundingSet(train_rows, args.data), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers, collate_fn=Collate(processor),
                        drop_last=True, persistent_workers=args.workers > 0)
    steps_per_epoch = len(loader) // args.grad_accum
    total_steps = int(math.ceil(steps_per_epoch * args.epochs))
    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        prog = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.ckpt_dir / "train_state.pt"
    step, best = 0, {"acc@0.5": -1.0}
    history: list[dict] = []
    if args.resume and state_path.exists():
        from safetensors.torch import load_file

        st = torch.load(state_path, map_location="cpu", weights_only=False)
        weights = load_file(str(args.ckpt_dir / "adapter_last" / "adapter_model.safetensors"))
        peft.set_peft_model_state_dict(model, weights)
        optimizer.load_state_dict(st["optimizer"])
        step, best, history = st["step"], st["best"], st.get("history", [])
        print(f"resumed at step {step}, best {best}")

    exp_id = registry.new_id("grounding_vlm")
    hw = registry.hardware()
    hp = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    registry.record(experiment_id=exp_id, model=str(args.model), architecture="qwen2.5-vl+lora",
                    hyperparameters={**hp, "quant": quant, "total_steps": total_steps},
                    dataset_versions={"dior_rsvg_official": json.loads(
                        (args.data / "manifests" / "stats.json").read_text())["hashes"]},
                    seed=args.seed, hardware=hw, status="running", notes=args.notes,
                    checkpoint=str(args.ckpt_dir))
    (args.ckpt_dir / "run_metadata.json").write_text(json.dumps({
        "experiment_id": exp_id, "task": "referring_grounding", "backbone": "Qwen2.5-VL + LoRA",
        "n_train": len(train_rows), "n_val_selection": len(val_rows), "quant": quant,
        "split_note": "OFFICIAL DIOR-RSVG train/val; test never read", "hardware": hw,
        "args": hp, "total_steps": total_steps,
    }, indent=2), encoding="utf-8")

    model.train()
    started = time.time()
    last_log = (started, step)
    micro, run_loss, run_n = 0, 0.0, 0
    done = step >= total_steps
    while not done:
        for batch in loader:
            batch = batch.to(model.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / args.grad_accum
            loss.backward()
            run_loss += loss.item() * args.grad_accum
            run_n += 1
            micro += 1
            if micro % args.grad_accum:
                continue
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % 20 == 0 or step == 1:
                now = time.time()
                rate = (now - last_log[0]) / max(1, step - last_log[1])
                last_log = (now, step)
                print(f"step {step}/{total_steps} loss {run_loss / run_n:.4f} lr {lr_at(step):.2e} "
                      f"{rate:.2f}s/step vram {torch.cuda.max_memory_allocated() / 2**30:.1f}G",
                      flush=True)
                run_loss, run_n = 0.0, 0

            if step % args.val_every == 0 or step == total_steps:
                m = evaluate(model, processor, val_rows, args.data, args.eval_batch)
                m["step"] = step
                history.append(m)
                print(f"VAL step {step}: {json.dumps(m)}", flush=True)
                if m["acc@0.5"] > best["acc@0.5"]:
                    best = m
                    save_adapter(model, args.ckpt_dir / "adapter_best")
                    print("  -> new best, adapter_best saved", flush=True)
            if step % args.save_every == 0 or step == total_steps:
                save_adapter(model, args.ckpt_dir / "adapter_last")
                torch.save({"step": step, "best": best, "history": history,
                            "optimizer": optimizer.state_dict()}, state_path)
                (args.ckpt_dir / "metrics.json").write_text(json.dumps({
                    "best_val": best, "history": history, "step": step,
                    "elapsed_s": time.time() - started}, indent=2), encoding="utf-8")
            if step >= total_steps:
                done = True
                break

    save_adapter(model, args.ckpt_dir / "adapter_final")
    elapsed = time.time() - started
    final = {"best_val": best, "history": history, "step": step, "elapsed_s": elapsed,
             "gpu_hours": elapsed / 3600}
    (args.ckpt_dir / "metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    registry.record(experiment_id=exp_id, status="done", duration_s=elapsed, training_steps=step,
                    validation_metrics=best, checkpoint=str(args.ckpt_dir / "adapter_best"),
                    checkpoint_sha256=registry.sha256_dir(args.ckpt_dir / "adapter_best"),
                    model=str(args.model), architecture="qwen2.5-vl+lora", seed=args.seed,
                    hardware=hw)
    print(f"done: best val {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
