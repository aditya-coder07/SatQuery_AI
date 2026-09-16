"""Multi-task LoRA SFT for Qwen2.5-VL over unified-format manifests.

Generalises `train_grounding_vlm.py` to any mix of tasks that the unified
JSONL carries (`docs/research/reproducibility.md` §6). One trainer for the
task-specific adapters (arm B of the unified-VLM study) **and** for the
shared adapter (arm A): the difference is only which manifests are listed.

Per task:

* ``<VQA>``        - answer is the target string; deployed system prompt
                     (`satquery/tools/rs_vqa.py`); val metric = exact match
                     after `normalise_answer`, reported per question type and
                     in the published convention (count excluded);
* ``<GROUNDING>``  - answer is the pretraining JSON box in the frame the
                     model sees; val metric = Acc@0.5 from generation.

Selection is on **generated** answers, never on loss. Adapter only; the base
model and every other adapter are untouched.

Usage::

    python training/train_vlm_sft.py --model models/qwen25_vl_3b \
        --train data/rsvqa_lr_official/manifests/train.jsonl \
        --val data/rsvqa_lr_official/manifests/val.jsonl \
        --ckpt-dir checkpoints/v3/vqa_official --epochs 1

Several ``--train`` / ``--val`` manifests may be given; ``--train-weight``
(one float per manifest) up- or down-samples a source per epoch.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics.vqa import normalise_answer  # noqa: E402
from training.common import registry  # noqa: E402
from training.common import vlm_grounding as vg  # noqa: E402

VQA_SYSTEM_PROMPT = (
    "You are a remote-sensing image analyst. Answer only from what is visible "
    "in the imagery. If the image does not support an answer, say so."
)
DEFAULT_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]
BF16_MIN_FREE_GB = 20.0
PUBLISHED_TYPES = ("presence", "comp", "rural_urban")


def load_manifest(path: Path, limit: int | None = None, seed: int = 0, weight: float = 1.0) -> list[dict]:
    root = path.parent.parent  # <dataset>/manifests/x.jsonl -> <dataset>
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            # One image (VQA, grounding, caption) or two (change caption).
            r["_images"] = [str(root / q) for q in (r.get("images") or [r["image"]])]
            rows.append(r)
    rng = random.Random(seed)
    if limit is not None and limit < len(rows):
        rng.shuffle(rows)
        rows = rows[:limit]
    if weight < 1.0:
        rows = rng.sample(rows, int(round(len(rows) * weight)))
    elif weight > 1.0:
        whole, frac = int(weight), weight - int(weight)
        rows = rows * whole + rng.sample(rows, int(round(len(rows) * frac)))
    return rows


def chat_for(row: dict, images: list, processor, with_answer: bool) -> list[dict]:
    task = row["task"]
    if task == "<GROUNDING>":
        image = images[0]
        answer = None
        if with_answer:
            rh, rw = vg.smart_resized_hw(processor, image.width, image.height)
            box = vg.scale_box(row["target"]["bbox_xyxy"], rw / image.width, rh / image.height)
            answer = vg.render_answer(box, row["target"].get("category") or "object")
        return vg.build_chat(image, row["question"], answer)
    # <VQA>, <CAPTION>, <CHANGE_CAPTION>: text answer, one or more images.
    chat = [
        {"role": "system", "content": [{"type": "text", "text": VQA_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": im} for im in images]
                                    + [{"type": "text", "text": row["question"]}]},
    ]
    if with_answer:
        target = row["target"]
        if isinstance(target, list):  # eval rows carry all references; train on the first
            target = target[0]
        chat.append({"role": "assistant", "content": [{"type": "text", "text": str(target)}]})
    return chat


class Rows:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        from PIL import Image

        r = self.rows[i]
        return [Image.open(q).convert("RGB") for q in r["_images"]], r


class Collate:
    def __init__(self, processor):
        self.processor = processor
        self.marker = vg.marker_ids(processor.tokenizer)

    def __call__(self, items):
        images, texts = [], []
        for imgs, r in items:
            texts.append(self.processor.apply_chat_template(chat_for(r, imgs, self.processor, True), tokenize=False))
            images.extend(imgs)  # flattened in order of appearance, as the processor expects
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        for i in range(labels.shape[0]):
            labels[i, :vg.assistant_start(batch["input_ids"][i], self.marker)] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch


def generate_batch(model, processor, image_lists, rows, max_new_tokens):
    import torch

    tok = processor.tokenizer
    prev = tok.padding_side
    tok.padding_side = "left"
    try:
        texts = [processor.apply_chat_template(chat_for(r, imgs, processor, False), tokenize=False,
                                               add_generation_prompt=True) for imgs, r in zip(image_lists, rows)]
        flat = [im for imgs in image_lists for im in imgs]
        batch = processor(text=texts, images=flat, return_tensors="pt", padding=True)
    finally:
        tok.padding_side = prev
    batch = batch.to(model.device)
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                             pad_token_id=tok.pad_token_id)
    replies = tok.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
    return replies, batch["image_grid_thw"]


def evaluate(model, processor, rows: list[dict], batch_size: int) -> dict:
    """Generation-based metrics per task; returns a `selection` scalar."""
    from PIL import Image

    model.eval()
    by_task: dict[str, list] = collections.defaultdict(list)
    for s in range(0, len(rows), batch_size):
        chunk = rows[s:s + batch_size]
        image_lists = [[Image.open(q).convert("RGB") for q in r["_images"]] for r in chunk]
        replies, grids = generate_batch(model, processor, image_lists, chunk, 48)
        gi = 0
        for r, imgs, reply in zip(chunk, image_lists, replies):
            grid = grids[gi]
            gi += len(imgs)
            if r["task"] == "<GROUNDING>":
                img = imgs[0]
                box = vg.parse_box(reply)
                iou = 0.0
                if box:
                    rh, rw = vg.resized_hw(grid)
                    iou = vg.iou_xyxy(vg.scale_box(box, img.width / rw, img.height / rh), r["target"]["bbox_xyxy"])
                by_task[r["task"]].append({"hit": iou >= 0.5, "iou": iou})
            elif r["task"] in ("<CAPTION>", "<CHANGE_CAPTION>"):
                refs = r["target"] if isinstance(r["target"], list) else [str(r["target"])]
                by_task[r["task"]].append({"hyp": reply, "refs": refs,
                                           "changeflag": (r.get("metadata") or {}).get("changeflag")})
            else:
                hit = normalise_answer(reply) == normalise_answer(str(r["target"]))
                by_task[r["task"]].append({"hit": hit, "type": (r.get("metadata") or {}).get("type", "all")})
    model.train()
    out: dict = {"n": len(rows)}
    sel = []
    for task, items in by_task.items():
        n = len(items)
        if task in ("<CAPTION>", "<CHANGE_CAPTION>"):
            from evaluation.metrics.caption_corpus import score_corpus

            hyps, refs = [i["hyp"] for i in items], [i["refs"] for i in items]
            res = {"n": n, "corpus": {k: round(v, 4) for k, v in score_corpus(hyps, refs).items()}}
            if task == "<CHANGE_CAPTION>":
                for name, flag in (("changed", 1), ("unchanged", 0)):
                    sub = [i for i in items if i["changeflag"] == flag]
                    if sub:
                        res[name] = {k: round(v, 4) for k, v in
                                     score_corpus([i["hyp"] for i in sub], [i["refs"] for i in sub]).items()}
            res["unique_fraction"] = round(len(set(hyps)) / n, 4)
            out[task] = res
            sel.append(res["corpus"]["bleu4"])
            continue
        acc = sum(i["hit"] for i in items) / n
        if task == "<GROUNDING>":
            out[task] = {"n": n, "acc@0.5": acc, "miou": sum(i["iou"] for i in items) / n}
            sel.append(acc)
        else:
            types = collections.defaultdict(list)
            for i in items:
                types[i["type"]].append(i["hit"])
            pub = [h for t, hs in types.items() for h in hs if t in PUBLISHED_TYPES]
            out[task] = {"n": n, "acc_all": acc,
                         "acc_published": (sum(pub) / len(pub)) if pub else acc,
                         "by_type": {t: round(sum(h) / len(h), 4) for t, h in sorted(types.items())}}
            sel.append(out[task]["acc_published"])
    out["selection"] = sum(sel) / len(sel) if sel else 0.0
    return out


def choose_quant(mode: str, torch) -> str:
    if mode != "auto":
        return mode
    free, _ = torch.cuda.mem_get_info()
    return "none" if free / 2**30 >= BF16_MIN_FREE_GB else "4bit"


def build_model(args, torch, peft):
    from transformers import AutoProcessor, BitsAndBytesConfig

    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    if args.max_pixels or args.min_pixels:
        vg.set_pixel_budget(processor, args.min_pixels, args.max_pixels)
    quant = choose_quant(args.quant, torch)
    kwargs = dict(device_map={"": 0}, local_files_only=True, trust_remote_code=False)
    if quant == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kwargs["dtype"] = torch.bfloat16
    model = AutoVLM.from_pretrained(str(args.model), **kwargs)
    model.config.use_cache = False
    if quant == "4bit":
        model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=not args.no_checkpointing)
    else:
        if not args.no_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if args.init_adapter:
        # Continue from an existing adapter (e.g. the deployed VQA adapter),
        # so a run can be "the deployed model plus more data" rather than a
        # fresh LoRA. Its LoraConfig is used as-is.
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.init_adapter), is_trainable=True)
    else:
        lora = peft.LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                               bias="none", task_type="CAUSAL_LM", target_modules=args.lora_targets,
                               exclude_modules=r".*visual.*")
        model = peft.get_peft_model(model, lora)
    if args.train_merger:
        for name, p in model.named_parameters():
            if "visual.merger" in name:
                p.requires_grad_(True)
    model.print_trainable_parameters()
    return model, processor, quant


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
    p.add_argument("--train", type=Path, nargs="+", required=True)
    p.add_argument("--train-weight", type=float, nargs="*", default=None)
    p.add_argument("--val", type=Path, nargs="+", required=True)
    p.add_argument("--val-limit", type=int, default=400, help="per val manifest")
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--init-adapter", type=Path, default=None)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--quant", choices=["auto", "none", "4bit"], default="auto")
    p.add_argument("--no-checkpointing", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", nargs="*", default=DEFAULT_LORA_TARGETS)
    p.add_argument("--train-merger", action="store_true")
    p.add_argument("--max-pixels", type=int, default=None)
    p.add_argument("--min-pixels", type=int, default=None,
                   help="upscale images below this many pixels (e.g. 1048576 = 1024x1024 for 800-px DIOR)")
    p.add_argument("--limit-train", type=int, default=None, help="per train manifest")
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--eval-batch", type=int, default=32)
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

    weights = args.train_weight or [1.0] * len(args.train)
    train_rows = [r for m, w in zip(args.train, weights) for r in load_manifest(m, args.limit_train, args.seed, w)]
    val_rows = [r for m in args.val for r in load_manifest(m, args.val_limit, args.seed)]
    random.Random(args.seed).shuffle(train_rows)
    print(f"train {len(train_rows)} ({collections.Counter(r['task'] for r in train_rows)}) | "
          f"val(selection) {len(val_rows)}")

    model, processor, quant = build_model(args, torch, peft)
    print(f"quant: {quant}")
    loader = DataLoader(Rows(train_rows), batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=Collate(processor), drop_last=True, persistent_workers=args.workers > 0)
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
    step, best, history = 0, {"selection": -1.0}, []
    if args.resume and state_path.exists():
        from safetensors.torch import load_file

        st = torch.load(state_path, map_location="cpu", weights_only=False)
        peft.set_peft_model_state_dict(model, load_file(str(args.ckpt_dir / "adapter_last" / "adapter_model.safetensors")))
        optimizer.load_state_dict(st["optimizer"])
        step, best, history = st["step"], st["best"], st.get("history", [])
        print(f"resumed at step {step}, best {best.get('selection')}")

    exp_id = registry.new_id("vlm_sft")
    hw = registry.hardware()
    hp = {k: (str(v) if isinstance(v, Path) else ([str(x) for x in v] if isinstance(v, list) else v))
          for k, v in vars(args).items()}
    manifest_hashes = {}
    for m in list(args.train) + list(args.val):
        st_path = m.parent / "stats.json"
        if st_path.exists():
            manifest_hashes[str(m)] = json.loads(st_path.read_text()).get("hashes", {}).get(m.name)
    registry.record(experiment_id=exp_id, model=str(args.model), architecture="qwen2.5-vl+lora",
                    hyperparameters={**hp, "quant": quant, "total_steps": total_steps},
                    dataset_versions=manifest_hashes, seed=args.seed, hardware=hw, status="running",
                    notes=args.notes, checkpoint=str(args.ckpt_dir))
    (args.ckpt_dir / "run_metadata.json").write_text(json.dumps({
        "experiment_id": exp_id, "task": "vlm_sft", "backbone": "Qwen2.5-VL + LoRA",
        "n_train": len(train_rows), "tasks": dict(collections.Counter(r["task"] for r in train_rows)),
        "n_val_selection": len(val_rows), "quant": quant, "manifests": manifest_hashes,
        "hardware": hw, "args": hp, "total_steps": total_steps}, indent=2), encoding="utf-8")

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
                      f"{rate:.2f}s/step vram {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
                run_loss, run_n = 0.0, 0
            if step % args.val_every == 0 or step == total_steps:
                m = evaluate(model, processor, val_rows, args.eval_batch)
                m["step"] = step
                history.append(m)
                print(f"VAL step {step}: {json.dumps(m)}", flush=True)
                if m["selection"] > best["selection"]:
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
    (args.ckpt_dir / "metrics.json").write_text(json.dumps({
        "best_val": best, "history": history, "step": step, "elapsed_s": elapsed,
        "gpu_hours": elapsed / 3600}, indent=2), encoding="utf-8")
    registry.record(experiment_id=exp_id, status="done", duration_s=elapsed, training_steps=step,
                    validation_metrics=best, checkpoint=str(args.ckpt_dir / "adapter_best"),
                    checkpoint_sha256=registry.sha256_dir(args.ckpt_dir / "adapter_best"),
                    model=str(args.model), architecture="qwen2.5-vl+lora", seed=args.seed, hardware=hw)
    print(f"done: best val selection {best.get('selection')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
