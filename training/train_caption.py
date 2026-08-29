"""Scene captioning on RSICD (plan task 2.8).

Task 2.8 pairs a `caption_v1` adapter with a `landcover_v1` narrative grounded
in index statistics. The narrative half already exists - `satquery/synth/
narrative.py` builds land-cover prose from measured NDVI/NDWI/built-up
fractions, and the verifier checks its claims against those same indices. This
supplies the other half: a learned captioner for free-form scene description,
where there is nothing for the index engine to ground.

Trained on RSICD (~10k aerial scenes, 5 reference captions each).

The division of labour is deliberate and worth stating: anything the physics
can measure is described deterministically and verified, and only genuinely
open-ended description is left to a learned model. That keeps quantitative
claims auditable and confines hallucination risk to the part of the answer
that is qualitative anyway.

Usage:
    python training/train_caption.py --data data/rsicd --ckpt-dir checkpoints/caption \
        --epochs 8
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common.checkpointing import (  # noqa: E402
    TrainingState, maybe_resume, save_checkpoint, set_seed, write_run_metadata,
)
from training.train_change_caption import (  # noqa: E402
    BOS, EOS, MAX_LEN, PAD, build_vocab, decode, encode,
)

IMAGE_SIZE = 224


def build_model(vocab_size: int, dim: int = 192):
    import torch
    import torch.nn as nn

    def block(cin, cout, stride=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    class SceneCaptioner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision = nn.Sequential(
                block(3, 32), block(32, 64, 2), block(64, 128, 2),
                block(128, dim, 2), block(dim, dim, 2), nn.AdaptiveAvgPool2d(1),
            )
            self.embed = nn.Embedding(vocab_size, dim, padding_idx=PAD)
            self.gru = nn.GRU(dim, dim, batch_first=True)
            self.out = nn.Linear(dim, vocab_size)

        def forward(self, image, tokens):
            context = self.vision(image).flatten(1)
            out, _ = self.gru(self.embed(tokens), context.unsqueeze(0).contiguous())
            return self.out(out)

        @torch.no_grad()
        def generate(self, image, max_len: int = MAX_LEN):
            hidden = self.vision(image).flatten(1).unsqueeze(0).contiguous()
            token = torch.full((image.shape[0], 1), BOS, dtype=torch.long,
                               device=image.device)
            produced = []
            for _ in range(max_len):
                out, hidden = self.gru(self.embed(token), hidden)
                token = self.out(out[:, -1]).argmax(-1, keepdim=True)
                produced.append(token)
            return torch.cat(produced, dim=1)

    return SceneCaptioner()


def load_rows(data_dir: Path, split: str) -> list[dict]:
    import pyarrow.parquet as pq

    files = sorted(
        f for f in glob.glob(str(data_dir / "**" / "*.parquet"), recursive=True)
        if split in Path(f).name
    )
    rows: list[dict] = []
    for path in files:
        table = pq.read_table(path).to_pydict()
        images = table.get("image") or []
        captions = table.get("captions") or []
        for image, caps in zip(images, captions):
            raw = image.get("bytes") if isinstance(image, dict) else image
            texts = [str(c).strip() for c in (caps or []) if str(c).strip()]
            if isinstance(raw, (bytes, bytearray)) and texts:
                rows.append({"image_bytes": bytes(raw), "captions": texts})
    return rows


class RSICD:
    def __init__(self, rows, vocab):
        self.rows = rows
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        from PIL import Image

        row = self.rows[i]
        img = Image.open(io.BytesIO(row["image_bytes"])).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        arr = np.asarray(img, dtype="float32").transpose(2, 0, 1) / 255.0
        # Train against the first reference; all five are kept for scoring.
        return arr, encode(row["captions"][0], self.vocab)


def batches(dataset, size, rng, shuffle=True):
    order = rng.permutation(len(dataset)) if shuffle else np.arange(len(dataset))
    for start in range(0, len(order), size):
        idx = order[start : start + size]
        i, t = zip(*(dataset[int(j)] for j in idx))
        yield np.stack(i), np.stack(t)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/caption"))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    import torch
    import torch.nn as nn

    from evaluation.metrics.all_tasks import bleu

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_rows = load_rows(args.data, "train")
    test_rows = load_rows(args.data, "test") or load_rows(args.data, "valid")
    if not train_rows:
        print(f"no rows under {args.data}", file=sys.stderr)
        return 1
    if not test_rows:
        cut = int(len(train_rows) * 0.9)
        train_rows, test_rows = train_rows[:cut], train_rows[cut:]
        print("NOTE: no test split found; held out the last 10% of train")

    # Train captions only - test vocabulary would leak and inflate BLEU.
    vocab = build_vocab([c for r in train_rows for c in r["captions"]])
    inverse = {i: w for w, i in vocab.items()}
    print(f"train {len(train_rows)} | test {len(test_rows)} | vocab {len(vocab)}")

    train_ds, test_ds = RSICD(train_rows, vocab), RSICD(test_rows, vocab)
    model = build_model(len(vocab), args.dim).to(device)
    print(f"parameters: {sum(q.numel() for q in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    state, _ = maybe_resume(args.ckpt_dir, model, optimizer, enabled=args.resume)

    write_run_metadata(args.ckpt_dir, {
        "task": "scene_caption_rsicd", "n_train": len(train_rows),
        "epochs": args.epochs, "lr": args.lr, "dim": args.dim,
        "vocab_size": len(vocab),
    })
    (args.ckpt_dir / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")

    rng = np.random.default_rng(args.seed)
    step = state.step
    started = time.time()

    for epoch in range(state.epoch, args.epochs):
        model.train()
        running, seen = 0.0, 0
        for img, tok in batches(train_ds, args.batch_size, rng):
            ib, tb = torch.from_numpy(img).to(device), torch.from_numpy(tok).to(device)
            logits = model(ib, tb[:, :-1])
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tb[:, 1:].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * img.shape[0]
            seen += img.shape[0]
            step += 1
            if step % args.save_every == 0:
                state.step, state.epoch = step, epoch
                save_checkpoint(args.ckpt_dir, step, model, optimizer, state=state)
        print(f"epoch {epoch+1}/{args.epochs}  loss {running/max(seen,1):.4f}  "
              f"({time.time()-started:.0f}s)", flush=True)

    state.step, state.epoch = step, args.epochs
    save_checkpoint(args.ckpt_dir, step, model, optimizer, state=state)

    model.eval()
    predictions = []
    for img, _ in batches(test_ds, args.batch_size, rng, shuffle=False):
        ids = model.generate(torch.from_numpy(img).to(device)).cpu().numpy()
        predictions.extend(decode(x, inverse) for x in ids)

    # Scored against all five references, as the dataset intends.
    scores = [
        bleu(hyp, row["captions"]) for hyp, row in zip(predictions, test_rows)
    ]
    unique = len(set(predictions))
    metrics = {
        "bleu4_sentence_mean": float(np.mean(scores)) if scores else 0.0,
        "n": len(scores),
        "unique_captions": unique,
        "unique_fraction": round(unique / max(len(predictions), 1), 4),
    }
    print(f"\nRSICD test BLEU-4 (sentence mean, 5 refs): "
          f"{metrics['bleu4_sentence_mean']:.4f}  n={metrics['n']}")
    # Caption diversity is reported alongside the score: a captioner that
    # emits one string for every image can still post a respectable BLEU on a
    # corpus with a dominant phrasing, and the count exposes that immediately.
    print(f"  unique captions: {unique} ({metrics['unique_fraction']:.1%})")
    for hyp, row in list(zip(predictions, test_rows))[:3]:
        print(f"  pred: {hyp[:68]}")
        print(f"  ref : {row['captions'][0][:68]}")

    (args.ckpt_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
