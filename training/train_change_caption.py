"""Mask-conditioned change captioning (plan task 2.5).

The plan specifies `change_caption_v1` as **mask-conditioned**, and that word
carries the design. A captioner given only the two dates has to rediscover
where the change is before it can describe it; one given the change mask from
task 2.4 starts from that answer and spends its capacity on describing the
change instead of locating it. It also ties the caption to the mask the system
already exported, so the prose and the raster cannot disagree.

Architecture: the siamese change encoder produces difference features, the
mask is encoded alongside them, and a small GRU decoder emits the caption
token by token over a vocabulary built from the training captions.

This is a compact model trained briefly - the plan's Phase 2 mandate is
"every row has a measured number, ugly is acceptable", not a competitive
captioner.

Usage:
    python training/train_change_caption.py --index data/levir_mci/index.json \
        --ckpt-dir checkpoints/change_caption --epochs 6
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common.checkpointing import (  # noqa: E402
    TrainingState, maybe_resume, save_checkpoint, set_seed, write_run_metadata,
)

PATCH = 256
MAX_LEN = 24
PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


def build_vocab(captions: list[str], min_count: int = 2) -> dict[str, int]:
    counts = Counter(t for c in captions for t in tokenize(c))
    vocab = {w: i for i, w in enumerate(SPECIALS)}
    for word, n in counts.most_common():
        if n >= min_count:
            vocab[word] = len(vocab)
    return vocab


def encode(caption: str, vocab: dict[str, int], max_len: int = MAX_LEN) -> np.ndarray:
    ids = [BOS] + [vocab.get(t, UNK) for t in tokenize(caption)][: max_len - 2] + [EOS]
    ids += [PAD] * (max_len - len(ids))
    return np.array(ids, dtype="int64")


def decode(ids, inverse: dict[int, str]) -> str:
    words = []
    for i in ids:
        i = int(i)
        if i in (PAD, BOS):
            continue
        if i == EOS:
            break
        words.append(inverse.get(i, "<unk>"))
    return " ".join(words)


def build_model(vocab_size: int, dim: int = 128):
    import torch
    import torch.nn as nn

    def block(cin, cout, stride=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    class MaskConditionedCaptioner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc = nn.Sequential(block(3, 32), block(32, 64, 2), block(64, 64, 2))
            # The mask gets its own small encoder rather than being stacked as
            # a fourth channel: keeping it separate means the difference
            # features and the "where" signal stay distinguishable, and the
            # model can still caption when the mask is empty.
            self.mask_enc = nn.Sequential(block(1, 16), block(16, 32, 2), block(32, 64, 2))
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.project = nn.Linear(64 * 2, dim)

            self.embed = nn.Embedding(vocab_size, dim, padding_idx=PAD)
            self.gru = nn.GRU(dim, dim, batch_first=True)
            self.out = nn.Linear(dim, vocab_size)

        def features(self, a, b, mask):
            fa, fb = self.enc(a), self.enc(b)
            diff = torch.abs(fa - fb)          # symmetric in date order
            fm = self.mask_enc(mask)
            joined = torch.cat([self.pool(diff).flatten(1), self.pool(fm).flatten(1)], 1)
            return self.project(joined)

        def forward(self, a, b, mask, tokens):
            context = self.features(a, b, mask)
            emb = self.embed(tokens)
            # The visual context seeds the recurrent state, so every emitted
            # token is conditioned on the change rather than on language
            # priors alone.
            out, _ = self.gru(emb, context.unsqueeze(0).contiguous())
            return self.out(out)

        @torch.no_grad()
        def generate(self, a, b, mask, max_len: int = MAX_LEN):
            context = self.features(a, b, mask)
            hidden = context.unsqueeze(0).contiguous()
            token = torch.full((a.shape[0], 1), BOS, dtype=torch.long, device=a.device)
            produced = []
            for _ in range(max_len):
                out, hidden = self.gru(self.embed(token), hidden)
                token = self.out(out[:, -1]).argmax(-1, keepdim=True)
                produced.append(token)
            return torch.cat(produced, dim=1)

    return MaskConditionedCaptioner()


class LevirCC:
    def __init__(self, rows: list[dict], vocab: dict[str, int]):
        self.rows = rows
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        from PIL import Image

        row = self.rows[i]
        a = np.asarray(Image.open(row["a"]).convert("RGB"), dtype="float32") / 255.0
        b = np.asarray(Image.open(row["b"]).convert("RGB"), dtype="float32") / 255.0
        if row.get("label") and Path(row["label"]).exists():
            m = np.asarray(Image.open(row["label"]).convert("L"), dtype="float32")
            m = (m > 127).astype("float32")
        else:
            m = np.zeros((PATCH, PATCH), dtype="float32")
        return (
            a.transpose(2, 0, 1), b.transpose(2, 0, 1), m[None],
            encode(row["caption"], self.vocab),
        )


def batches(dataset, size, rng, shuffle=True):
    order = rng.permutation(len(dataset)) if shuffle else np.arange(len(dataset))
    for start in range(0, len(order), size):
        idx = order[start : start + size]
        a, b, m, t = zip(*(dataset[int(i)] for i in idx))
        yield np.stack(a), np.stack(b), np.stack(m), np.stack(t)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/change_caption"))
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--limit-train", type=int)
    p.add_argument("--limit-eval", type=int)
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

    index = json.loads(args.index.read_text(encoding="utf-8"))
    train_rows = index["splits"]["train"]
    test_rows = index["splits"].get("test", []) or index["splits"].get("val", [])
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
    if args.limit_eval:
        test_rows = test_rows[: args.limit_eval]

    # Vocabulary is built from TRAIN captions only. Including the test split
    # would leak test vocabulary into the model and inflate BLEU.
    vocab = build_vocab([r["caption"] for r in train_rows])
    inverse = {i: w for w, i in vocab.items()}
    print(f"train {len(train_rows)} | test {len(test_rows)} | vocab {len(vocab)}")

    train_ds = LevirCC(train_rows, vocab)
    test_ds = LevirCC(test_rows, vocab)

    model = build_model(len(vocab), args.dim).to(device)
    print(f"parameters: {sum(q.numel() for q in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    state, _ = maybe_resume(args.ckpt_dir, model, optimizer, enabled=args.resume)

    write_run_metadata(args.ckpt_dir, {
        "task": "change_caption_mask_conditioned", "n_train": len(train_rows),
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
        for a, b, m, t in batches(train_ds, args.batch_size, rng):
            ab, bb = torch.from_numpy(a).to(device), torch.from_numpy(b).to(device)
            mb, tb = torch.from_numpy(m).to(device), torch.from_numpy(t).to(device)
            # Teacher forcing: predict token i+1 from tokens up to i.
            logits = model(ab, bb, mb, tb[:, :-1])
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tb[:, 1:].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * a.shape[0]
            seen += a.shape[0]
            step += 1
            if step % args.save_every == 0:
                state.step, state.epoch = step, epoch
                save_checkpoint(args.ckpt_dir, step, model, optimizer, state=state)
        print(f"epoch {epoch+1}/{args.epochs}  loss {running/max(seen,1):.4f}  "
              f"({time.time()-started:.0f}s)", flush=True)

    state.step, state.epoch = step, args.epochs
    save_checkpoint(args.ckpt_dir, step, model, optimizer, state=state)

    if test_ds and len(test_ds):
        model.eval()
        scores, samples = [], []
        for a, b, m, t in batches(test_ds, args.batch_size, rng, shuffle=False):
            ids = model.generate(
                torch.from_numpy(a).to(device), torch.from_numpy(b).to(device),
                torch.from_numpy(m).to(device),
            ).cpu().numpy()
            for produced, target in zip(ids, t):
                hyp, ref = decode(produced, inverse), decode(target, inverse)
                scores.append(bleu(hyp, [ref]))
                if len(samples) < 3:
                    samples.append((hyp, ref))

        value = float(np.mean(scores)) if scores else 0.0
        print(f"\nLEVIR-CC test BLEU-4 (sentence mean): {value:.4f}  n={len(scores)}")
        for hyp, ref in samples:
            print(f"  pred: {hyp[:70]}")
            print(f"  ref : {ref[:70]}")
        (args.ckpt_dir / "metrics.json").write_text(
            json.dumps({"bleu4_sentence_mean": value, "n": len(scores)}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
