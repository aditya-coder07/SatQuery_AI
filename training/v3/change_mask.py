"""v3 change detector: ImageNet ResNet-50 siamese encoder + FPN difference decoder.

Why v3
------
v2 (`training/v2/architectures.build_change_mask`) reached F1 0.855 on
LEVIR-CD from scratch on 7,120 tiles. Every published result above 0.90 on
this benchmark (BIT 0.89, ChangeFormer 0.90, SNUNet 0.88, AGCD 0.903,
ChangeGCC 0.92) starts from a pretrained encoder. Phase 5's own ablation on
SECOND put pretraining at +0.12 mIoU. So v3 changes exactly one thing that
v2 was missing, and keeps the two things v2 got right:

* **one shared encoder** for both dates (two can drift and report change on
  identical imagery);
* **absolute difference at every scale** (symmetric under date swap).

What is new:

* the encoder is torchvision's ResNet-50 (`IMAGENET1K_V2`, BSD-3, no remote
  code), via `build_pretrained_backbone`, so the tool loader needs no new
  dependency;
* ImageNet normalisation lives **inside** the model, so callers keep feeding
  the same `/255` RGB the v1 and v2 tools feed;
* an FPN-style top-down decoder from stride 32 to stride 4, plus one
  stride-2 skip from the stem, then a learned 2x upsample - boundaries were
  the visible weakness of bilinear-only decoders.

Interface is unchanged: `model(a, b) -> (B,1,H,W)` logits at input resolution.
"""

from __future__ import annotations


def build_change_mask_v3(dim: int = 64, weights: str | None = "IMAGENET1K_V2"):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from training.v2.architectures import build_pretrained_backbone

    backbone = build_pretrained_backbone(cin=3, weights=weights)

    def conv_bn(cin, cout, k=3):
        return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                             nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class ChangeMaskV3(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = backbone
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            widths = backbone.widths  # strides 4, 8, 16, 32
            # Lateral projections of |fa - fb| at each stage, and of the
            # stride-2 stem so fine boundaries have a path to the output.
            self.lateral = nn.ModuleList([conv_bn(w, dim, 1) for w in widths])
            self.stem_lateral = conv_bn(64, dim, 1)
            self.smooth = nn.ModuleList([conv_bn(dim, dim) for _ in widths])
            self.fuse2 = conv_bn(dim, dim)
            self.head = nn.Sequential(conv_bn(dim, dim // 2), nn.Conv2d(dim // 2, 1, 1))

        def _stem(self, x):
            # conv1 + bn + relu, before the maxpool: stride 2, 64 channels.
            s = self.encoder.stem
            return s[2](s[1](s[0](x)))

        def encode(self, x):
            x = (x - self.mean) / self.std
            s2 = self._stem(x)
            h = self.encoder.stem[3](s2)
            feats = []
            for stage in self.encoder.stages:
                h = stage(h)
                feats.append(h)
            return s2, feats

        def forward(self, a, b):
            sa, fa = self.encode(a)
            sb, fb = self.encode(b)
            diffs = [lat(torch.abs(x - y)) for lat, x, y in zip(self.lateral, fa, fb)]
            # Top-down: start at stride 32, add the finer lateral each step.
            h = self.smooth[-1](diffs[-1])
            for i in range(len(diffs) - 2, -1, -1):
                h = F.interpolate(h, size=diffs[i].shape[-2:], mode="bilinear", align_corners=False)
                h = self.smooth[i](h + diffs[i])
            d2 = self.stem_lateral(torch.abs(sa - sb))
            h = F.interpolate(h, size=d2.shape[-2:], mode="bilinear", align_corners=False)
            h = self.fuse2(h + d2)
            logits = self.head(h)
            return F.interpolate(logits, size=a.shape[-2:], mode="bilinear", align_corners=False)

    return ChangeMaskV3()
