"""v3 optical-SAR fusion: pretrained dual encoders, gated per-scale fusion,
a segmentation triad, and modality dropout.

Why v3
------
v1 and v2 measured fusion as tile-level multi-label presence on 120-px
downsampled tiles, with from-scratch encoders, on a tile-random split that
leaks scenes (audit finding F1). Both found the fused head *worse* than
optical alone. Three of those choices bias the answer toward "no gain":
presence over a 512-m tile is a coarse target that optical alone saturates;
a from-scratch SAR stream on 1,548 tiles learns little; and scene leakage
rewards memorising optical texture.

v3 asks the question the way the WHU-OPT-SAR literature does (MCANet,
ASANet): **per-pixel land-cover segmentation**, 7 classes, mIoU. It keeps the
project's triad discipline - one model, three heads over shared streams, so
the only difference between arms is which modality each sees:

    A. optical-only   B. SAR-only   C. fused

and adds:

* **pretrained encoders** for both streams (ImageNet ResNet-50; the SAR
  stem is the RGB filters averaged to one channel - the standard carry-over);
* **gated late fusion at every scale**: `f = o + sigmoid(g([o, s])) * s'`
  - the network can shut SAR off where it is uninformative and let it in
  where optical is ambiguous (water/shadow, cloud, wet soil);
* **modality dropout** on the fused head at train time, so the same head
  can be scored with SAR absent (or optical absent) - which is the
  "selective modality use" the PS wants, and a robustness number.

`forward(optical, sar, drop=None)` returns three logit maps `(B,7,H,W)`.
"""

from __future__ import annotations

N_CLASSES = 7  # WHU-OPT-SAR labels 1..7; 0 is unlabeled and ignored


def build_optsar_fusion_v3(dim: int = 96, weights: str | None = "IMAGENET1K_V2",
                           n_classes: int = N_CLASSES):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from training.v2.architectures import build_pretrained_backbone

    def conv_bn(cin, cout, k=3):
        return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                             nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class FPNHead(nn.Module):
        """Lateral 1x1s + top-down sum + 3x3 smooth, to stride 4, then a classifier."""

        def __init__(self, widths):
            super().__init__()
            self.lateral = nn.ModuleList([conv_bn(w, dim, 1) for w in widths])
            self.smooth = nn.ModuleList([conv_bn(dim, dim) for _ in widths])
            self.cls = nn.Sequential(conv_bn(dim, dim), nn.Conv2d(dim, n_classes, 1))

        def forward(self, feats, size):
            h = self.smooth[-1](self.lateral[-1](feats[-1]))
            for i in range(len(feats) - 2, -1, -1):
                h = F.interpolate(h, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
                h = self.smooth[i](h + self.lateral[i](feats[i]))
            return F.interpolate(self.cls(h), size=size, mode="bilinear", align_corners=False)

    class FusionTriadV3(nn.Module):
        def __init__(self):
            super().__init__()
            self.optical = build_pretrained_backbone(cin=4, weights=weights)
            self.sar = build_pretrained_backbone(cin=1, weights=weights)
            widths = self.optical.widths
            self.head_optical = FPNHead(widths)
            self.head_sar = FPNHead(widths)
            self.head_fused = FPNHead(widths)
            # Per-scale gate: how much of the SAR feature enters the fused
            # stream at each location. Initialised near zero so training
            # starts from "optical" and must earn the SAR contribution.
            self.sar_proj = nn.ModuleList([conv_bn(w, w, 1) for w in widths])
            self.gate = nn.ModuleList([nn.Conv2d(2 * w, w, 1) for w in widths])
            for g in self.gate:
                nn.init.zeros_(g.weight)
                nn.init.constant_(g.bias, -2.0)

        def forward(self, optical, sar, drop: str | None = None):
            """`drop` in {None, "sar", "optical"} zeroes that modality's
            features in the FUSED stream only - the single-modality heads
            always see their own input, so the triad stays comparable."""
            fo = self.optical(optical)
            fs = self.sar(sar)
            size = optical.shape[-2:]
            fused = []
            for o, s, proj, gate in zip(fo, fs, self.sar_proj, self.gate):
                if drop == "sar":
                    s = torch.zeros_like(s)
                if drop == "optical":
                    o = torch.zeros_like(o)
                g = torch.sigmoid(gate(torch.cat([o, s], dim=1)))
                fused.append(o + g * proj(s))
            return (self.head_optical(fo, size), self.head_sar(fs, size),
                    self.head_fused(fused, size))

    return FusionTriadV3()
