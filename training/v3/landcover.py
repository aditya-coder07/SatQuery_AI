"""v3 land-cover encoder: SSL4EO-S12 pretrained ResNet-50 on 12 Sentinel-2 bands.

Why v3
------
Track A v2 reached mAP 0.315 from scratch on 65,867 patches (11% of
BigEarthNet v2) and 40 more epochs moved it by 0.002: capacity and schedule
were not the limit, initialisation was. Every published BigEarthNet result
above 0.80 starts from an encoder pretrained on Sentinel-2 itself. SSL4EO-S12
(Wang et al. 2022) releases exactly that - a ResNet-50 MoCo-v2 on 13-band
Sentinel-2, weights **CC-BY-4.0**, served by torchgeo as a plain timm state
dict with no remote code. The published "10% fine-tune" protocol is ~27k
patches, so our 66k subset is already in the regime those numbers use.

Band handling
-------------
SSL4EO-S12 pretrained on L1C (13 bands incl. B10 cirrus). reBEN v2 is L2A
(12 bands, no B10) in the order B01 B02 B03 B04 B05 B06 B07 B08 B8A B09
B11 B12. The stem's B10 input filter is dropped; everything else maps 1:1.

Missing bands (Cartosat-2E has 4 of the 12) are handled the way Track A
always has: `forward(x, mask, gsd)` zeroes absent bands, and training with
band dropout makes that a trained-for condition rather than a shock. The
GSD argument is accepted and ignored - BigEarthNet is single-resolution -
so the tool interface (`satquery/tools/landcover.py`) is unchanged.
"""

from __future__ import annotations

N_CLASSES = 19
SSL4EO_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8a", "B9", "B10", "B11", "B12"]
BEN12_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


def _keep_indices() -> list[int]:
    norm = [b.upper().replace("B0", "B") for b in SSL4EO_BANDS]  # B1..B12, B8A
    want = [b.upper().replace("B0", "B") for b in BEN12_BANDS]
    return [norm.index(b) for b in want]


def build_landcover_v3(pretrained: bool = True, n_classes: int = N_CLASSES, drop_rate: float = 0.2):
    import timm
    import torch
    import torch.nn as nn

    if pretrained:
        from torchgeo.models import ResNet50_Weights, resnet50

        net = resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO)  # in_chans=13, no head
        with torch.no_grad():
            w = net.conv1.weight[:, _keep_indices()].clone()
        conv = nn.Conv2d(12, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            conv.weight.copy_(w)
        net.conv1 = conv
    else:
        net = timm.create_model("resnet50", pretrained=False, in_chans=12, num_classes=0)

    class LandcoverV3(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = net
            self.drop = nn.Dropout(drop_rate)
            self.head = nn.Linear(2048, n_classes)

        def forward(self, x, mask=None, gsd=None):
            if mask is not None:
                x = x * mask[:, :, None, None]
            f = self.backbone.forward_features(x) if hasattr(self.backbone, "forward_features") else self.backbone(x)
            if f.dim() == 4:
                f = f.mean(dim=(2, 3))
            return self.head(self.drop(f))

    return LandcoverV3()
