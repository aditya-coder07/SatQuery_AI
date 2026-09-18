"""set_pixel_budget: a min_pixels above the native size upscales the frame
the model sees; max_pixels caps it; both flow through smart_resized_hw."""

from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")

from training.common import vlm_grounding as vg


def _proc():
    return SimpleNamespace(image_processor=SimpleNamespace(patch_size=14, merge_size=2, size={}))


def test_min_pixels_upscales_and_max_caps():
    p = _proc()
    h, w = vg.smart_resized_hw(p, 800, 800)
    assert (h, w) == (812, 812)  # native, rounded to the 28-multiple
    vg.set_pixel_budget(p, min_pixels=1024 * 1024)
    h, w = vg.smart_resized_hw(p, 800, 800)
    assert h == w and h >= 1024 and h % 28 == 0
    vg.set_pixel_budget(p, min_pixels=None, max_pixels=512 * 512)
    h, w = vg.smart_resized_hw(p, 800, 800)
    assert h * w <= 512 * 512 and h % 28 == 0
