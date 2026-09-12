"""v3 change detector: shape contract and date-swap symmetry, no download."""

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def model():
    from training.v3.change_mask import build_change_mask_v3

    m = build_change_mask_v3(dim=16, weights=None)
    m.eval()
    return m


def test_output_matches_input_resolution(model):
    a, b = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        out = model(a, b)
    assert out.shape == (2, 1, 64, 64)


def test_symmetric_under_date_swap(model):
    a, b = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(a, b), model(b, a), atol=1e-5)


def test_identical_dates_give_no_change_signal_path(model):
    # |fa - fb| is exactly zero, so the output is a constant bias map.
    a = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        out = model(a, a)
    assert torch.allclose(out, out.mean(), atol=1e-4)


def test_train_change_mask_routes_v3():
    from training.train_change_mask import build_model

    import training.v3.change_mask as v3

    called = {}

    def fake(dim=64, weights="IMAGENET1K_V2"):
        called["dim"] = dim
        return torch.nn.Identity()

    orig = v3.build_change_mask_v3
    v3.build_change_mask_v3 = fake
    try:
        build_model(32, arch="v3")
    finally:
        v3.build_change_mask_v3 = orig
    assert called == {"dim": 32}
