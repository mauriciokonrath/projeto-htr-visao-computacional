import numpy as np
import pytest

from htrcv.classical import ClassicalConfig
from htrcv.photometry import VARIANTS, apply_treatment, normalize_background


@pytest.mark.parametrize("shape", [(12, 45), (31, 160), (62, 250)])
@pytest.mark.parametrize("level", [1, 127, 255])
def test_constant_crop_becomes_white(shape, level):
    result, kernel = normalize_background(np.full(shape, level, np.uint8))
    assert np.all(result == 255) and kernel % 2 == 1 and kernel >= 3


def test_black_crop_stays_black():
    result, _ = normalize_background(np.zeros((20, 50), np.uint8))
    assert np.all(result == 0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_treatments_preserve_geometry(variant):
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 256, (40, 200), dtype=np.uint8)
    result, _ = apply_treatment(crop, variant)
    assert result.shape == crop.shape and result.dtype == np.uint8
    if variant == "original":
        assert np.array_equal(result, crop)


def test_unknown_treatment_rejected():
    with pytest.raises(ValueError):
        apply_treatment(np.zeros((10, 10), np.uint8), "sharpen")


def test_projection_closing_enabled_by_default():
    assert ClassicalConfig().layout_projection_closing is True
