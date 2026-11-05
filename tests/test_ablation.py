"""Testes da comparação pareada com dados sintéticos."""

import numpy as np
import pytest
from PIL import Image

from htrcv.ablation import compare_lines
from htrcv.metrics import text_counts
from htrcv.preprocessing import preprocess_page, validate_preprocessing


def row(line, reference, prediction, document="d1"):
    return dict(
        page_id=document,
        line_id=line,
        document_id=document,
        reading_index=0,
        split="train",
        status="recognized",
        reference_raw=reference,
        reference=reference,
        prediction_raw=prediction,
        prediction=prediction,
        crop_sha256="fixture",
        **text_counts(reference, prediction),
    )


def test_pairing_counts_all_lines_and_uses_micro_not_macro():
    a = [row("one", "abcd", "axcd"), row("two", "ab", "ab", "d2")]
    b = [row("two", "ab", "a", "d2"), row("one", "abcd", "abcd")]
    summary, pairs, docs = compare_lines(a, b)
    assert summary["baseline"]["cer"] == summary["variant"]["cer"] == 1 / 6
    assert summary["delta_cer_pp"] == 0
    assert summary["improved_lines"] == summary["worsened_lines"] == 1
    assert summary["tied_lines"] == 0
    assert len(pairs) == len(docs) == 2
    assert docs[0]["delta_cer_pp"] == -25
    assert docs[1]["delta_cer_pp"] == 50


@pytest.mark.parametrize("change", ["missing", "duplicate", "reference", "count", "split", "rate"])
def test_invalid_comparisons_are_rejected(change):
    a = [row("one", "abc", "axc"), row("two", "xyz", "xyz")]
    b = [dict(r) for r in a]
    if change == "missing":
        b.pop()
    elif change == "duplicate":
        b.append(dict(b[0]))
    elif change == "reference":
        b[0] = row("one", "def", "axc")
    elif change == "count":
        b[0]["char_errors"] = 0
    elif change == "split":
        b[0]["split"] = "val"
    else:
        b[0]["cer"] = 0
    with pytest.raises(ValueError):
        compare_lines(a, b)


@pytest.mark.parametrize("field,value", [("prediction", "azc"), ("crop_sha256", "changed")])
def test_control_requires_identical_text_and_pixels(field, value):
    a = row("one", "abc", "axc")
    b = dict(a)
    b[field] = value
    if field == "prediction":
        b["prediction_raw"] = value
    with pytest.raises(ValueError, match="Controle"):
        compare_lines([a], [b], require_identical=True)


def test_clahe_preserves_image_geometry_and_identity_is_pixel_exact():
    pixels = np.tile(np.arange(100, 164, dtype=np.uint8), (32, 1))
    original = Image.fromarray(pixels)
    assert np.array_equal(np.asarray(preprocess_page(original)), pixels)
    result = preprocess_page(original, dict(method="clahe", clip_limit=2.0, tile_grid_size=[8, 8]))
    assert result.size == original.size and result.mode == "L"
    assert not np.array_equal(np.asarray(result), pixels)
    assert np.array_equal(np.asarray(original), pixels)


@pytest.mark.parametrize("clip", [float("nan"), float("inf"), 0, -1, True])
def test_invalid_contrast_configuration(clip):
    with pytest.raises(ValueError):
        validate_preprocessing(dict(method="clahe", clip_limit=clip, tile_grid_size=[8, 8]))
