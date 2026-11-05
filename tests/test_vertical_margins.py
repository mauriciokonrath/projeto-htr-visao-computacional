import pytest

from htrcv.classical import ClassicalConfig
from htrcv.layout import add_vertical_margins


def test_zero_margin_is_exact_and_input_is_not_modified():
    boxes = [[10, 10, 50, 20], [10, 30, 50, 40]]
    padded, diagnostics = add_vertical_margins(boxes, 100, 0)
    assert padded == boxes
    assert all(d["added_top"] == d["added_bottom"] == 0 for d in diagnostics)
    add_vertical_margins(boxes, 100, 0.25)
    assert boxes == [[10, 10, 50, 20], [10, 30, 50, 40]]


def test_margin_keeps_ids_and_existing_pixels_with_neighbor_limits():
    boxes = [[10, 10, 50, 20], [10, 22, 50, 32]]
    padded, _ = add_vertical_margins(boxes, 100, 1)
    assert len(padded) == len(boxes)
    assert padded[0][3] == padded[1][1] == 21
    assert all(
        a[0] == b[0] and a[2] == b[2] and b[1] <= a[1] and b[3] >= a[3]
        for a, b in zip(boxes, padded)
    )


def test_image_bounds_and_separate_columns():
    boxes = [[0, 0, 20, 10], [50, 8, 70, 18], [50, 90, 70, 100]]
    padded, _ = add_vertical_margins(boxes, 100, 0.5)
    assert padded == [[0, 0, 20, 15], [50, 3, 70, 23], [50, 85, 70, 100]]
    assert add_vertical_margins([], 100, 0.25) == ([], [])


def test_existing_overlaps_are_not_silently_cropped():
    boxes = [[0, 10, 100, 40], [0, 30, 100, 50]]
    padded, _ = add_vertical_margins(boxes, 100, 0.25)
    assert padded[0][3] == 40 and padded[1][1] == 30


@pytest.mark.parametrize("ratio", [-0.1, float("nan"), 1.1])
def test_invalid_margin_rejected(ratio):
    with pytest.raises(ValueError):
        ClassicalConfig(layout_vertical_margin_ratio=ratio)
    with pytest.raises(ValueError):
        add_vertical_margins([], 100, ratio)
