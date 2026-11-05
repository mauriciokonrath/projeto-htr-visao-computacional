import numpy as np
import pytest

from htrcv.layout import fragment_groups


def ink_for(boxes):
    ink = np.zeros((150, 400), np.uint8)
    for x1, y1, x2, y2 in boxes:
        ink[y1 + 3 : y2 - 3, x1:x2] = 255
    return ink


def test_same_band_fragments_form_one_partition_without_duplicates():
    boxes = [[20, 30, 90, 50], [110, 30, 180, 50], [200, 30, 290, 50]]
    assert fragment_groups(boxes, ink_for(boxes), [[0, 400]]) == [[0, 1, 2]]


def test_columns_and_different_rows_never_merge():
    boxes = [[20, 30, 180, 50], [205, 30, 380, 50], [20, 51, 180, 71]]
    assert fragment_groups(boxes, ink_for(boxes), [[0, 200], [200, 400]]) == [[0], [1], [2]]


def test_large_gap_or_different_ink_centroids_reject_merge():
    boxes = [[20, 30, 90, 50], [200, 30, 290, 50]]
    assert fragment_groups(boxes, ink_for(boxes), [[0, 400]]) == [[0], [1]]
    boxes = [[20, 30, 90, 50], [110, 30, 180, 50]]
    ink = np.zeros((150, 400), np.uint8)
    ink[30:34, 20:90] = 255
    ink[45:50, 110:180] = 255
    assert fragment_groups(boxes, ink, [[0, 400]]) == [[0], [1]]


def test_empty_ink_cannot_support_a_merge():
    boxes = [[20, 30, 90, 50], [110, 30, 180, 50]]
    assert fragment_groups(boxes, np.zeros((150, 400), np.uint8), [[0, 400]]) == [[0], [1]]
    assert fragment_groups([], np.zeros((150, 400), np.uint8), [[0, 400]]) == []


def test_outside_column_is_rejected():
    with pytest.raises(ValueError):
        fragment_groups(
            [[190, 30, 210, 50]], np.zeros((150, 400), np.uint8), [[0, 200], [200, 400]]
        )
