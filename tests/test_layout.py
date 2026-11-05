import cv2
import numpy as np
import pytest

from htrcv.classical import ClassicalConfig, segment
from htrcv.line_matching import box_iou_matrix, match_annotated_boxes


def test_two_columns_and_border_without_using_gt():
    gray = np.full((600, 800), 255, np.uint8)
    cv2.rectangle(gray, (2, 2), (797, 597), 0, 4)
    for y in range(100, 401, 50):
        cv2.rectangle(gray, (80, y), (340, y + 8), 0, -1)
        cv2.rectangle(gray, (460, y), (720, y + 8), 0, -1)
    config = ClassicalConfig(deskew=False, clahe=False, threshold="otsu", layout_refinement=True)
    result = segment(gray, config)
    assert len(result["layout"]["gutter_cuts_work"]) == 1
    cut = result["layout"]["gutter_cuts_work"][0]
    assert 340 < cut < 460
    assert len(result["boxes"]) == 14
    assert all(x2 <= cut or x1 >= cut for x1, _, x2, _ in result["boxes"])
    assert all(y1 > 50 and y2 < 500 for _, y1, _, y2 in result["boxes"])
    assert np.array_equal(result["recognition_image"], gray)


def test_blank_refined_layout_is_empty():
    result = segment(np.full((400, 500), 255, np.uint8), ClassicalConfig(layout_refinement=True))
    assert result["boxes"] == [] and result["layout"]["gutter_cuts_work"] == []


def test_single_column_is_not_cut_at_word_gaps():
    gray = np.full((600, 800), 255, np.uint8)
    for i, y in enumerate(range(100, 401, 50)):
        for x in range(70 + i * 3, 700, 95):
            cv2.rectangle(gray, (x, y), (x + 65, y + 10), 0, -1)
    result = segment(
        gray, ClassicalConfig(deskew=False, clahe=False, threshold="otsu", layout_refinement=True)
    )
    assert result["layout"]["gutter_cuts_work"] == []


def test_one_candidate_cannot_cover_two_annotations_in_count():
    result = match_annotated_boxes([[0, 0, 10, 10], [0, 10, 10, 20]], [[0, 0, 10, 20]])
    assert result["matched_annotations"] == 1 and result["unmatched_annotations"] == 1
    assert result["unassigned_candidates"] == 0


def test_matching_augments_instead_of_losing_a_valid_pair():
    # A primeira referência tem dois candidatos válidos; a segunda, só o compartilhado.
    result = match_annotated_boxes(
        [[0, 0, 10, 10], [4, 0, 14, 10]], [[1, 0, 11, 10], [0, 0, 6, 10]], threshold=0.5
    )
    assert result["matched_annotations"] == 2
    result = match_annotated_boxes(
        [[4, 0, 14, 10], [0, 0, 10, 10]], [[1, 0, 11, 10], [0, 0, 6, 10]], threshold=0.5
    )
    assert result["matched_annotations"] == 2


def test_empty_candidates_are_all_misses_not_dropped():
    result = match_annotated_boxes([[0, 0, 10, 10]], [])
    assert result["annotation_match_rate"] == 0 and result["unmatched_annotations"] == 1
    assert result["per_reference"][0]["best_box_iou"] == 0
    assert box_iou_matrix([], []).shape == (0, 0)
    with pytest.raises(ValueError):
        box_iou_matrix([[0, 0, 0, 1]], [])
