import numpy as np
import pytest

from htrcv.postprocessing import CONFIG, filter_candidates, guard_candidates


def candidate(cid, baseline):
    return dict(candidate_id=cid, baseline=baseline, bbox=[0, 0, 100, 100])


def test_paper_support_preserves_dark_text_holes_and_rejects_exterior():
    gray = np.zeros((200, 200), np.uint8)
    gray[25:175, 25:175] = 200
    gray[80:95, 50:150] = 20
    extraction = dict(
        candidates=[
            candidate("text", [[50, 90], [149, 90]]),
            candidate("outside", [[50, 185], [149, 185]]),
            candidate("top", [[20, 1], [180, 1]]),
        ]
    )
    result, audit, mask, info = filter_candidates(gray, extraction, CONFIG)
    assert [c["candidate_id"] for c in result["candidates"]] == ["text"]
    assert mask[90, 100] and not info["abstained"]
    assert audit[1]["excluded"] and audit[2]["excluded"]
    assert len(extraction["candidates"]) == 3


def test_small_bright_component_abstains_instead_of_rejecting_whole_page():
    gray = np.zeros((200, 200), np.uint8)
    gray[90:110, 90:110] = 255
    extraction = dict(candidates=[candidate("middle", [[30, 100], [170, 100]])])
    result, _, mask, info = filter_candidates(gray, extraction, CONFIG)
    assert info["abstained"] and mask.all() and len(result["candidates"]) == 1


def test_interior_curves_exact_and_boundary_guard_small_without_mutation():
    candidates = [
        dict(candidate_id="interior", baseline=[[10, 10], [80, 10]]),
        dict(candidate_id="edge", baseline=[[0, 50], [4, 50], [99, 50]]),
    ]
    result, audit = guard_candidates(candidates, (100, 100))
    assert result[0] == candidates[0] and not audit[0]["changed"]
    assert result[1]["baseline"] == [[2, 50], [4, 50], [97, 50]]
    assert candidates[1]["baseline"][0][0] == 0
    assert audit[1]["max_coordinate_displacement_before_duplicate_collapse"] == 2


def test_duplicate_boundary_points_collapse_but_candidate_never_silently_drops():
    result, audit = guard_candidates(
        [dict(candidate_id="edge", baseline=[[0, 40], [1, 42], [4, 44], [90, 45]])], (100, 100)
    )
    assert result[0]["baseline"][:2] == [[2, 41], [4, 44]] and audit[0]["crop_points"] == 3
    with pytest.raises(ValueError):
        guard_candidates([dict(candidate_id="tiny", baseline=[[0, 0], [1, 1]])], (100, 100))
