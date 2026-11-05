import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from htrcv.classical import rectangle_geometry
from htrcv.recognition import build_predicted_lines


def setup_geometry():
    image = Image.fromarray(np.full((100, 200), 255, np.uint8))
    valid = np.zeros((100, 200), np.uint8)
    valid[10:91, 10:191] = 255
    candidates = [dict(candidate_id="b0", baseline=[[20, 50], [50, 50.5], [100, 50], [180, 50]])]
    config = json.loads(
        (Path(__file__).parents[1] / "configs/pilot_baseline_crops.json").read_text()
    )
    return image, valid, candidates, config


def test_predicted_geometry_uses_image_neighbors_roi_not_gt():
    image, valid, candidates, config = setup_geometry()
    captured = {}

    def polygonizer(**kwargs):
        captured.update(kwargs)
        return [[[20, 30], [180, 30], [180, 65], [20, 65]]]

    lines, geometry = build_predicted_lines(image, valid, candidates, config, polygonizer)
    assert lines[0].transcription is None and lines[0].id == "b0"
    assert geometry[0]["source_baseline"] == candidates[0]["baseline"]
    assert geometry[0]["polygon_pixels_outside_roi"] == 0
    assert captured["baselines"] == [[[20, 50], [180, 50]]]
    assert len(captured["suppl_obj"]) == 1 and captured["raise_on_error"]
    assert captured["topline"] is False


@pytest.mark.parametrize("mode", ["none", "missing", "outside", "zero_area", "exception"])
def test_polygonization_failure_not_repaired_with_gt(mode):
    image, valid, candidates, config = setup_geometry()

    def broken(**kwargs):
        if mode == "none":
            return [None]
        if mode == "missing":
            return []
        if mode == "outside":
            return [[[0, 30], [180, 30], [180, 65], [0, 65]]]
        if mode == "zero_area":
            return [[[20, 50], [50, 50], [180, 50]]]
        raise RuntimeError("polygonizer failed")

    with pytest.raises((ValueError, RuntimeError)):
        build_predicted_lines(image, valid, candidates, config, broken)


def test_rectangle_baseline_at_75_percent_height():
    line = rectangle_geometry([10, 20, 50, 40], "predicted")
    assert line.transcription is None
    assert line.polygon == ((10, 20), (49, 20), (49, 39), (10, 39))
    assert line.baseline == ((10, 34), (49, 34))
    with pytest.raises(ValueError):
        rectangle_geometry([1, 1, 4, 4], "small")
