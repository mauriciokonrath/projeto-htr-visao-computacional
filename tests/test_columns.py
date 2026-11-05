import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pytest

from htrcv.baseline_pilot import extract_baselines
from htrcv.classical import ClassicalConfig, rotation_matrix
from htrcv.columns import barrier_mask, image_barriers

CONFIG = json.loads(
    (Path(__file__).parents[1] / "configs/pilot_baseline_geometry.json").read_text()
)


def test_no_gutter_keeps_extraction_exact_and_blank_image_abstains():
    p = np.zeros((80, 220), np.float32)
    p[30:37, 10:210] = 1
    full = np.ones(p.shape, np.uint8)
    allowed, info = image_barriers(
        np.full(p.shape, 255, np.uint8), asdict(ClassicalConfig(layout_refinement=True))
    )
    assert info["cuts_work"] == [] and allowed.all()
    a, la = extract_baselines(p, full, CONFIG)
    b, lb = extract_baselines(p, allowed, CONFIG)
    assert a == b and np.array_equal(la, lb)


@pytest.mark.parametrize("angle", [-15, 0, 15])
def test_rotated_barrier_blocks_eight_connected_path_even_after_closing(angle):
    p = np.ones((100, 300), np.float32)
    matrix, size = rotation_matrix(p.shape, angle)
    mask = barrier_mask(p.shape, matrix, [size[0] / 2])
    _, labels = extract_baselines(p, mask, CONFIG)
    assert not labels[mask == 0].any()
    assert cv2.connectedComponents((labels > 0).astype(np.uint8), connectivity=8)[0] == 3


def test_barrier_rejects_scaled_or_nonfinite_geometry():
    for matrix in [[[2, 0, 0], [0, 1, 0]], [[1, 0, float("nan")], [0, 1, 0]]]:
        with pytest.raises(ValueError):
            barrier_mask((80, 220), matrix, [100])
