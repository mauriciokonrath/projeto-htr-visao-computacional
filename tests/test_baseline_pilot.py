import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from test_pilot_data import pilot_fixture

from htrcv.baseline_pilot import (
    evaluate_baselines,
    extract_baselines,
    geometry_config,
    prepare_baseline_pilot,
    raster_line,
)
from htrcv.io import sha256
from htrcv.pilot_data import load_pilot, prepare_pilot


def config():
    return geometry_config(Path(__file__).parents[1] / "configs/pilot_baseline_geometry.json")


def scene():
    shape = (160, 240)
    annotations = [
        dict(line_id=f"l_{i}", baseline=[[20, y], [215, y]]) for i, y in enumerate((40, 80, 120))
    ]
    target = np.zeros(shape, np.uint8)
    for a in annotations:
        target |= raster_line(a["baseline"], shape, 7)
    return annotations, target, np.full(shape, 255, np.uint8)


def score(target, annotations, valid, spec=None):
    spec = config() if spec is None else spec
    extraction, labels = extract_baselines(target.astype(np.float32) / 255, valid, spec)
    return evaluate_baselines(annotations, extraction, labels, valid, spec)


def test_separated_curves_and_missing_line():
    annotations, target, valid = scene()
    result = score(target, annotations, valid)
    assert result["matched_lines"] == 3 and result["merged_candidates"] == 0
    assert result["fragmented_references"] == 0
    target[110:130] = 0
    result = score(target, annotations, valid)
    assert result["matched_lines"] == 2 and result["omitted_lines"] == 1
    assert result["per_reference"][2]["candidate_id"] is None


def test_joined_lines_are_not_multiple_recoveries():
    annotations, target, valid = scene()
    cv2.line(target, (100, 40), (100, 80), 255, 3)
    result = score(target, annotations, valid)
    assert result["candidates"] == 2 and result["matched_lines"] == 1
    assert result["merged_candidates"] == 1 and result["omitted_lines"] == 2


def test_fragments_cannot_count_as_full_lines():
    annotations, target, valid = scene()
    target[30:50, 110:130] = 0
    result = score(target, annotations, valid)
    assert result["fragmented_references"] == 1
    assert result["candidates"] == 4 and result["matched_lines"] == 2
    assert result["unassigned_candidates"] == 2


def test_tolerance_and_ignored_pixels():
    annotations, target, valid = scene()
    shifted = np.roll(target, 7, axis=0)
    assert score(shifted, annotations, valid)["matched_lines"] == 3
    assert score(np.roll(target, 14, axis=0), annotations, valid)["matched_lines"] == 0
    valid[:20] = 0
    target[:10] = 255
    assert score(target, annotations, valid)["candidates"] == 3
    empty = score(np.zeros_like(target), annotations, valid)
    assert empty["omitted_lines"] == 3 and empty["candidates"] == 0


def test_one_to_one_duplicate_reference():
    annotations, target, valid = scene()
    annotations.append(dict(line_id="duplicate", baseline=annotations[0]["baseline"]))
    result = score(target, annotations, valid)
    assert result["matched_lines"] == 3 and result["omitted_lines"] == 1
    assert result["merged_candidates"] == 1


@pytest.mark.parametrize(
    "points", [[], [[1, 2]], [[2, 2], [1, 2]], [[1, 2], [250, 3]], [[1, 2], [2, float("nan")]]]
)
def test_invalid_baselines_rejected(points):
    with pytest.raises(ValueError):
        raster_line(points, (160, 240))


def test_prepare_baseline_preserves_inputs_and_rejects_missing_baseline(tmp_path):
    manifest, review = pilot_fixture(tmp_path)
    source = tmp_path / "source"
    prepare_pilot(manifest, review, source, tile_size=64, stride=48)
    # Adiciona uma baseline sintética ao fixture.
    synthetic = json.loads((source / "annotations.json").read_text())
    synthetic[0]["baseline"] = [[12, 27], [100, 27]]
    (source / "annotations.json").write_text(json.dumps(synthetic))
    metadata = json.loads((source / "dataset.json").read_text())
    metadata["files"]["annotations.json"] = sha256(source / "annotations.json")
    (source / "dataset.json").write_text(json.dumps(metadata))
    spec = Path(__file__).parents[1] / "configs/pilot_baseline_geometry.json"
    before = sha256(source / "dataset.json")
    output = tmp_path / "derived"
    meta = prepare_baseline_pilot(source, spec, output)
    loaded, rows, arrays = load_pilot(output)
    assert meta == loaded and meta["target_kind"] == "baseline_bands"
    assert meta["parent_dataset_sha256"] == before == sha256(source / "dataset.json")
    assert meta["baseline_components"] == 1
    for name in ("annotations.json", "image.png", "valid.png", "review.json"):
        assert sha256(source / name) == sha256(output / name)
    assert 0 < meta["positive_pixels"] < meta["valid_pixels"]
    assert not np.any((arrays["target"] > 0) & (arrays["valid"] == 0))
    with pytest.raises(FileExistsError):
        prepare_baseline_pilot(source, spec, output)
    # Baseline ausente deve ser rejeitada.
    annotations = json.loads((source / "annotations.json").read_text())
    annotations[0]["baseline"] = []
    (source / "annotations.json").write_text(json.dumps(annotations))
    metadata = json.loads((source / "dataset.json").read_text())
    metadata["files"]["annotations.json"] = sha256(source / "annotations.json")
    (source / "dataset.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Baseline exige"):
        prepare_baseline_pilot(source, spec, tmp_path / "bad")
