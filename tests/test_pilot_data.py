import json

import numpy as np
import pytest
from test_workflow import make_dataset

from htrcv.io import read_csv, sha256
from htrcv.pagexml import parse_pagexml
from htrcv.pilot_data import load_pilot, prepare_pilot, reconstruct_tiles, tile_origins
from htrcv.workflow import prepare


def pilot_fixture(tmp_path):
    make_dataset(tmp_path / "raw")
    prepare(tmp_path / "raw", tmp_path / "prepared", dataset="synthetic_only")
    manifest = tmp_path / "prepared/pages.csv"
    row = next(r for r in read_csv(manifest) if r["split"] == "train")
    page = parse_pagexml(manifest.parent / row["pagexml_path"])
    review = dict(
        page_id=row["page_id"],
        split="train",
        review_status="pilot_geometry_reviewed",
        manifest_sha256=sha256(manifest),
        image_sha256=row["image_sha256"],
        pagexml_sha256=row["pagexml_sha256"],
        expected_line_ids=[line.id for line in page.lines],
        roi_polygon=[[0, 0], [220, 0], [220, 140], [0, 140]],
    )
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review))
    return manifest, path


def test_pilot_valid_pixels_and_tiles_cover_roi_without_split_leakage(tmp_path):
    manifest, review = pilot_fixture(tmp_path)
    output = tmp_path / "pilot"
    result = prepare_pilot(manifest, review, output, tile_size=64, stride=48)
    meta, rows, arrays = load_pilot(output)
    assert result == meta and meta["split"] == "train" and meta["annotated_lines"] == 1
    assert 0 < meta["positive_pixels"] < meta["valid_pixels"] < 160 * 240
    assert all(r["split"] == "train" for r in rows)
    probs = [np.ones((64, 64), np.float32) * 0.7 for r in rows]
    reconstructed, counts = reconstruct_tiles(probs, rows, arrays["image"].shape, arrays["valid"])
    assert np.allclose(reconstructed[arrays["valid"] > 0], 0.7)
    assert (counts[arrays["valid"] > 0] > 0).all() and counts.max() > 1
    with pytest.raises(FileExistsError):
        prepare_pilot(manifest, review, output)
    (output / rows[0]["valid_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Tile alterado"):
        load_pilot(output)


@pytest.mark.parametrize("mutation", ["test", "cut_polygon", "ids", "source"])
def test_invalid_pilot_review_rejected(tmp_path, mutation):
    manifest, review = pilot_fixture(tmp_path)
    spec = json.loads(review.read_text())
    if mutation == "test":
        spec["split"] = "test"
    elif mutation == "cut_polygon":
        spec["roi_polygon"] = [[0, 0], [15, 0], [15, 15], [0, 15]]
    elif mutation == "ids":
        spec["expected_line_ids"] = []
    else:
        spec["image_sha256"] = "0" * 64
    review.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        prepare_pilot(manifest, review, tmp_path / "pilot")
    assert not (tmp_path / "pilot").exists()


def test_tile_edges_and_missing_coverage():
    assert tile_origins(20, 64, 48) == [0]
    assert tile_origins(100, 64, 48) == [0, 36]
    with pytest.raises(ValueError):
        tile_origins(100, 32, 40)
    with pytest.raises(ValueError, match="sem cobertura"):
        reconstruct_tiles([np.ones((2, 2))], [dict(x=0, y=0)], (3, 3), np.ones((3, 3)))
