import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from test_core import make_xml

from htrcv.io import read_csv, save_image, write_csv
from htrcv.workflow import coverage_metrics, evaluate_transcriptions, prepare, run_classical


def make_dataset(root):
    for i in range(6):
        folder = root / f"document_{i}"
        folder.mkdir(parents=True)
        gray = np.full((160, 240), 255, np.uint8)
        cv2.putText(gray, "TEST", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1)
        save_image(folder / "page.png", gray)
        (folder / "page.xml").write_text(make_xml(text="TEST"), encoding="utf-8")


def test_real_workflow_with_synthetic_fixture(tmp_path):
    raw, processed, output = tmp_path / "raw", tmp_path / "prepared", tmp_path / "run"
    make_dataset(raw)
    info = prepare(raw, processed, dataset="synthetic_test_fixture")
    assert info["pages"] == 6
    config = Path(__file__).parents[1] / "configs" / "classical.yaml"
    result = run_classical(processed / "pages.csv", config, output)
    assert result["pages"] == 1 and result["cer"] is None and result["htr_status"] == "not_run"
    assert result["micro_iou"] is None and result["evaluated_pages"] == 0
    reviewed = read_csv(processed / "coverage_review.csv")
    for row in reviewed:
        row["coverage"] = "full_page"  # O fixture sintético tem anotações completas.
    write_csv(processed / "coverage_review.csv", reviewed)
    scored = run_classical(
        processed / "pages.csv",
        config,
        tmp_path / "scored",
        coverage_review=processed / "coverage_review.csv",
    )
    assert 0 <= scored["micro_iou"] <= 1 and scored["evaluated_pages"] == 1
    assert json.loads((output / "status.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        prepare(raw, processed)
    page = read_csv(processed / "pages.csv")
    val = next(row for row in page if row["split"] == "val")
    image = processed / val["image_path"]
    image.write_bytes(image.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="alterado"):
        run_classical(processed / "pages.csv", config, tmp_path / "run_changed")


def test_page_omissions_are_not_silently_excluded(tmp_path):
    reference, hypothesis = tmp_path / "ref.csv", tmp_path / "hyp.csv"
    write_csv(reference, [dict(page_id="a", text="hello"), dict(page_id="b", text="world")])
    write_csv(hypothesis, [dict(page_id="a", text="hello")])
    with pytest.raises(ValueError, match="mesmas páginas"):
        evaluate_transcriptions(reference, hypothesis, tmp_path / "out")
    write_csv(hypothesis, [dict(page_id="a", text="hello"), dict(page_id="b", text="")])
    result = evaluate_transcriptions(reference, hypothesis, tmp_path / "out")
    assert result["cer"] == result["wer"] == 0.5


def test_xml_cannot_reference_outside_raw(tmp_path):
    raw = tmp_path / "raw"
    make_dataset(raw)
    xml = raw / "document_0" / "page.xml"
    xml.write_text(make_xml(image="../../outside.png"))
    with pytest.raises(ValueError, match="fora"):
        prepare(raw, tmp_path / "out")


def test_unannotated_text_outside_roi_is_ignored(tmp_path):
    gt = np.array([[255, 0, 0]], np.uint8)
    prediction = np.array([[255, 0, 255]], np.uint8)
    save_image(tmp_path / "roi.png", np.array([[255, 255, 0]], np.uint8))
    assert (
        coverage_metrics(gt, prediction, {"coverage": "roi", "roi_path": "roi.png"}, tmp_path)[
            "iou"
        ]
        == 1
    )
    assert coverage_metrics(gt, prediction, {"coverage": "full_page"}, tmp_path)["iou"] == 0.5
    assert coverage_metrics(gt, prediction, {}, tmp_path)["iou"] is None


def test_invalid_coverage_rejected(tmp_path):
    with pytest.raises(ValueError):
        coverage_metrics(np.zeros((2, 2)), np.zeros((2, 2)), {"coverage": "typo"}, tmp_path)
