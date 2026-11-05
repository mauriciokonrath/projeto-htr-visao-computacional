import cv2
import numpy as np
import pytest

from htrcv.classical import ClassicalConfig, estimate_skew, rotation_matrix, segment
from htrcv.metrics import aggregate_text, segmentation_counts, text_counts
from htrcv.pagexml import page_mask, parse_pagexml
from htrcv.splits import grouped_splits


@pytest.mark.parametrize(
    "reference,prediction,cer,wer",
    [
        ("abc", "abc", 0, 0),
        ("abc", "axc", 1 / 3, 1),
        ("abc", "ab", 1 / 3, 1),
        ("abc", "abcd", 1 / 3, 1),
        ("um dois", "um três", 3 / 7, 1 / 2),
        ("ação", "ac\u0327a\u0303o", 0, 0),
        ("a", "aaaa", 3, 1),
        ("A", "a", 1, 1),
    ],
)
def test_text_metrics(reference, prediction, cer, wer):
    result = text_counts(reference, prediction)
    assert result["cer"] == pytest.approx(cer)
    assert result["wer"] == pytest.approx(wer)


def test_empty_reference_and_micro_average():
    empty = text_counts("", "xyz")
    assert empty["cer"] is None and empty["char_errors"] == 3
    assert aggregate_text([empty, text_counts("ab", "ab")])["cer"] == 1.5
    assert aggregate_text([text_counts("a", "b"), text_counts("abc", "abc")])["cer"] == 0.25


def test_segmentation():
    result = segmentation_counts(np.array([1, 1, 0]), np.array([0, 1, 1]))
    assert result["iou"] == pytest.approx(1 / 3)
    assert result["dice"] == 0.5
    assert segmentation_counts(np.zeros(3), np.zeros(3))["iou"] == 1
    with pytest.raises(ValueError):
        segmentation_counts(np.ones(3), np.ones(2))


def make_xml(image="page.png", points="10,10 190,10 190,30 10,30", text="João &amp; Maria"):
    return f'''<PcGts xmlns="urn:page"><Page imageFilename="{image}" imageWidth="240" imageHeight="160">
      <TextRegion id="r"><Coords points="0,0 240,0 240,160 0,160"/>
      <TextLine id="l"><Coords points="{points}"/><TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine>
      </TextRegion></Page></PcGts>'''


def test_parser(tmp_path):
    path = tmp_path / "page.xml"
    path.write_text(make_xml(), encoding="utf-8")
    page = parse_pagexml(path)
    assert page.region_ids == ("r",)
    assert page.lines[0].transcription == "João & Maria"
    mask = page_mask(page)
    assert mask.shape == (160, 240) and mask[20, 30] == 255 and mask[0, 0] == 0


@pytest.mark.parametrize("points", ["-1,0 10,0 10,10", "0,0 241,0 10,10", "1,1 2,2 3,3"])
def test_bad_polygon(tmp_path, points):
    path = tmp_path / "page.xml"
    path.write_text(make_xml(points=points))
    with pytest.raises(ValueError):
        parse_pagexml(path)


def test_reading_order_and_transcription_alternatives(tmp_path):
    path = tmp_path / "page.xml"
    path.write_text("""<PcGts><Page imageFilename="a.png" imageWidth="100" imageHeight="100">
    <ReadingOrder><OrderedGroup id="g"><RegionRefIndexed index="1" regionRef="r1"/>
    <RegionRefIndexed index="0" regionRef="r2"/></OrderedGroup></ReadingOrder>
    <TextRegion id="r1"><TextLine id="a"><Coords points="0,0 10,0 10,10"/>
    <TextEquiv index="1"><Unicode>wrong</Unicode></TextEquiv>
    <TextEquiv index="0"><Unicode>right</Unicode></TextEquiv></TextLine></TextRegion>
    <TextRegion id="r2"><TextLine id="b"><Coords points="0,20 10,20 10,30"/></TextLine></TextRegion>
    </Page></PcGts>""")
    page = parse_pagexml(path)
    assert [line.id for line in page.lines] == ["b", "a"]
    assert page.lines[1].transcription == "right"
    assert page.lines[0].transcription is None


def test_grouped_splits_and_writer_links():
    rows = [dict(document_id=f"d{i}", writer_id="shared" if i < 2 else "") for i in range(12)]
    a = grouped_splits(rows)
    assert a == grouped_splits(list(reversed(rows)))
    assert a["d0"] == a["d1"] and set(a.values()) == {"train", "val", "test"}
    with pytest.raises(ValueError):
        grouped_splits(rows[:2])


@pytest.mark.parametrize("angle", [-8, 7])
def test_deskew_angle_sign_and_recovery(angle):
    original = np.full((600, 800), 255, np.uint8)
    for y in range(80, 550, 50):
        cv2.line(original, (100, y), (700, y), 0, 2)
    matrix, size = rotation_matrix(original.shape, angle)
    rotated = cv2.warpAffine(original, matrix, size, borderValue=255)
    config = ClassicalConfig()
    assert estimate_skew(rotated, config)["angle_deg"] == pytest.approx(-angle, abs=0.5)
    result = segment(rotated, config)
    assert abs(estimate_skew(result["recognition_image"], config)["angle_deg"]) < 0.5
    assert result["mask"].shape == rotated.shape


def test_blank_page_does_not_rotate():
    result = segment(np.full((400, 400), 255, np.uint8), ClassicalConfig())
    assert result["applied_angle"] == 0 and result["boxes"] == []


def test_connected_lines_are_not_discarded_as_one_tall_block():
    gray = np.full((500, 600), 255, np.uint8)
    for y in [100, 130, 160, 190]:
        cv2.line(gray, (80, y), (500, y), 0, 5)
    cv2.line(gray, (280, 100), (280, 190), 0, 1)
    result = segment(gray, ClassicalConfig(deskew=False, clahe=False, threshold="otsu"))
    assert len(result["boxes"]) >= 4
    for y in [100, 130, 160, 190]:
        assert any(y1 <= y < y2 for _, y1, _, y2 in result["boxes"])
