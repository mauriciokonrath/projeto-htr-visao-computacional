import pytest

from htrcv import final_evaluation as fe
from htrcv.pagexml import Line, Page

REF = [
    dict(
        line_id="a",
        reference_raw="abc",
        polygon=[[0, 0], [9, 0], [9, 9], [0, 9]],
        baseline=[[0, 8], [9, 8]],
    )
]


def test_zero_candidates_keeps_deletions():
    rows, _, unused = fe.score(REF, [], [], "E1", 0.5)
    assert rows[0]["char_errors"] == 3 and rows[0]["word_errors"] == 1
    assert rows[0]["candidate_id"] is None and unused == []


def test_empty_reference_retained():
    refs = [dict(REF[0], reference_raw="")]
    rows, _, _ = fe.score(
        refs, [dict(candidate_id="p", prediction_raw="x")], [dict(bbox=[0, 0, 10, 10])], "E1", 0.5
    )
    assert rows[0]["char_errors"] == 1 and rows[0]["cer"] is None


def test_matching_uses_geometry_not_text():
    predictions = [
        dict(candidate_id="wrong_place", prediction_raw="abc"),
        dict(candidate_id="correct_place", prediction_raw="xyz"),
    ]
    geometry = [dict(bbox=[20, 20, 30, 30]), dict(bbox=[0, 0, 10, 10])]
    rows, _, unused = fe.score(REF, predictions, geometry, "E1", 0.5)
    assert rows[0]["candidate_id"] == "correct_place" and rows[0]["char_errors"] == 3
    assert unused[0]["candidate_id"] == "wrong_place"


def test_partial_failure_is_recorded(tmp_path):
    directory = tmp_path / "condition"

    def fail():
        (directory / "partial.txt").write_text("one completed candidate")
        raise RuntimeError("falha no candidato seguinte")

    state = fe.attempt(directory, fail)
    assert state["status"] == "failed"
    assert (directory / "partial.txt").exists() and (directory / "failure.json").exists()


def test_incomplete_method_has_no_global_metric():
    valid = dict(
        method="E1",
        reference_lines=1,
        metrics=dict(char_errors=0),
        matched=1,
        omitted=0,
        unassigned=0,
    )
    failed = dict(valid, metrics=None, matched=None, omitted=None, unassigned=None)
    stats = fe.summarize([valid, failed], {"E1": []}, ["E1"], 2)["E1"]
    assert stats["complete"] is False and stats["metrics"] is None
    assert stats["completed_pages"] == 1


def test_missing_transcription_invalidates_reference():
    line = Line("a", "r", ((0, 0), (9, 0), (9, 9)), ((0, 8), (9, 8)), None)
    page = Page("x", 10, 10, ("r",), (line,), "xml")
    with pytest.raises(ValueError, match="Missing reference"):
        fe.references(page)


def test_existing_directory_is_not_overwritten(tmp_path):
    directory = tmp_path / "existing"
    directory.mkdir()
    with pytest.raises(FileExistsError):
        fe.attempt(directory, lambda: {})


def test_e4_requires_identity():
    with pytest.raises(ValueError, match="identity"):
        fe.score(REF, [dict(candidate_id="wrong", prediction_raw="abc")], [{}], "E4_control", 0.5)
