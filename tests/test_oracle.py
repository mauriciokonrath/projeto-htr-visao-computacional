"""Testes do controle E4 com reconhecedor simulado."""

import json
from pathlib import Path

import pytest
import yaml
from test_workflow import make_dataset

from htrcv import oracle
from htrcv.io import sha256
from htrcv.kraken_backend import Recognition, load_htr_config
from htrcv.workflow import prepare


@pytest.fixture
def oracle_inputs(tmp_path):
    raw = tmp_path / "raw"
    make_dataset(raw)
    for xml in raw.rglob("*.xml"):
        xml.write_text(
            xml.read_text().replace("<TextEquiv>", '<Baseline points="10,25 190,25"/><TextEquiv>')
        )
    prepared = tmp_path / "prepared"
    prepare(raw, prepared, dataset="synthetic_unit_test")
    model = tmp_path / "fake-model-for-tests.bin"
    model.write_bytes(b"not a real model; testing checksum enforcement only")
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/htr.yaml").read_text())
    config.update(model_path=str(model), model_sha256=sha256(model))
    config_path = tmp_path / "htr.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return prepared / "pages.csv", config_path, model


class FakeRecognizer:
    details = {"backend": "fake_for_unit_tests"}
    text = "TEST"

    def __init__(self, model, config):
        pass

    def recognize_line(self, image, line):
        return Recognition(self.text, image.crop(line.bbox), 0.5)


@pytest.mark.parametrize("prediction,expected", [("TEST", 0.0), ("", 1.0)])
def test_oracle_line_aggregation_and_empty_prediction(
    oracle_inputs, tmp_path, monkeypatch, prediction, expected
):
    manifest, config, model = oracle_inputs

    class Recognizer(FakeRecognizer):
        text = prediction

    monkeypatch.setattr(oracle, "KrakenBackend", Recognizer)
    output = tmp_path / "run"
    result = oracle.run_oracle(manifest, config, output)
    assert result["cer"] == result["wer"] == expected
    assert result["lines"] == 1 and result["failed_lines"] == 0
    assert result["model_sha256"] == sha256(model)
    record = json.loads((output / "predictions.jsonl").read_text())
    assert sha256(output / record["crop_path"]) == record["crop_sha256"]
    assert json.loads((output / "status.json").read_text())["status"] == "complete"


def test_checkpoint_mismatch_refused(oracle_inputs):
    _, config, model = oracle_inputs
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Hash"):
        load_htr_config(config)


def test_failures_do_not_become_success_metrics(oracle_inputs, tmp_path, monkeypatch):
    manifest, config, _ = oracle_inputs

    class Broken(FakeRecognizer):
        def recognize_line(self, image, line):
            raise RuntimeError("test extraction failure")

    monkeypatch.setattr(oracle, "KrakenBackend", Broken)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="extraction"):
        oracle.run_oracle(manifest, config, output)
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    assert not (output / "metrics.json").exists()


def test_missing_transcription_is_not_omitted(tmp_path):
    raw = tmp_path / "raw"
    make_dataset(raw)
    for xml in raw.rglob("*.xml"):
        xml.write_text(
            xml.read_text().replace(
                "<TextEquiv><Unicode>TEST</Unicode></TextEquiv>",
                '<Baseline points="10,25 190,25"/>',
            )
        )
    prepared = tmp_path / "prepared"
    prepare(raw, prepared, dataset="synthetic_test")
    model = tmp_path / "fake.bin"
    model.write_bytes(b"test")
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/htr.yaml").read_text())
    config.update(model_path=str(model), model_sha256=sha256(model))
    config_path = tmp_path / "htr.yaml"
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="GT textual ausente"):
        oracle.run_oracle(prepared / "pages.csv", config_path, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_oracle_refuses_source_mutation(oracle_inputs, tmp_path):
    manifest, config, _ = oracle_inputs
    for xml in (tmp_path / "raw").rglob("*.xml"):
        xml.write_text(xml.read_text() + " ")
    with pytest.raises(ValueError, match="alterado"):
        oracle.run_oracle(manifest, config, tmp_path / "run")


def test_contrast_does_not_touch_validation(oracle_inputs, tmp_path):
    manifest, config, _ = oracle_inputs
    with pytest.raises(ValueError, match="treino"):
        oracle.run_oracle(
            manifest,
            config,
            tmp_path / "run",
            split="val",
            preprocessing=dict(method="clahe", clip_limit=2.0, tile_grid_size=[8, 8]),
        )
    assert not (tmp_path / "run").exists()
