"""Teste de integração da avaliação E0–E4 com modelos simulados.

O reconhecedor devolve a transcrição da linha anotada mais próxima, de modo que
os erros medidos vêm apenas da localização das linhas.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from htrcv.final_evaluation import Models, execute, select_pages
from htrcv.io import write_csv
from htrcv.kraken_backend import Recognition
from htrcv.pagexml import Line

WIDTH, HEIGHT = 600, 400
TEXTS = ["primeira linha", "segunda linha", "terceira linha"]
BASELINES = [100, 180, 260]
X0, X1 = 60, 540


def write_page(folder):
    image = np.full((HEIGHT, WIDTH), 235, np.uint8)
    lines = []
    for i, (text, y) in enumerate(zip(TEXTS, BASELINES)):
        cv2.putText(image, text.upper(), (X0, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 20, 4)
        polygon = f"{X0},{y - 45} {X1},{y - 45} {X1},{y + 12} {X0},{y + 12}"
        lines.append(
            f'<TextLine id="l{i}"><Coords points="{polygon}"/>'
            f'<Baseline points="{X0},{y} {X1},{y}"/>'
            f"<TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine>"
        )
    Image.fromarray(image).save(folder / "page.png")
    (folder / "page.xml").write_text(
        f'<PcGts xmlns="urn:page"><Page imageFilename="page.png" imageWidth="{WIDTH}" '
        f'imageHeight="{HEIGHT}"><TextRegion id="r"><Coords points="0,0 {WIDTH},0 '
        f'{WIDTH},{HEIGHT} 0,{HEIGHT}"/>{"".join(lines)}</TextRegion></Page></PcGts>',
        encoding="utf-8",
    )


class FakeRecognizer:
    details = dict(backend="simulado")

    def recognize_line(self, image, line):
        y = np.mean([p[1] for p in line.baseline])
        nearest = int(np.argmin([abs(y - b) for b in BASELINES]))
        x1, y1, x2, y2 = line.bbox
        return Recognition(TEXTS[nearest], image.crop((x1, y1, x2, y2)), 1.0)


class FakeNative:
    """Encontra só as duas primeiras linhas, para gerar uma omissão."""

    details = dict(backend="simulado")

    def segment(self, rgb):
        lines = []
        for i, y in enumerate(BASELINES[:2]):
            polygon = ((X0, y - 45), (X1, y - 45), (X1, y + 12), (X0, y + 12))
            lines.append(Line(f"e0_{i:04d}", "native", polygon, ((X0, y), (X1, y)), None))
        return lines, [dict(candidate_id=line.id) for line in lines]


def baseline_probability(gray):
    probability = np.zeros(gray.shape, np.float32)
    for y in BASELINES:
        probability[y - 3 : y + 4, X0:X1] = 1
    return probability, np.ones(gray.shape, np.uint8)


def box_polygonizer(im, baselines, **kwargs):
    polygons = []
    for baseline in baselines:
        xs = [p[0] for p in baseline]
        y = int(round(np.mean([p[1] for p in baseline])))
        polygons.append(
            [[min(xs), y - 45], [max(xs), y - 45], [max(xs), y + 12], [min(xs), y + 12]]
        )
    return polygons


def load_plan():
    config = json.loads((Path(__file__).parents[1] / "configs/avaliacao_final.json").read_text())
    config["manifest_path"] = "data/pages.csv"
    config["models"] = dict(recognizer="none", pilot_unet="none")
    return config


def setup(tmp_path):
    data = tmp_path / "data"
    (data / "doc").mkdir(parents=True)
    write_page(data / "doc")
    write_csv(
        data / "pages.csv",
        [
            dict(
                page_id="p1",
                document_id="doc",
                image_path="doc/page.png",
                pagexml_path="doc/page.xml",
                split="test",
                width=WIDTH,
                height=HEIGHT,
            )
        ],
    )
    return select_pages(data / "pages.csv", "test"), load_plan()


def models(probability=baseline_probability):
    return Models(FakeRecognizer(), FakeNative(), probability, box_polygonizer)


def test_all_conditions_are_scored(tmp_path):
    selected, config = setup(tmp_path)
    summary = execute(tmp_path, tmp_path / "run", selected, config, models())
    assert all(s["complete"] for s in summary.values())

    control = summary["E4_control"]
    assert control["matched"] == 3 and control["metrics"]["cer"] == 0

    native = summary["E0_native"]
    assert native["matched"] == 2 and native["omitted"] == 1
    assert native["metrics"]["char_errors"] == len(TEXTS[2])

    for method in ("E2", "E3"):
        assert summary[method]["matched"] == 3 and summary[method]["metrics"]["cer"] == 0
    # E1 depende do layout sintético e da versão do OpenCV; aqui só se exige a pontuação.
    assert summary["E1"]["reference_lines_scored"] == 3
    status = json.loads((tmp_path / "run/status.json").read_text())
    assert status["status"] == "complete" and status["conditions_scored"] == 5


def test_annotations_are_read_after_all_predictions(tmp_path):
    selected, config = setup(tmp_path)
    execute(tmp_path, tmp_path / "run", selected, config, models())
    events = [
        json.loads(s)["event"] for s in (tmp_path / "run/events.jsonl").read_text().splitlines()
    ]
    first_reference = events.index("reference_parse_start")
    assert events.count("prediction_attempt_finished") == 4
    assert all(e != "prediction_attempt_finished" for e in events[first_reference:])


def test_neural_failure_only_affects_neural_conditions(tmp_path):
    def broken(gray):
        raise RuntimeError("falha simulada na U-Net")

    selected, config = setup(tmp_path)
    summary = execute(tmp_path, tmp_path / "run", selected, config, models(broken))
    assert not summary["E2"]["complete"] and summary["E2"]["metrics"] is None
    assert not summary["E3"]["complete"] and summary["E3"]["metrics"] is None
    assert summary["E4_control"]["complete"] and summary["E0_native"]["complete"]
    assert (tmp_path / "run/p1/neural_failure.json").exists()
    status = json.loads((tmp_path / "run/status.json").read_text())
    assert status["status"] == "complete_with_failures"
