"""Ablações de visão computacional nas páginas de treino do iForal.

fotometria: reconhece os mesmos recortes de linha (extraídos com a geometria
    anotada) nas variantes original, CLAHE, correção de fundo e fundo + CLAHE.
geometria: fatorial 2x2 da correção de inclinação (Hough) e do fechamento
    morfológico do perfil horizontal na segmentação clássica.

Uso, a partir da raiz do repositório:

    python experiments/ablacoes.py fotometria --lines runs/contraste/control/per_line.csv \\
        --output runs/ablacoes/fotometria
    python experiments/ablacoes.py geometria --output runs/ablacoes/geometria
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from htrcv.classical import ClassicalConfig, segment
from htrcv.line_matching import match_annotated_boxes
from htrcv.metrics import aggregate_text, text_counts
from htrcv.pagexml import parse_pagexml
from htrcv.photometry import VARIANTS, apply_treatment

ROOT = Path(__file__).resolve().parents[1]
GEOMETRIC_VARIANTS = ("H1_M1", "H0_M1", "H1_M0", "H0_M0")
COUNTS = ("reference_count", "candidate_count", "matched", "omitted", "unassigned")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def train_pages(manifest):
    return [p for p in read_csv(manifest) if p["split"] == "train"]


def load_recognizer(model_path):
    """Reconhecedor Kraken aplicado diretamente a um recorte de linha já extraído."""
    import torch
    from kraken.containers import BaselineLine, BaselineOCRRecord
    from kraken.lib import models
    from kraken.lib.dataset import ImageInputTransforms

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    net = models.load_any(str(model_path), train=False, device="cpu")
    batch, channels, height, width = net.nn.input
    transform = ImageInputTransforms(batch, height, width, channels, (16, 0), False)
    dummy = BaselineLine(
        id="crop", baseline=[(0, 1), (2, 1)], boundary=[(0, 0), (2, 0), (2, 2), (0, 2)]
    )

    def recognize(array):
        tensor = transform(Image.fromarray(array))
        if tensor.max() == tensor.min():
            raise ValueError("Entrada vazia após a conversão para tensor")
        with torch.inference_mode():
            predictions = net.predict(tensor.unsqueeze(0))[0]
        text = "".join(p[0] for p in predictions)
        record = BaselineOCRRecord(
            text, [(0, 1) for _ in predictions], [float(p[3]) for p in predictions], dummy
        )
        return record.logical_order().prediction

    return recognize


def summarize_lines(results):
    documents = {}
    for doc in sorted({r["document_id"] for r in results}):
        subset = [r for r in results if r["document_id"] == doc]
        documents[doc] = dict(lines=len(subset), **aggregate_text(subset))
    return dict(
        lines=len(results),
        micro=aggregate_text(results),
        documents=documents,
        macro_cer=float(np.mean([d["cer"] for d in documents.values()])),
        macro_wer=float(np.mean([d["wer"] for d in documents.values()])),
    )


def photometric(args):
    samples = read_csv(args.lines)
    base = Path(args.lines).parent
    recognize = load_recognizer(args.model)
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for variant in VARIANTS:
        folder = args.output / variant
        (folder / "crops").mkdir(parents=True)
        results = []
        with (folder / "per_line.jsonl").open("x") as log:
            for i, row in enumerate(samples):
                crop = np.array(Image.open(base / row["crop_path"]).convert("L"))
                treated, kernel = apply_treatment(crop, variant)
                destination = folder / "crops" / f"{i:04}.png"
                Image.fromarray(treated).save(destination)
                start = time.monotonic()
                hypothesis = recognize(treated)
                elapsed = time.monotonic() - start
                # Controle: sem tratamento, a predição deve ser idêntica à registrada.
                if variant == "original" and hypothesis != row["prediction_raw"]:
                    raise RuntimeError(f"Controle não reproduzido na linha {i}")
                result = dict(
                    index=i,
                    page_id=row["page_id"],
                    document_id=row["document_id"],
                    line_id=row["line_id"],
                    reference_raw=row["reference_raw"],
                    prediction_raw=hypothesis,
                    crop_sha256=sha256(destination),
                    kernel=kernel,
                    seconds=elapsed,
                    **text_counts(row["reference_raw"], hypothesis),
                )
                log.write(json.dumps(result, ensure_ascii=False) + "\n")
                results.append(result)
        summaries[variant] = summarize_lines(results)
        write_json(folder / "summary.json", summaries[variant])
        print(variant, summaries[variant]["micro"], flush=True)
    write_json(args.output / "summary.json", summaries)


def geometric(args):
    config = json.loads(args.config.read_text())["classical_config"]
    pages = train_pages(ROOT / args.manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    prepared = (ROOT / args.manifest).parent
    results = []
    for page in pages:
        image = np.array(Image.open(prepared / page["image_path"]).convert("L"))
        annotation = parse_pagexml(prepared / page["pagexml_path"])
        references = [list(line.bbox) for line in annotation.lines]
        for name in GEOMETRIC_VARIANTS:
            cfg = ClassicalConfig(**config)
            cfg.deskew = name[1] == "1"
            cfg.layout_projection_closing = name[-1] == "1"
            seg = segment(image, cfg)
            boxes = []
            for polygon in seg["polygons"]:
                xy = np.array(polygon)
                boxes.append([*xy.min(0), *(xy.max(0) + 1)])
            # Comparação no sistema de coordenadas original da página.
            match = match_annotated_boxes(references, boxes, 0.5)
            out = args.output / page["page_id"] / name
            out.mkdir(parents=True)
            write_json(
                out / "candidates.json",
                dict(boxes=boxes, polygons=seg["polygons"], reference_boxes=references),
            )
            write_json(out / "matching.json", match)
            result = dict(
                page_id=page["page_id"],
                document_id=page["document_id"],
                variant=name,
                reference_count=len(references),
                candidate_count=len(boxes),
                matched=match["matched_annotations"],
                omitted=match["unmatched_annotations"],
                unassigned=match["unassigned_candidates"],
                angle_deg=seg.get("angle_deg"),
            )
            results.append(result)
            write_json(out / "result.json", result)
            print(page["document_id"], name, f"{result['matched']}/{len(references)}", flush=True)
    write_json(args.output / "per_document.json", results)
    summary = {
        v: {k: sum(r[k] for r in results if r["variant"] == v) for k in COUNTS}
        for v in GEOMETRIC_VARIANTS
    }
    write_json(args.output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("fotometria")
    p.add_argument("--lines", type=Path, required=True, help="per_line.csv do controle E4")
    p.add_argument("--model", type=Path, default=ROOT / "models/htr/TraPrInq.mlmodel")
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("geometria")
    p.add_argument("--config", type=Path, default=ROOT / "configs/avaliacao_final.json")
    p.add_argument("--manifest", type=Path, default=Path("data/iforal-prepared/pages.csv"))
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"fotometria": photometric, "geometria": geometric}[args.command](args)


if __name__ == "__main__":
    main()
