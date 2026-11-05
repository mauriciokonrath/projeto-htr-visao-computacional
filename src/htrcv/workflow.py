import json
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np
import yaml

from .classical import ClassicalConfig, quality_features, segment
from .io import load_gray, overlay, read_csv, relative, save_image, save_json, sha256, write_csv
from .metrics import aggregate_text, segmentation_counts, text_counts
from .pagexml import page_mask, parse_pagexml
from .splits import grouped_splits


def prepare(raw, output, dataset="iforal", seed=42, document_map=None):
    raw, output = Path(raw).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(
            f"Destino já existe: {output}. Use outro diretório para preservar os splits."
        )
    xml_paths = sorted(raw.rglob("*.xml"))
    if not xml_paths:
        raise ValueError(f"Nenhum PAGE XML em {raw}.")
    mapping = {row["pagexml_path"]: row for row in read_csv(document_map)} if document_map else None
    pages, lines, parsed = [], [], []
    # Valida as entradas antes de escrever qualquer saída.
    for xml in xml_paths:
        rel = xml.relative_to(raw)
        if mapping is not None:
            record = mapping[rel.as_posix()]
            document_id, writer_id = record["document_id"], record.get("writer_id", "")
        else:
            if len(rel.parts) < 2:
                raise ValueError("Use subpastas por documento ou forneça --document-map.")
            document_id, writer_id = rel.parts[0], ""
        page = parse_pagexml(xml)
        image = (xml.parent / page.image_filename).resolve()
        if not image.is_relative_to(raw):
            raise ValueError(f"imageFilename fora de --raw: {xml}")
        gray = load_gray(image)
        if gray.shape != (page.height, page.width):
            raise ValueError(f"Dimensões imagem/XML divergentes: {xml}")
        if not page.lines:
            raise ValueError(f"Página sem linhas anotadas: {xml}; cobertura GT precisa de revisão.")
        if any(not line.polygon for line in page.lines):
            raise ValueError(f"PAGE com apenas baselines não suportado para máscaras exatas: {xml}")
        import hashlib

        page_id = "p_" + hashlib.sha256(rel.as_posix().encode()).hexdigest()[:16]
        pages.append(
            dict(
                page_id=page_id,
                source_dataset=dataset,
                document_id=document_id,
                writer_id=writer_id,
                image_path=relative(image, output),
                pagexml_path=relative(xml, output),
                split="",
                width=page.width,
                height=page.height,
                has_gt_lines=True,
                has_gt_transcription=all(line.transcription is not None for line in page.lines),
                reading_order=page.reading_order,
                image_sha256=sha256(image),
                pagexml_sha256=sha256(xml),
                mask_path=f"masks/{page_id}.png",
            )
        )
        parsed.append(page)
        for order, line in enumerate(page.lines):
            lines.append(
                dict(
                    line_id=f"{page_id}:{line.id}",
                    page_id=page_id,
                    region_id=line.region_id,
                    polygon=json.dumps(line.polygon),
                    bbox=json.dumps(line.bbox),
                    transcription=line.transcription if line.transcription is not None else "",
                    has_transcription=line.transcription is not None,
                    reading_index=order,
                    split="",
                )
            )
    assignments = grouped_splits(pages, seed=seed)
    for row in pages:
        row["split"] = assignments[row["document_id"]]
    by_id = {row["page_id"]: row for row in pages}
    for row in lines:
        row["split"] = by_id[row["page_id"]]["split"]
    output.mkdir(parents=True)
    save_json(output / "status.json", {"status": "running"})
    for row, page in zip(pages, parsed):
        gray = load_gray(output / row["image_path"])
        save_image(output / row["mask_path"], page_mask(page))
        save_image(
            output / "overlays" / f"{row['page_id']}.png",
            overlay(gray, [line.polygon for line in page.lines]),
        )
        for index, line in enumerate(page.lines):
            x1, y1, x2, y2 = line.bbox
            # Isola o polígono para não incluir linhas vizinhas no recorte.
            local_mask = np.zeros_like(gray)
            cv2.fillPoly(local_mask, [np.array(line.polygon, np.int32)], 255)
            crop = np.where(local_mask, gray, 255).astype(np.uint8)[y1:y2, x1:x2]
            save_image(output / "gt_crops" / row["page_id"] / f"{index:04d}.png", crop)
    write_csv(output / "pages.csv", pages)
    write_csv(output / "lines.csv", lines)
    write_csv(
        output / "coverage_review.csv",
        [
            dict(page_id=row["page_id"], coverage="unreviewed", roi_path="", notes="")
            for row in pages
        ],
    )
    save_json(
        output / "splits.json",
        dict(
            seed=seed,
            groups=assignments,
            policy="por document_id; escribas compartilhados ficam no mesmo split",
        ),
    )
    save_json(
        output / "status.json",
        dict(
            status="complete",
            pages=len(pages),
            lines=len(lines),
            dataset=dataset,
            seed=seed,
            note="Geometria importada; revisar ordem de leitura e qualidade das anotações.",
        ),
    )
    return dict(pages=len(pages), lines=len(lines), output=str(output))


def coverage_metrics(gt, prediction, review, base):
    coverage = review.get("coverage", "unreviewed")
    if coverage == "unreviewed":
        return {key: None for key in ("tp", "fp", "fn", "iou", "dice", "precision", "recall")}
    if coverage == "full_page":
        return segmentation_counts(gt, prediction)
    if coverage == "roi":
        if not review.get("roi_path"):
            raise ValueError("coverage=roi exige roi_path.")
        roi = load_gray(Path(base) / review["roi_path"]) > 0
        if roi.shape != gt.shape or not roi.any():
            raise ValueError("ROI deve ter dimensões da página e ao menos um pixel válido.")
        return segmentation_counts(gt[roi], prediction[roi])
    raise ValueError("coverage deve ser unreviewed, full_page ou roi.")


def run_classical(manifest, config_path, output, split="val", coverage_review=None):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) - {"seed", "classical"}:
        raise ValueError("Config deve conter apenas seed e classical.")
    config = ClassicalConfig(**payload.get("classical", {}))
    seed = int(payload.get("seed", 42))
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    rows = [row for row in read_csv(manifest) if row["split"] == split]
    if not rows:
        raise ValueError(f"Split vazio: {split}")
    reviews = {}
    if coverage_review:
        review_rows = read_csv(coverage_review)
        reviews = {r["page_id"]: r for r in review_rows}
        if len(reviews) != len(review_rows) or any(row["page_id"] not in reviews for row in rows):
            raise ValueError("Revisão de cobertura exige ids únicos e todas as páginas do split.")
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "status.json", {"status": "running"})
    (output / "config.yaml").write_text(
        yaml.safe_dump(dict(seed=seed, classical=asdict(config))), encoding="utf-8"
    )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unversioned"
    (output / "git_commit.txt").write_text(commit + "\n")
    environment = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    (output / "environment.txt").write_text("\n".join(environment) + "\n")
    save_json(
        output / "provenance.json",
        dict(
            python=platform.python_version(),
            platform=platform.platform(),
            processor=platform.processor(),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            manifest_sha256=sha256(manifest),
            split=split,
            htr_checkpoint=None,
            coverage_review_sha256=sha256(coverage_review) if coverage_review else None,
            source_hashes={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        ),
    )
    pages, lines = [], []
    for row in rows:
        start = time.perf_counter()
        image = manifest.parent / row["image_path"]
        xml = manifest.parent / row["pagexml_path"]
        if sha256(image) != row["image_sha256"] or sha256(xml) != row["pagexml_sha256"]:
            raise ValueError("Imagem/XML alterado após preparação; crie novo manifesto.")
        gray = load_gray(image)
        gt = page_mask(parse_pagexml(xml))
        prediction = segment(gray, config)
        elapsed = time.perf_counter() - start
        features = quality_features(gray, config)
        review = reviews.get(row["page_id"], {"coverage": "unreviewed"})
        counts = coverage_metrics(
            gt,
            prediction["mask"],
            review,
            Path(coverage_review).parent if coverage_review else manifest.parent,
        )
        page_dir = output / "pages" / row["page_id"]
        save_json(page_dir / "coverage.json", review)
        if review["coverage"] == "roi":
            roi_path = Path(coverage_review).parent / review["roi_path"]
            save_image(page_dir / "evaluation_roi.png", load_gray(roi_path))
        save_image(page_dir / "classical_mask.png", prediction["mask"])
        save_image(page_dir / "overlay.png", overlay(gray, prediction["polygons"], (230, 60, 30)))
        save_image(page_dir / "gray.png", gray)
        save_image(page_dir / "deskewed.png", prediction["recognition_image"])
        save_image(page_dir / "enhanced.png", prediction["enhanced"])
        save_image(page_dir / "binary.png", prediction["binary"])
        if prediction["layout"] is not None:
            save_image(page_dir / "cleaned_binary.png", prediction["cleaned_binary"])
            save_json(page_dir / "layout.json", prediction["layout"])
        save_image(page_dir / "edges.png", cv2.Canny(gray, 50, 150))
        save_json(
            page_dir / "geometry.json",
            dict(
                transform_original_to_work=prediction["transform"],
                polygons_original=prediction["polygons"],
                boxes_work=prediction["boxes"],
                applied_angle=prediction["applied_angle"],
                skew=prediction["skew"],
            ),
        )
        save_json(page_dir / "quality.json", features)
        for index, (x1, y1, x2, y2) in enumerate(prediction["boxes"]):
            crop_path = page_dir / "lines" / f"{index:04d}.png"
            save_image(crop_path, prediction["recognition_image"][y1:y2, x1:x2])
            lines.append(
                dict(
                    page_id=row["page_id"],
                    predicted_line_id=f"{index:04d}",
                    crop_path=relative(crop_path, output),
                    bbox_work=json.dumps([x1, y1, x2, y2]),
                    transcription="",
                    htr_status="not_run",
                )
            )
        pages.append(
            dict(
                page_id=row["page_id"],
                document_id=row["document_id"],
                split=split,
                source_dataset=row["source_dataset"],
                annotation_coverage=review["coverage"],
                **counts,
                **features,
                predicted_lines=len(prediction["boxes"]),
                time_seconds=elapsed,
                cer=None,
                wer=None,
            )
        )
    write_csv(output / "per_page.csv", pages)
    write_csv(
        output / "per_line.csv",
        lines,
        ["page_id", "predicted_line_id", "crop_path", "bbox_work", "transcription", "htr_status"],
    )
    evaluated = [row for row in pages if row["iou"] is not None]
    tp, fp, fn = (sum(row[key] for row in evaluated) for key in ("tp", "fp", "fn"))
    result = dict(
        experiment="E1_segmentation_only",
        split=split,
        pages=len(pages),
        evaluated_pages=len(evaluated),
        micro_iou=(tp / (tp + fp + fn) if tp + fp + fn else 1.0) if evaluated else None,
        micro_dice=(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0)
        if evaluated
        else None,
        mean_iou=float(np.mean([row["iou"] for row in evaluated])) if evaluated else None,
        mean_dice=float(np.mean([row["dice"] for row in evaluated])) if evaluated else None,
        mean_time_seconds=float(np.mean([row["time_seconds"] for row in pages])),
        cer=None,
        wer=None,
        htr_status="not_run",
        limitation="Candidatos de linha por morfologia; ordem geométrica simples; ainda sem HTR.",
    )
    save_json(output / "metrics.json", result)
    save_json(output / "status.json", {"status": "complete"})
    return result


def evaluate_transcriptions(reference, predictions, output):
    """CSV com colunas page_id,text; páginas faltantes ou extras geram erro."""

    def load(path):
        rows = read_csv(path)
        if any(not row.get("page_id") or "text" not in row for row in rows):
            raise ValueError("CSV exige page_id,text.")
        indexed = {row["page_id"]: row["text"] for row in rows}
        if len(indexed) != len(rows):
            raise ValueError("page_id duplicado.")
        return indexed

    refs, preds = load(reference), load(predictions)
    if not refs or refs.keys() != preds.keys():
        raise ValueError(
            "Referências e hipóteses devem conter exatamente as mesmas páginas, sem omissões."
        )
    rows = [dict(page_id=key, **text_counts(refs[key], preds[key])) for key in sorted(refs)]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "per_page.csv", rows)
    result = dict(
        **aggregate_text(rows),
        pages=len(rows),
        normalization="NFC, espaços colapsados; caixa e acentos preservados",
        reference_sha256=sha256(reference),
        predictions_sha256=sha256(predictions),
    )
    save_json(output / "metrics.json", result)
    return result
