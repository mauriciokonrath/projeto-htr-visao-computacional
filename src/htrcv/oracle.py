"""E4: reconhecimento sobre a geometria anotada (controle oráculo)."""

import json
import platform
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import yaml
from PIL import Image

from .io import read_csv, save_json, sha256, write_csv
from .kraken_backend import KrakenBackend, load_htr_config, validate_line_geometry
from .metrics import aggregate_text, normalize, text_counts
from .pagexml import parse_pagexml
from .preprocessing import preprocess_page, validate_preprocessing


def run_oracle(manifest, config_path, output, split="val", model_override=None, preprocessing=None):
    validate_preprocessing(preprocessing)
    if preprocessing is not None and split != "train":
        raise ValueError("Pré-processamento só é aceito no split de treino.")
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    config, model_path = load_htr_config(config_path, model_override)
    if config["crop_protocol"] != "kraken_gt_baseline":
        raise ValueError("E4 exige o protocolo kraken_gt_baseline.")
    if split not in {"train", "val", "test"}:
        raise ValueError("Split inválido.")
    rows = [row for row in read_csv(manifest) if row["split"] == split]
    if not rows or len({r["page_id"] for r in rows}) != len(rows):
        raise ValueError("Split vazio ou page_id duplicado.")
    prepared = []
    for row in rows:
        image_path, xml_path = (
            manifest.parent / row[key] for key in ("image_path", "pagexml_path")
        )
        if sha256(image_path) != row["image_sha256"] or sha256(xml_path) != row["pagexml_sha256"]:
            raise ValueError("Imagem/XML alterado após preparação; crie novo manifesto.")
        page = parse_pagexml(xml_path)
        with Image.open(image_path) as image:
            if image.size != (page.width, page.height):
                raise ValueError("Dimensões imagem/XML divergentes.")
        if not page.lines:
            raise ValueError("Página sem linhas; avaliação recusada.")
        for line in page.lines:
            if line.transcription is None:
                raise ValueError(
                    f"GT textual ausente em {row['page_id']}:{line.id}; nenhuma linha será omitida."
                )
            validate_line_geometry(line, page.width, page.height)
        prepared.append((row, page, image_path))
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "status.json", {"status": "running"})
    save_json(output / "preprocessing.json", preprocessing)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    save_json(
        output / "provenance.json",
        dict(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            python=platform.python_version(),
            platform=platform.platform(),
            manifest_sha256=sha256(manifest),
            split=split,
            image_preprocessing=preprocessing,
            model_sha256=config["model_sha256"],
            model_source=config["model_source"],
            normalization="NFC, espaços colapsados; caixa, acentos e pontuação preservados",
            evaluation_scope="todas as linhas anotadas do split; pareamento pela própria anotação",
            source_hashes={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
            sources=[
                dict(
                    page_id=r["page_id"],
                    image_sha256=r["image_sha256"],
                    pagexml_sha256=r["pagexml_sha256"],
                )
                for r in rows
            ],
        ),
    )
    (output / "environment.txt").write_text(
        "\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()))
        + "\n",
        encoding="utf-8",
    )
    line_rows, page_rows = [], []
    current_page, current_line = None, None
    started = time.perf_counter()
    try:
        load_start = time.perf_counter()
        backend = KrakenBackend(model_path, config)
        load_seconds = time.perf_counter() - load_start
        save_json(output / "backend.json", backend.details)
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as journal:
            for row, page, image_path in prepared:
                current_page = row["page_id"]
                page_start = time.perf_counter()
                page_lines = []
                with Image.open(image_path) as source:
                    image = preprocess_page(source, preprocessing)
                page_dir = output / "pages" / current_page
                (page_dir / "crops").mkdir(parents=True)
                if preprocessing is not None:
                    image.save(page_dir / "preprocessed.png")
                for index, line in enumerate(page.lines):
                    current_line = line.id
                    line_start = time.perf_counter()
                    recognition = backend.recognize_line(image, line)
                    recognition_seconds = time.perf_counter() - line_start
                    crop_path = page_dir / "crops" / f"{index:04d}.png"
                    recognition.crop.save(crop_path)
                    entry = dict(
                        page_id=current_page,
                        document_id=row["document_id"],
                        split=split,
                        line_id=line.id,
                        reading_index=index,
                        reference_raw=line.transcription,
                        prediction_raw=recognition.text,
                        reference=normalize(line.transcription),
                        prediction=normalize(recognition.text),
                        **text_counts(line.transcription, recognition.text),
                        mean_model_confidence=recognition.mean_confidence,
                        recognition_seconds=recognition_seconds,
                        crop_path=str(crop_path.relative_to(output)),
                        crop_sha256=sha256(crop_path),
                        status="recognized",
                    )
                    journal.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
                    journal.flush()
                    line_rows.append(entry)
                    page_lines.append(entry)
                    if (index + 1) % 10 == 0:
                        print(
                            f"{row['document_id']}: {index + 1}/{len(page.lines)} linhas",
                            flush=True,
                        )
                page_result = dict(
                    page_id=current_page,
                    document_id=row["document_id"],
                    split=split,
                    source_dataset=row["source_dataset"],
                    lines=len(page_lines),
                    **aggregate_text(page_lines),
                    reading_order=page.reading_order,
                    elapsed_seconds=time.perf_counter() - page_start,
                )
                page_rows.append(page_result)
                (page_dir / "predicted_lines.txt").write_text(
                    "\n".join(r["prediction"] for r in page_lines) + "\n", encoding="utf-8"
                )
                save_json(page_dir / "metrics.json", page_result)
                write_csv(output / "per_line.csv", line_rows)
                write_csv(output / "per_page.csv", page_rows)
                print(f"{row['document_id']}: concluído ({len(page_lines)} linhas)", flush=True)
        totals = aggregate_text(line_rows)
        result = dict(
            experiment="E4" if preprocessing is None else "E4-CLAHE",
            image_preprocessing=preprocessing,
            protocol=config["crop_protocol"],
            split=split,
            pages=len(page_rows),
            lines=len(line_rows),
            **totals,
            aggregation="soma das distâncias de edição / soma dos comprimentos de referência",
            model_sha256=config["model_sha256"],
            device=config["device"],
            model_load_seconds=load_seconds,
            elapsed_seconds=time.perf_counter() - started,
            failed_lines=0,
            missing_transcription_lines=0,
            scope="somente linhas anotadas",
            htr_status="complete",
        )
        save_json(output / "metrics.json", result)
        save_json(
            output / "status.json",
            dict(status="complete", pages=len(page_rows), lines=len(line_rows)),
        )
        return result
    except Exception as error:
        save_json(
            output / "status.json",
            dict(
                status="failed",
                page_id=current_page,
                line_id=current_line,
                completed_lines=len(line_rows),
                error_type=type(error).__name__,
                error=str(error),
            ),
        )
        # Execução parcial não gera métricas agregadas.
        raise
