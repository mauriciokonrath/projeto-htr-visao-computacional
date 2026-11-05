"""Poligonização das baselines previstas e reconhecimento linha a linha."""

import json

import cv2
import numpy as np

from .baseline_pilot import raster_line
from .io import sha256, write_csv
from .kraken_backend import validate_line_geometry
from .metrics import normalize
from .pagexml import Line

PREDICTION_FIELDS = [
    "candidate_id",
    "prediction_raw",
    "prediction",
    "mean_model_confidence",
    "crop_path",
    "crop_sha256",
    "crop_width",
    "crop_height",
    "status",
]


def build_predicted_lines(image, valid, candidates, config, polygonizer=None):
    """Recebe apenas a imagem e as baselines previstas."""
    if valid.shape != (image.height, image.width):
        raise ValueError("ROI e imagem incompatíveis.")
    contours, _ = cv2.findContours(
        (valid > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) != 1 or len({c["candidate_id"] for c in candidates}) != len(candidates):
        raise ValueError("ROI única e IDs de candidatos únicos são obrigatórios.")
    if not candidates:
        return [], []
    baselines = []
    for c in candidates:
        raster_line(c["baseline"], valid.shape)  # valores finitos, dentro da imagem e monotônicos
        points = np.rint(np.asarray(c["baseline"])).astype(np.int32)
        simplified = cv2.approxPolyDP(points, config["simplify_epsilon_pixels"], False).reshape(
            -1, 2
        )
        raster_line(simplified, valid.shape)
        baselines.append(simplified.tolist())
    boundary = contours[0].reshape(-1, 2).tolist()
    boundary.append(boundary[0])
    if polygonizer is None:
        from kraken.lib.segmentation import calculate_polygonal_environment

        polygonizer = calculate_polygonal_environment
    polygons = polygonizer(
        im=image.convert("L"),
        baselines=baselines,
        suppl_obj=[boundary],
        scale=None,
        topline=False,
        raise_on_error=True,
    )
    if len(polygons) != len(candidates) or any(p is None for p in polygons):
        raise ValueError("Poligonização incompleta.")
    lines, geometry = [], []
    for c, baseline, polygon in zip(candidates, baselines, polygons):
        polygon = np.asarray(polygon)
        if (
            polygon.ndim != 2
            or polygon.shape[1] != 2
            or len(polygon) < 3
            or not np.isfinite(polygon).all()
            or not np.equal(polygon, np.rint(polygon)).all()
            or abs(cv2.contourArea(polygon.astype(np.float32))) < 1
        ):
            raise ValueError("Polígono previsto inválido.")
        line = Line(
            c["candidate_id"],
            "predicted_unet",
            tuple(tuple(map(int, p)) for p in polygon),
            tuple(tuple(map(int, p)) for p in baseline),
            None,
        )
        validate_line_geometry(line, image.width, image.height)
        polygon_mask = np.zeros_like(valid)
        cv2.fillPoly(polygon_mask, [polygon.astype(np.int32)], 255)
        outside = int(np.count_nonzero((polygon_mask > 0) & (valid == 0)))
        # Geometria inválida gera erro em vez de ser corrigida.
        if outside:
            raise ValueError("Polígono previsto ultrapassou a ROI válida.")
        lines.append(line)
        geometry.append(
            dict(
                candidate_id=line.id,
                source_baseline=c["baseline"],
                baseline=line.baseline,
                polygon=line.polygon,
                bbox=line.bbox,
                polygon_pixels_outside_roi=outside,
            )
        )
    return lines, geometry


def recognize_lines(backend, image, lines, directory):
    directory.mkdir(parents=True, exist_ok=False)
    rows = []
    with (directory / "predictions.jsonl").open("w") as journal:
        for i, line in enumerate(lines):
            result = backend.recognize_line(image, line)
            path = directory / "crops" / f"{i:04d}.png"
            path.parent.mkdir(exist_ok=True)
            result.crop.save(path)
            row = dict(
                candidate_id=line.id,
                prediction_raw=result.text,
                prediction=normalize(result.text),
                mean_model_confidence=result.mean_confidence,
                crop_path=str(path.relative_to(directory)),
                crop_sha256=sha256(path),
                crop_width=result.crop.width,
                crop_height=result.crop.height,
                status="recognized",
            )
            rows.append(row)
            journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            journal.flush()
            if (i + 1) % 10 == 0:
                print(f"{directory.name}: {i + 1}/{len(lines)} linhas reconhecidas", flush=True)
    write_csv(directory / "per_candidate.csv", rows, PREDICTION_FIELDS)
    return rows
