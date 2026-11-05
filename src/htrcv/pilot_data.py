"""Prepara a ROI de treino da U-Net com rótulos e máscara de pixels válidos."""

import json
from pathlib import Path

import cv2
import numpy as np

from .io import load_gray, read_csv, save_image, save_json, sha256, source_hashes, write_csv
from .metrics import normalize
from .pagexml import page_mask, parse_pagexml


def tile_origins(length, size, stride):
    if type(size) is not int or type(stride) is not int or not 0 < stride <= size:
        raise ValueError("Tile/stride inválidos ou deixam lacunas.")
    if length <= 0:
        raise ValueError("Dimensão inválida.")
    if length <= size:
        return [0]
    return sorted(set([*range(0, length - size + 1, stride), length - size]))


def padded_tile(array, x, y, size, fill=0):
    result = np.full((size, size), fill, dtype=array.dtype)
    view = array[y : y + size, x : x + size]
    result[: view.shape[0], : view.shape[1]] = view
    return result


def reconstruct_tiles(probabilities, rows, shape, valid):
    """Média das probabilidades sobrepostas; cada pixel é avaliado uma vez."""
    if len(probabilities) != len(rows):
        raise ValueError("Número de tiles divergente.")
    total, count = np.zeros(shape, np.float64), np.zeros(shape, np.uint16)
    for prediction, row in zip(probabilities, rows):
        x, y = int(row["x"]), int(row["y"])
        h, w = min(prediction.shape[0], shape[0] - y), min(prediction.shape[1], shape[1] - x)
        if h <= 0 or w <= 0 or not np.isfinite(prediction).all():
            raise ValueError("Tile fora da imagem ou não finito.")
        total[y : y + h, x : x + w] += prediction[:h, :w]
        count[y : y + h, x : x + w] += 1
    if np.any((valid > 0) & (count == 0)):
        raise ValueError("Pixels válidos sem cobertura dos tiles.")
    result = np.zeros(shape, np.float32)
    np.divide(total, count, out=result, where=count > 0)
    return result, count


def prepare_pilot(manifest, review_path, output, tile_size=256, stride=192):
    manifest, review_path, output = map(Path, (manifest, review_path, output))
    review = json.loads(review_path.read_text())
    if review["split"] != "train" or review["review_status"] != "pilot_geometry_reviewed":
        raise ValueError("Piloto exige ROI revisada apenas no treino.")
    if review["manifest_sha256"] != sha256(manifest):
        raise ValueError("Manifesto alterado após revisão.")
    rows = [r for r in read_csv(manifest) if r["page_id"] == review["page_id"]]
    if len(rows) != 1 or rows[0]["split"] != "train":
        raise ValueError("Página ausente/duplicada ou fora do treino.")
    row = rows[0]
    image, xml = manifest.parent / row["image_path"], manifest.parent / row["pagexml_path"]
    for key, path in (("image_sha256", image), ("pagexml_sha256", xml)):
        if sha256(path) != row[key] or row[key] != review[key]:
            raise ValueError("Fonte alterada após revisão.")
    page, gray = parse_pagexml(xml), load_gray(image)
    if gray.shape != (page.height, page.width):
        raise ValueError("Dimensões divergentes.")
    expected = review["expected_line_ids"]
    if len(set(expected)) != len(expected) or set(expected) != {line.id for line in page.lines}:
        raise ValueError("Universo de linhas divergente da revisão.")
    if any(line.transcription is None or not normalize(line.transcription) for line in page.lines):
        raise ValueError("Transcrição ausente/vazia.")
    polygon = np.asarray(review["roi_polygon"], np.float64)
    if (
        polygon.ndim != 2
        or polygon.shape[1] != 2
        or len(polygon) < 3
        or not np.isfinite(polygon).all()
        or not np.equal(polygon, np.round(polygon)).all()
        or (polygon < 0).any()
        or (polygon[:, 0] >= page.width).any()
        or (polygon[:, 1] >= page.height).any()
    ):
        raise ValueError("ROI inválida ou fora da imagem.")
    polygon = polygon.astype(np.int32)
    valid = np.zeros_like(gray)
    cv2.fillPoly(valid, [polygon], 255)
    target = page_mask(page)
    if np.any((target > 0) & (valid == 0)):
        raise ValueError("ROI corta polígonos GT; revisar geometria antes de preparar.")
    if not np.any((target == 0) & (valid > 0)) or not np.any(target > 0):
        raise ValueError("ROI precisa incluir texto e fundo, não apenas a máscara GT.")
    xs, ys = (
        tile_origins(page.width, tile_size, stride),
        tile_origins(page.height, tile_size, stride),
    )
    output.mkdir(parents=True, exist_ok=False)
    save_image(output / "image.png", gray)
    save_image(output / "target.png", target)
    save_image(output / "valid.png", valid)
    save_json(output / "review.json", review)
    annotations = [
        dict(
            line_id=line.id,
            reference_raw=line.transcription,
            reference=normalize(line.transcription),
            bbox=line.bbox,
            polygon=line.polygon,
            baseline=line.baseline,
            review="alinhamento inspecionado; transcrição original mantida",
        )
        for line in page.lines
    ]
    save_json(output / "annotations.json", annotations)
    tile_rows = []
    for y in ys:
        for x in xs:
            vv = padded_tile(valid, x, y, tile_size)
            if not np.any(vv):
                continue
            tid = f"t_{len(tile_rows):04d}"
            item = dict(
                tile_id=tid,
                page_id=row["page_id"],
                document_id=row["document_id"],
                split="train",
                x=x,
                y=y,
                size=tile_size,
                valid_pixels=int(np.count_nonzero(vv)),
            )
            for name, array, fill in (
                ("image", gray, 255),
                ("target", target, 0),
                ("valid", valid, 0),
            ):
                path = output / "tiles" / tid / f"{name}.png"
                save_image(path, padded_tile(array, x, y, tile_size, fill))
                item[name + "_path"] = str(path.relative_to(output))
                item[name + "_sha256"] = sha256(path)
            tile_rows.append(item)
    write_csv(output / "tiles.csv", tile_rows)
    full_files = [
        "image.png",
        "target.png",
        "valid.png",
        "review.json",
        "annotations.json",
        "tiles.csv",
    ]
    metadata = dict(
        page_id=row["page_id"],
        document_id=row["document_id"],
        split="train",
        manifest_sha256=sha256(manifest),
        review_sha256=sha256(review_path),
        image_sha256=row["image_sha256"],
        pagexml_sha256=row["pagexml_sha256"],
        width=page.width,
        height=page.height,
        tile_size=tile_size,
        stride=stride,
        tiles=len(tile_rows),
        annotated_lines=len(page.lines),
        reference_chars=sum(len(normalize(line.transcription)) for line in page.lines),
        reference_words=sum(len(normalize(line.transcription).split()) for line in page.lines),
        valid_pixels=int(np.count_nonzero(valid)),
        positive_pixels=int(np.count_nonzero(target)),
        negative_valid_pixels=int(np.count_nonzero((target == 0) & (valid > 0))),
        full_page_approved=False,
        benchmark_roi_approved=False,
        pilot_roi_reviewed_by="inspeção visual; sem certificação especializada",
        source_hashes=source_hashes(),
        files={name: sha256(output / name) for name in full_files},
        purpose="ROI de treino da U-Net piloto; rótulos são envelopes de linha",
    )
    save_json(output / "dataset.json", metadata)
    return metadata


def load_pilot(directory):
    directory = Path(directory)
    meta = json.loads((directory / "dataset.json").read_text())
    if meta["split"] != "train" or meta["full_page_approved"] or meta["benchmark_roi_approved"]:
        raise ValueError("O carregador aceita apenas o piloto de treino.")
    for filename, digest in meta["files"].items():
        if sha256(directory / filename) != digest:
            raise ValueError("Arquivo do piloto alterado: " + filename)
    rows = read_csv(directory / "tiles.csv")
    if len(rows) != meta["tiles"] or len({r["tile_id"] for r in rows}) != len(rows):
        raise ValueError("Universo de tiles divergente.")
    arrays = {name: load_gray(directory / f"{name}.png") for name in ("image", "target", "valid")}
    if any(a.shape != (meta["height"], meta["width"]) for a in arrays.values()):
        raise ValueError("Dimensões do piloto divergentes.")
    for row in rows:
        if row["split"] != "train" or row["page_id"] != meta["page_id"]:
            raise ValueError("Mistura de páginas/splits no piloto.")
        for name in arrays:
            path = directory / row[name + "_path"]
            if sha256(path) != row[name + "_sha256"]:
                raise ValueError("Tile alterado: " + str(path))
            expected = padded_tile(
                arrays[name],
                int(row["x"]),
                int(row["y"]),
                int(row["size"]),
                255 if name == "image" else 0,
            )
            if not np.array_equal(load_gray(path), expected):
                raise ValueError("Tile não reproduz os pixels originais.")
    return meta, rows, arrays
