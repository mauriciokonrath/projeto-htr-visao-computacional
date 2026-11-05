import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path, base):
    return os.path.relpath(Path(path).resolve(), Path(base).resolve())


def load_gray(path):
    with Image.open(path) as image:
        return np.array(image.convert("L"))


def save_image(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def overlay(gray, polygons, color=(0, 190, 60)):
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for polygon in polygons:
        cv2.polylines(
            rgb,
            [np.round(polygon).astype(np.int32)],
            True,
            color,
            max(1, round(gray.shape[1] / 700)),
        )
    return rgb


def source_hashes():
    """SHA-256 dos módulos do pacote, registrado junto aos artefatos gerados."""
    return {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
