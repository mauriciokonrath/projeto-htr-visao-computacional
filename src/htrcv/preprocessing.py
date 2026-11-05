"""Pré-processamento de imagem que preserva a geometria."""

import math

import cv2
import numpy as np
from PIL import Image


def validate_preprocessing(config):
    if config is None:
        return
    if not isinstance(config, dict) or set(config) != {"method", "clip_limit", "tile_grid_size"}:
        raise ValueError("Pré-processamento exige method, clip_limit e tile_grid_size.")
    if config["method"] != "clahe":
        raise ValueError("Tratamento suportado: clahe.")
    clip = config["clip_limit"]
    grid = config["tile_grid_size"]
    if type(clip) not in {int, float} or not math.isfinite(clip) or clip <= 0:
        raise ValueError("clip_limit deve ser finito e positivo.")
    if (
        not isinstance(grid, list)
        or len(grid) != 2
        or any(type(v) is not int or v < 1 for v in grid)
    ):
        raise ValueError("tile_grid_size exige dois inteiros positivos.")


def preprocess_page(image, config=None):
    validate_preprocessing(config)
    gray = image.convert("L")
    if config is None:
        return gray
    clahe = cv2.createCLAHE(
        clipLimit=float(config["clip_limit"]), tileGridSize=tuple(config["tile_grid_size"])
    )
    return Image.fromarray(clahe.apply(np.asarray(gray)))
