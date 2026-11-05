"""Tratamentos fotométricos aplicados a recortes de linha em tons de cinza."""

import cv2
import numpy as np

VARIANTS = ("original", "clahe", "background", "background_clahe")


def normalize_background(image):
    """Compensa variações de fundo por divisão pelo fundo estimado.

    O fundo B é estimado com fechamento morfológico usando um elemento quadrado
    de lado ímpar próximo de metade da altura do recorte (mínimo 3). A saída é
    round(255 * I / max(B, 1)), saturada em [0, 255].
    """
    kernel = max(3, int(round(image.shape[0] * 0.5)) | 1)
    background = cv2.morphologyEx(
        image,
        cv2.MORPH_CLOSE,
        np.ones((kernel, kernel), np.uint8),
        borderType=cv2.BORDER_REPLICATE,
    )
    ratio = image.astype(np.float32) * 255 / np.maximum(background.astype(np.float32), 1)
    return np.clip(np.rint(ratio), 0, 255).astype(np.uint8), kernel


def apply_treatment(image, name, clip_limit=2.0, tile_grid=(8, 8)):
    """Aplica uma das variantes de VARIANTS; geometria e tipo são preservados."""
    if name not in VARIANTS:
        raise ValueError(f"Tratamento desconhecido: {name}")
    result, kernel = image.copy(), None
    if name.startswith("background"):
        result, kernel = normalize_background(result)
    if "clahe" in name:
        result = cv2.createCLAHE(clip_limit, tile_grid).apply(result)
    if result.shape != image.shape or result.dtype != np.uint8:
        raise ValueError("Tratamento alterou a geometria ou o tipo da imagem.")
    return result, kernel
