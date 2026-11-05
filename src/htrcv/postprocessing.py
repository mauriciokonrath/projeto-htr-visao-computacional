"""Pós-processamento comum a E1, E2 e E3.

1. Filtro de borda: descarta candidatos cuja baseline está quase toda fora da
   página estimada ou na faixa externa da imagem.
2. Guarda: desloca as baselines restantes para no mínimo 2 px da borda.
"""

import copy

import cv2
import numpy as np

from .baseline_pilot import raster_line

CONFIG = dict(
    blur_sigma_short_side_ratio=0.01,
    minimum_page_area_ratio=0.25,
    edge_band_short_side_ratio=0.02,
    minimum_rejected_fraction=0.90,
)


def page_support(gray, config):
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Imagem grayscale uint8 obrigatória.")
    sigma = max(1.0, min(gray.shape) * config["blur_sigma_short_side_ratio"])
    smooth = cv2.GaussianBlur(gray, (0, 0), sigma)
    threshold, binary = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count <= 1:
        return np.ones(gray.shape, np.uint8), dict(
            abstained=True, reason="no_bright_component", threshold=threshold
        )
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == index).astype(np.uint8)
    # Preenche o exterior da página; buracos internos continuam como página.
    padded = np.pad(mask, 1)
    filled = padded.copy()
    cv2.floodFill(filled, None, (0, 0), 2)
    mask = (filled[1:-1, 1:-1] != 2).astype(np.uint8)
    area = float(mask.mean())
    abstained = area < config["minimum_page_area_ratio"]
    if abstained:
        mask[:] = 1
    return mask, dict(
        abstained=abstained,
        reason="small_page_component" if abstained else None,
        threshold=threshold,
        blur_sigma=sigma,
        proposed_area_ratio=area,
    )


def filter_candidates(gray, extraction, config):
    if config != CONFIG:
        raise ValueError("Configuração do filtro de borda diferente da esperada.")
    mask, diagnostic = page_support(gray, config)
    h, w = gray.shape
    band = max(1, round(min(h, w) * config["edge_band_short_side_ratio"]))
    y, x = np.ogrid[:h, :w]
    edge = (x < band) | (x >= w - band) | (y < band) | (y >= h - band)
    kept, audit = [], []
    ids = [c["candidate_id"] for c in extraction["candidates"]]
    if len(ids) != len(set(ids)):
        raise ValueError("IDs de candidatos duplicados.")
    for candidate in extraction["candidates"]:
        pixels = raster_line(candidate["baseline"], gray.shape) > 0
        outside = float(np.mean(mask[pixels] == 0))
        border = float(np.mean(edge[pixels]))
        reasons = []
        if outside >= config["minimum_rejected_fraction"]:
            reasons.append("outside_estimated_page")
        if border >= config["minimum_rejected_fraction"]:
            reasons.append("outer_image_band")
        if not reasons:
            kept.append(candidate)
        audit.append(
            dict(
                candidate_id=candidate["candidate_id"],
                bbox=candidate["bbox"],
                excluded=bool(reasons),
                reasons=reasons,
                outside_page_fraction=outside,
                edge_band_fraction=border,
                baseline_pixels=int(pixels.sum()),
            )
        )
    result = copy.deepcopy(extraction)
    result["candidates"] = copy.deepcopy(kept)
    result["edge_filter"] = dict(
        config=config, input_candidates=len(ids), excluded=sum(r["excluded"] for r in audit)
    )
    return result, audit, mask, dict(**diagnostic, band_pixels=band)


def guard_candidates(candidates, shape):
    """Restringe as baselines ao interior da imagem com folga de 2 px.

    Valores de x repetidos após o corte são reduzidos à mediana de y.
    IDs e número de candidatos são preservados; nenhum ponto move mais de 2 px.
    """
    if min(shape) < 6:
        raise ValueError("Imagem pequena demais para guarda de dois pixels.")
    result, audit = [], []
    for candidate in candidates:
        points = np.asarray(candidate["baseline"], float)
        raster_line(points, shape)
        adjusted = np.clip(points, [2, 2], [shape[1] - 3, shape[0] - 3])
        displacement = float(np.abs(adjusted - points).max())
        if displacement > 2:
            raise ValueError("Deslocamento excede guarda.")
        xs = np.unique(adjusted[:, 0])
        adjusted = np.array([[x, float(np.median(adjusted[adjusted[:, 0] == x, 1]))] for x in xs])
        raster_line(adjusted, shape)
        if len(adjusted) < 2 or adjusted[-1, 0] - adjusted[0, 0] < 5:
            raise ValueError("A guarda reduziu o candidato a menos de dois pontos.")
        c = copy.deepcopy(candidate)
        c["baseline"] = adjusted.tolist()
        result.append(c)
        audit.append(
            dict(
                candidate_id=c["candidate_id"],
                changed=c["baseline"] != candidate["baseline"],
                max_coordinate_displacement_before_duplicate_collapse=displacement,
                original_baseline=candidate["baseline"],
                crop_baseline=c["baseline"],
                original_points=len(points),
                crop_points=len(adjusted),
            )
        )
    return result, audit
