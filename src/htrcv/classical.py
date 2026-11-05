"""Segmentação clássica de linhas (E1): CLAHE, Hough, binarização e projeções."""

from dataclasses import dataclass

import cv2
import numpy as np

from .layout import layout_candidates
from .pagexml import Line


@dataclass
class ClassicalConfig:
    clahe: bool = True
    clip_limit: float = 2.0
    deskew: bool = True
    max_angle: float = 15.0
    min_hough_lines: int = 3
    min_confidence: float = 0.5
    threshold: str = "adaptive"
    block_size: int = 51
    threshold_c: float = 15.0
    denoise: bool = False
    morphology: bool = True
    grouping_width_ratio: float = 0.020
    grouping_height_ratio: float = 0.001
    min_area_ratio: float = 0.00003
    min_width_ratio: float = 0.015
    min_aspect: float = 2.0
    split_tall_components: bool = True
    line_height_ratio: float = 0.015
    layout_refinement: bool = False
    layout_min_component_area_ratio: float = 0.00001
    layout_max_component_height_ratio: float = 0.08
    layout_gutter_threshold: float = 0.35
    layout_min_gutter_width_ratio: float = 0.006
    layout_peak_threshold: float = 0.35
    layout_min_line_width_ratio: float = 0.12
    layout_horizontal_threshold: float = 0.30
    layout_min_ink_density: float = 0.15
    layout_vertical_margin_ratio: float = 0.0
    layout_merge_fragments: bool = False
    layout_fragment_gap_ratio: float = 0.10
    layout_projection_closing: bool = True

    def __post_init__(self):
        if (
            type(self.layout_merge_fragments) is not bool
            or not 0 < self.layout_fragment_gap_ratio <= 0.25
        ):
            raise ValueError("Configuração de união de fragmentos inválida.")
        if not 0 <= self.layout_vertical_margin_ratio <= 1:
            raise ValueError("Margem vertical deve estar em [0,1].")
        if self.threshold not in {"otsu", "adaptive"}:
            raise ValueError("threshold deve ser otsu ou adaptive.")
        if self.block_size < 3 or self.block_size % 2 == 0:
            raise ValueError("block_size deve ser ímpar e >= 3.")
        if not 0 < self.max_angle <= 45 or self.min_hough_lines < 1:
            raise ValueError("Parâmetros Hough inválidos.")
        if not 0 <= self.min_confidence <= 1 or self.clip_limit <= 0:
            raise ValueError("Confiança deve estar em [0,1] e clip_limit deve ser positivo.")
        for value in (
            self.grouping_width_ratio,
            self.grouping_height_ratio,
            self.min_area_ratio,
            self.min_width_ratio,
            self.line_height_ratio,
            self.layout_min_component_area_ratio,
            self.layout_max_component_height_ratio,
            self.layout_gutter_threshold,
            self.layout_min_gutter_width_ratio,
            self.layout_peak_threshold,
            self.layout_min_line_width_ratio,
            self.layout_horizontal_threshold,
            self.layout_min_ink_density,
        ):
            if not 0 < value <= 1:
                raise ValueError("Razões espaciais devem estar em (0,1].")


def estimate_skew(gray, config):
    edges = cv2.Canny(gray, 50, 150)
    h, w = gray.shape
    segments = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 1800,
        threshold=max(20, w // 30),
        minLineLength=max(20, w // 10),
        maxLineGap=max(5, w // 100),
    )
    angles, accepted = [], []
    if segments is not None:
        for x1, y1, x2, y2 in segments.reshape(-1, 4):
            angle = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 90) % 180 - 90
            if abs(angle) <= config.max_angle:
                angles.append(float(angle))
                accepted.append([int(x1), int(y1), int(x2), int(y2)])
    median = float(np.median(angles)) if angles else 0.0
    agreement = float(np.mean(np.abs(np.array(angles) - median) <= 2.0)) if angles else 0.0
    confidence = agreement * min(1.0, len(angles) / config.min_hough_lines)
    return dict(angle_deg=median, confidence=confidence, n_segments=len(angles), segments=accepted)


def rotation_matrix(shape, angle):
    """Rotaciona em uma tela expandida para não cortar os cantos da página."""
    h, w = shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(np.ceil(w * cosine + h * sine)), int(np.ceil(h * cosine + w * sine))
    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2
    return matrix, (new_w, new_h)


def quality_features(gray, config):
    counts = np.bincount(gray.ravel(), minlength=256)
    probabilities = counts[counts > 0] / gray.size
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    skew = estimate_skew(gray, config)
    return dict(
        contrast_std=float(gray.std()),
        p5=float(np.percentile(gray, 5)),
        p50=float(np.percentile(gray, 50)),
        p95=float(np.percentile(gray, 95)),
        entropy=float(-np.sum(probabilities * np.log2(probabilities))),
        blur_score=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        foreground_density=float(np.count_nonzero(ink) / ink.size),
        background_nonuniformity=float(cv2.GaussianBlur(gray, (0, 0), max(gray.shape) / 40).std()),
        skew_angle_deg=skew["angle_deg"],
        abs_skew_angle_deg=abs(skew["angle_deg"]),
        skew_confidence=skew["confidence"],
        hough_segments=skew["n_segments"],
    )


def legacy_candidates(binary, config):
    h, w = binary.shape
    grouped = binary.copy()
    if config.morphology:
        k = max(1, round(min(h, w) / 1800))
        grouped = cv2.morphologyEx(grouped, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        kernel = np.ones(
            (
                max(1, round(h * config.grouping_height_ratio)),
                max(1, round(w * config.grouping_width_ratio)),
            ),
            np.uint8,
        )
        grouped = cv2.morphologyEx(grouped, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(grouped)
    boxes = []
    typical_height = max(5, h * config.line_height_ratio)
    for label, (x, y, bw, bh, area) in enumerate(stats[1:], 1):
        if area < h * w * config.min_area_ratio or bw < w * config.min_width_ratio:
            continue
        if config.split_tall_components and bh > 2 * typical_height:
            # Ascendentes/descendentes podem unir uma coluna inteira; divide-se a
            # projeção horizontal nos vales, usando a máscara do componente.
            local = (labels[y : y + bh, x : x + bw] == label) & (binary[y : y + bh, x : x + bw] > 0)
            profile = local.sum(axis=1).astype(np.float32)
            smooth = cv2.GaussianBlur(
                profile[:, None], (1, 0), sigmaX=0, sigmaY=max(1, typical_height / 8)
            ).ravel()
            maxima = [
                i
                for i in range(1, bh - 1)
                if smooth[i] >= smooth[i - 1]
                and smooth[i] > smooth[i + 1]
                and smooth[i] > 0.15 * smooth.max()
            ]
            peaks = []
            for peak in sorted(maxima, key=lambda i: -smooth[i]):
                if all(abs(peak - other) >= typical_height * 0.65 for other in peaks):
                    peaks.append(peak)
            peaks.sort()
            cuts = (
                [0]
                + [
                    left + int(np.argmin(smooth[left : right + 1]))
                    for left, right in zip(peaks, peaks[1:])
                ]
                + [bh]
            )
            for top, bottom in zip(cuts, cuts[1:]):
                ys, xs = np.nonzero(local[top:bottom])
                if not len(xs):
                    continue
                x1, x2 = int(x + xs.min()), int(x + xs.max() + 1)
                y1, y2 = int(y + top + ys.min()), int(y + top + ys.max() + 1)
                if (
                    x2 - x1 >= w * config.min_width_ratio
                    and (x2 - x1) / (y2 - y1) >= config.min_aspect
                ):
                    boxes.append([x1, y1, x2, y2])
        elif bw / bh >= config.min_aspect:
            boxes.append([int(x), int(y), int(x + bw), int(y + bh)])
    boxes.sort(key=lambda box: (box[1], box[0]))
    return boxes


def segment(gray, config):
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Imagem deve ser grayscale uint8.")
    skew = estimate_skew(gray, config)
    apply_rotation = (
        config.deskew
        and skew["n_segments"] >= config.min_hough_lines
        and skew["confidence"] >= config.min_confidence
    )
    angle = skew["angle_deg"] if apply_rotation else 0.0
    matrix, size = rotation_matrix(gray.shape, angle)
    recognition = cv2.warpAffine(gray, matrix, size, borderValue=255)
    enhanced = (
        cv2.createCLAHE(config.clip_limit, (8, 8)).apply(recognition)
        if config.clahe
        else recognition.copy()
    )
    if config.denoise:
        enhanced = cv2.medianBlur(enhanced, 3)
    if config.threshold == "otsu":
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            config.block_size,
            config.threshold_c,
        )
    h, w = binary.shape
    layout = None
    cleaned = None
    if config.layout_refinement:
        boxes, cleaned, layout = layout_candidates(binary, config)
    else:
        boxes = legacy_candidates(binary, config)
    work_mask = np.zeros_like(binary)
    for x1, y1, x2, y2 in boxes:
        work_mask[y1:y2, x1:x2] = 255
    inverse = cv2.invertAffineTransform(matrix)
    original_mask = cv2.warpAffine(
        work_mask, inverse, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0
    )
    polygons = []
    for x1, y1, x2, y2 in boxes:
        corners = np.array([[[x1, y1], [x2 - 1, y1], [x2 - 1, y2 - 1], [x1, y2 - 1]]], np.float32)
        polygons.append(cv2.transform(corners, inverse).reshape(-1, 2).tolist())
    return dict(
        recognition_image=recognition,
        enhanced=enhanced,
        binary=binary,
        mask=original_mask,
        boxes=boxes,
        polygons=polygons,
        transform=matrix.tolist(),
        applied_angle=angle,
        skew=skew,
        layout=layout,
        cleaned_binary=cleaned,
    )


def rectangle_geometry(box, candidate_id):
    """Baseline horizontal sintética a 75% da altura da caixa."""
    if len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError("Retângulo previsto deve ter quatro inteiros.")
    x1, y1, x2, y2 = box
    if x2 - x1 < 6 or y2 - y1 < 2:
        raise ValueError("Retângulo pequeno demais para extração nativa.")
    y = y1 + round(0.75 * (y2 - y1 - 1))
    return Line(
        id=candidate_id,
        region_id="predicted",
        polygon=((x1, y1), (x2 - 1, y1), (x2 - 1, y2 - 1), (x1, y2 - 1)),
        baseline=((x1, y), (x2 - 1, y)),
        transcription=None,
    )
