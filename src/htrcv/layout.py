"""Refinamento de layout: colunas, projeções horizontais e fragmentos."""

import cv2
import numpy as np


def intervals(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def clean_components(binary, config):
    h, w = binary.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    keep = np.zeros(count, np.uint8)
    rejected = []
    for label, (x, y, bw, bh, area) in enumerate(stats[1:], 1):
        reason = None
        if area < max(6, h * w * config.layout_min_component_area_ratio):
            reason = "small_component"
        elif bh > h * config.layout_max_component_height_ratio or bw > w * 0.7:
            reason = "oversized_structure"
        elif bh < w * 0.005 and bw > w * 0.3:
            reason = "thin_horizontal_rule"
        elif bw < w * 0.005 and bh > h * 0.04:
            reason = "thin_vertical_rule"
        if reason:
            rejected.append(
                dict(bbox=[int(x), int(y), int(x + bw), int(y + bh)], area=int(area), reason=reason)
            )
        else:
            keep[label] = 255
    return keep[labels], rejected


def detect_gutter(cleaned, config):
    """Detecta no máximo um separador interno (hipótese de duas colunas)."""
    h, w = cleaned.shape
    profile = (cleaned[int(h * 0.15) : int(h * 0.85)] > 0).sum(axis=0).astype(np.float64)
    if not profile.any():
        return [], []
    smooth = cv2.GaussianBlur(profile[None, :], (0, 1), sigmaX=max(1, w * 0.003)).ravel()
    level = float(np.percentile(smooth, 80))
    candidates = []
    for left, right in intervals(smooth < config.layout_gutter_threshold * level):
        middle = int((left + right) // 2)
        if (
            right - left < max(3, w * config.layout_min_gutter_width_ratio)
            or not 0.2 * w < middle < 0.8 * w
            or min(profile[:middle].sum(), profile[middle:].sum()) < 0.2 * profile.sum()
        ):
            continue
        candidates.append(
            dict(
                left=int(left),
                right=int(right),
                center=middle,
                relative_density=float(smooth[left:right].mean() / level),
            )
        )
    if len(candidates) != 1:
        # Vários vales alinhados indicam layout ambíguo.
        return [], candidates
    return [candidates[0]["center"]], candidates


def projection_lines(cleaned, left, right, config):
    h, _ = cleaned.shape
    local = cleaned[:, left:right] > 0
    profile = local.sum(axis=1).astype(np.float64)
    typical_height = max(5, h * config.line_height_ratio)
    smooth = cv2.GaussianBlur(
        profile[:, None], (1, 0), sigmaX=0, sigmaY=max(1, typical_height / 8)
    ).ravel()
    if not smooth.any():
        return []
    level = config.layout_peak_threshold * np.percentile(smooth, 95)
    maxima = [
        i
        for i in range(1, h - 1)
        if smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1] and smooth[i] > level
    ]
    peaks = []
    for peak in sorted(maxima, key=lambda i: (-smooth[i], i)):
        if all(abs(peak - other) >= typical_height * 0.65 for other in peaks):
            peaks.append(peak)
    peaks.sort()
    boxes = []
    for i, peak in enumerate(peaks):
        lo = max(0, int(peak - typical_height))
        hi = min(h, int(peak + typical_height) + 1)
        if i:
            lo = max(lo, peaks[i - 1] + int(np.argmin(smooth[peaks[i - 1] : peak + 1])))
        if i + 1 < len(peaks):
            hi = min(hi, peak + int(np.argmin(smooth[peak : peaks[i + 1] + 1])) + 1)
        if hi <= lo:
            continue
        # Descarta caudas fracas da projeção.
        active = np.flatnonzero(smooth[lo:hi] >= smooth[peak] * 0.25)
        if not len(active):
            continue
        top, bottom = lo + int(active[0]), lo + int(active[-1]) + 1
        band = local[top:bottom]
        xp = band.sum(axis=0).astype(float)
        xp = cv2.GaussianBlur(xp[None, :], (0, 1), sigmaX=max(1, (right - left) * 0.003)).ravel()
        active_x = xp >= max(1.0, xp.max() * config.layout_horizontal_threshold)
        # Fechamento 1D: une espaços curtos entre palavras dentro da mesma coluna.
        if config.layout_projection_closing:
            gap = max(1, round((right - left) * 0.04))
            active_x = (
                cv2.morphologyEx(
                    active_x.astype(np.uint8)[None, :],
                    cv2.MORPH_CLOSE,
                    np.ones((1, gap), np.uint8),
                ).ravel()
                > 0
            )
        for x1, x2 in intervals(active_x):
            if x2 - x1 < (right - left) * config.layout_min_line_width_ratio:
                continue
            if (x2 - x1) / max(1, bottom - top) < config.min_aspect:
                continue
            if band[:, x1:x2].mean() < config.layout_min_ink_density:
                continue
            boxes.append(
                [
                    left + max(0, int(x1) - 2),
                    max(0, top - 2),
                    min(right, left + int(x2) + 2),
                    min(h, bottom + 2),
                ]
            )
    return boxes


def add_vertical_margins(boxes, image_height, ratio):
    """Expande as caixas verticalmente até o ponto médio entre linhas vizinhas.

    IDs e extensão em x são mantidos. Só contam como vizinhas caixas com
    sobreposição horizontal de pelo menos metade da menor, separando colunas.
    """
    if not 0 <= ratio <= 1:
        raise ValueError("Margem vertical deve estar em [0,1].")
    expanded, diagnostics = [], []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        center = (y1 + y2 - 1) / 2
        above, below = [], []
        for j, (a, b, c, d) in enumerate(boxes):
            overlap = max(0, min(x2, c) - max(x1, a))
            if i == j or overlap < 0.5 * min(x2 - x1, c - a):
                continue
            other_center = (b + d - 1) / 2
            if other_center < center:
                above.append((other_center + center) / 2)
            elif other_center > center:
                below.append((other_center + center) / 2)
        padding = int(np.ceil((y2 - y1) * ratio))
        top, bottom = max(0, y1 - padding), min(image_height, y2 + padding)
        if above:
            top = min(y1, max(top, int(np.ceil(max(above)))))
        if below:
            bottom = max(y2, min(bottom, int(np.ceil(min(below)))))
        box = [x1, top, x2, bottom]
        expanded.append(box)
        diagnostics.append(
            dict(
                candidate_index=i,
                original_box=[x1, y1, x2, y2],
                expanded_box=box,
                added_top=y1 - top,
                added_bottom=bottom - y2,
            )
        )
    return expanded, diagnostics


def fragment_groups(boxes, ink, columns, max_gap_ratio=0.10):
    """Agrupa fragmentos adjacentes da mesma faixa de projeção e coluna.

    Exige centróides verticais próximos e lacunas limitadas. Retorna uma
    partição dos índices de origem.
    """
    if not 0 < max_gap_ratio <= 0.25:
        raise ValueError("Gap de fragmentos deve estar em (0,0.25].")
    bands = {}
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        column = next(
            (k for k, (left, right) in enumerate(columns) if left <= x1 < x2 <= right), None
        )
        if column is None:
            raise ValueError("Candidato fora das colunas previstas.")
        rows = np.nonzero(ink[y1:y2, x1:x2])[0]
        center = float(rows.mean() + y1) if len(rows) else None
        bands.setdefault((column, y1, y2), []).append((i, center))
    groups = []
    for (column, y1, y2), entries in bands.items():
        entries.sort(key=lambda item: (boxes[item[0]][0], item[0]))
        current, centers = [], []
        limit = min(3 * (y2 - y1), max_gap_ratio * (columns[column][1] - columns[column][0]))
        for i, center in entries:
            gap = boxes[i][0] - boxes[current[-1]][2] if current else 0
            eligible = (
                current
                and center is not None
                and len(centers) == len(current)
                and -4 <= gap <= limit
                and max([*centers, center]) - min([*centers, center]) <= 0.25 * (y2 - y1)
            )
            if current and not eligible:
                groups.append(current)
                current, centers = [], []
            current.append(i)
            if center is not None:
                centers.append(center)
        if current:
            groups.append(current)
    groups.sort(key=min)
    return groups


def layout_candidates(binary, config):
    cleaned, rejected = clean_components(binary, config)
    cuts, gutter_candidates = detect_gutter(cleaned, config)
    boundaries = [0, *cuts, binary.shape[1]]
    boxes = []
    for left, right in zip(boundaries, boundaries[1:]):
        boxes.extend(projection_lines(cleaned, left, right, config))
    groups = (
        fragment_groups(
            boxes, cleaned, list(zip(boundaries, boundaries[1:])), config.layout_fragment_gap_ratio
        )
        if config.layout_merge_fragments
        else [[i] for i in range(len(boxes))]
    )
    margin_diagnostics = []
    if config.layout_vertical_margin_ratio:
        boxes, margin_diagnostics = add_vertical_margins(
            boxes, binary.shape[0], config.layout_vertical_margin_ratio
        )
    premerge_boxes = boxes
    if config.layout_merge_fragments:
        boxes = [
            [
                min(premerge_boxes[i][0] for i in group),
                min(premerge_boxes[i][1] for i in group),
                max(premerge_boxes[i][2] for i in group),
                max(premerge_boxes[i][3] for i in group),
            ]
            for group in groups
        ]
    # Ordem por coluna e depois por posição vertical.
    return (
        boxes,
        cleaned,
        dict(
            method="component_cleanup_gutter_projection",
            gutter_cuts_work=cuts,
            gutter_candidates=gutter_candidates,
            columns_work=[[a, b] for a, b in zip(boundaries, boundaries[1:])],
            rejected_components=rejected,
            input_ink_pixels=int(np.count_nonzero(binary)),
            retained_ink_pixels=int(np.count_nonzero(cleaned)),
            vertical_margin_diagnostics=margin_diagnostics,
            fragment_groups=groups,
            premerge_boxes=premerge_boxes if config.layout_merge_fragments else [],
            limitations="no máximo duas colunas; títulos, notas e iniciais decoradas podem ser fragmentados",
        ),
    )
