"""Pareamento entre caixas anotadas e candidatos por IoU."""

import numpy as np


def box_iou_matrix(reference_boxes, candidate_boxes):
    a = np.asarray(reference_boxes, dtype=float).reshape(-1, 4)
    b = np.asarray(candidate_boxes, dtype=float).reshape(-1, 4)
    for boxes in (a, b):
        if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError("Caixas devem ter coordenadas finitas e área positiva.")
    overlap = np.maximum(
        0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2])
    )
    intersection = overlap.prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection)


def match_annotated_boxes(reference_boxes, candidate_boxes, threshold=0.5):
    """Pareamento um para um de cardinalidade máxima (caminhos aumentantes).

    Caixas no mesmo sistema de coordenadas, com limites direito/inferior exclusivos.
    O IoU é calculado entre caixas, não entre polígonos.
    """
    if not 0 < threshold <= 1:
        raise ValueError("Limiar IoU deve estar em (0,1].")
    scores = box_iou_matrix(reference_boxes, candidate_boxes)
    graph = [
        sorted(np.flatnonzero(row >= threshold).tolist(), key=lambda j: (-row[j], j))
        for row in scores
    ]
    owners = {}

    def augment(i, seen):
        for j in graph[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owners or augment(owners[j], seen):
                owners[j] = i
                return True
        return False

    for i in range(len(reference_boxes)):
        augment(i, set())
    matches = {i: j for j, i in owners.items()}
    per_reference = [
        dict(
            reference_index=i,
            candidate_index=matches.get(i),
            matched_iou=float(scores[i, matches[i]]) if i in matches else None,
            best_box_iou=float(scores[i].max()) if scores.shape[1] else 0.0,
            qualifying_candidates=len(graph[i]),
        )
        for i in range(scores.shape[0])
    ]
    return dict(
        reference_count=len(reference_boxes),
        candidate_count=len(candidate_boxes),
        matched_annotations=len(matches),
        unmatched_annotations=len(reference_boxes) - len(matches),
        unassigned_candidates=len(candidate_boxes) - len(matches),
        annotation_match_rate=len(matches) / len(reference_boxes) if len(reference_boxes) else None,
        threshold=threshold,
        per_reference=per_reference,
        caveat="cobertura das anotações por caixas; candidatos sem par não são classificados",
    )
