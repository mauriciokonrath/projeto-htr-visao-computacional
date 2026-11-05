"""Alvos de baseline para a U-Net, extração de baselines e diagnóstico por tolerância.

O diagnóstico é local e não reimplementa o avaliador READ.
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from .io import save_image, save_json, sha256, source_hashes, write_csv
from .pilot_data import load_pilot, padded_tile


def geometry_config(path):
    config = json.loads(Path(path).read_text())
    integers = (
        "target_thickness",
        "close_width",
        "min_component_area",
        "min_horizontal_span",
        "sample_step",
        "tolerance_pixels",
    )
    fractions = (
        "threshold",
        "minimum_reference_coverage",
        "minimum_prediction_coverage",
        "association_reference_coverage",
    )
    if set(config) != set(integers + fractions):
        raise ValueError("Configuração geométrica incompleta/desconhecida.")
    if any(type(config[k]) is not int or config[k] < 1 for k in integers):
        raise ValueError("Geometria exige inteiros positivos.")
    if any(config[k] % 2 == 0 for k in ("target_thickness", "close_width")):
        raise ValueError("Espessura e fechamento devem ser ímpares.")
    if any(not 0 < config[k] <= 1 for k in fractions) or config["threshold"] != 0.5:
        raise ValueError("Limiares inválidos.")
    return config


def raster_line(points, shape, thickness=1):
    points = np.asarray(points, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 2
        or not np.isfinite(points).all()
        or (points < 0).any()
        or (points[:, 0] >= shape[1]).any()
        or (points[:, 1] >= shape[0]).any()
        or np.any(np.diff(points[:, 0]) <= 0)
    ):
        raise ValueError("Baseline exige >=2 pontos finitos, dentro da imagem e x crescente.")
    result = np.zeros(shape, np.uint8)
    cv2.polylines(
        result,
        [np.rint(points).astype(np.int32)],
        False,
        255,
        thickness=thickness,
        lineType=cv2.LINE_8,
    )
    return result


def prepare_baseline_pilot(source, config_path, output):
    source, output = Path(source), Path(output)
    meta, rows, arrays = load_pilot(source)
    if meta.get("target_kind", "line_envelopes") != "line_envelopes":
        raise ValueError("Fonte deve ser o piloto original de envelopes.")
    config = geometry_config(config_path)
    annotations = json.loads((source / "annotations.json").read_text())
    if len(annotations) != meta["annotated_lines"]:
        raise ValueError("Número de baselines divergente.")
    target = np.zeros_like(arrays["image"])
    for line in annotations:
        band = raster_line(line["baseline"], target.shape, config["target_thickness"])
        if np.any((band > 0) & (arrays["valid"] == 0)):
            raise ValueError("Faixa de baseline cruza a borda da ROI.")
        if np.any((band > 0) & (target > 0)):
            raise ValueError("Faixas de referência se tocam ou se sobrepõem.")
        target |= band
    components = cv2.connectedComponents(target, connectivity=8)[0] - 1
    if components != len(annotations):
        raise ValueError("Alvo não preserva um componente por baseline.")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("image.png", "valid.png", "review.json", "annotations.json"):
        shutil.copy2(source / name, output / name)
    save_image(output / "target.png", target)
    for row in rows:
        for name in ("image", "target", "valid"):
            path = output / row[name + "_path"]
            array = target if name == "target" else arrays[name]
            save_image(
                path,
                padded_tile(
                    array,
                    int(row["x"]),
                    int(row["y"]),
                    int(row["size"]),
                    255 if name == "image" else 0,
                ),
            )
            row[name + "_sha256"] = sha256(path)
    write_csv(output / "tiles.csv", rows)
    save_json(output / "geometry_config.json", config)
    files = (*meta["files"], "geometry_config.json")
    meta.update(
        target_kind="baseline_bands",
        target_description="faixas finas rasterizadas a partir das baselines do PAGE XML",
        parent_dataset_sha256=sha256(source / "dataset.json"),
        source_hashes=source_hashes(),
        geometry_config=config,
        baseline_components=int(components),
        positive_pixels=int(np.count_nonzero(target)),
        negative_valid_pixels=int(np.count_nonzero((target == 0) & (arrays["valid"] > 0))),
        files={name: sha256(output / name) for name in files},
        purpose="alvo de baselines para a U-Net piloto (ROI de treino)",
    )
    save_json(output / "dataset.json", meta)
    return meta


def extract_baselines(probability, valid, config):
    """Extrai baselines do mapa de probabilidade, sem usar anotações."""
    if (
        probability.shape != valid.shape
        or probability.ndim != 2
        or not np.isfinite(probability).all()
        or (probability < 0).any()
        or (probability > 1).any()
    ):
        raise ValueError("Probabilidade inválida.")
    binary = ((probability >= config["threshold"]) & (valid > 0)).astype(np.uint8)
    closed = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((1, config["close_width"]), np.uint8)
    )
    closed[valid == 0] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    candidates, rejected = [], []
    for component in range(1, count):
        x, y, w, h, area = map(int, stats[component])
        item = dict(component_id=component, bbox=[x, y, x + w, y + h], area=area)
        if area < config["min_component_area"] or w < config["min_horizontal_span"]:
            rejected.append(dict(**item, reason="below_fixed_area_or_horizontal_span"))
            continue
        local = labels[y : y + h, x : x + w] == component
        columns = sorted(set([*range(0, w, config["sample_step"]), w - 1]))
        points = [
            [x + column, y + float(np.median(np.flatnonzero(local[:, column])))]
            for column in columns
            if local[:, column].any()
        ]
        if len(points) < 2:
            rejected.append(dict(**item, reason="fewer_than_two_sampled_columns"))
            continue
        candidates.append(dict(**item, baseline=points))
    candidates.sort(key=lambda c: (float(np.median(np.asarray(c["baseline"])[:, 1])), c["bbox"][0]))
    for i, candidate in enumerate(candidates):
        candidate["candidate_id"] = f"b_{i:04d}"
    return dict(
        candidates=candidates, rejected_components=rejected, raw_components=count - 1, config=config
    ), labels


def _distance(mask):
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def evaluate_baselines(annotations, extraction, labels, valid, config):
    """Cobertura simétrica das linhas centrais dentro de uma tolerância fixa.

    Pares: um para um, cardinalidade máxima, ambas as coberturas >= 0.8.
    Fusões e fragmentos são contados separadamente do pareamento.
    """
    shape = valid.shape
    if labels.shape != shape:
        raise ValueError("Mapa de componentes incompatível.")
    candidates = extraction["candidates"]
    refs = [raster_line(a["baseline"], shape) > 0 for a in annotations]
    preds = [raster_line(c["baseline"], shape) > 0 for c in candidates]
    if any(np.any(r & (valid == 0)) for r in refs):
        raise ValueError("Referência fora da ROI.")
    recall = np.zeros((len(refs), len(preds)), float)
    precision = np.zeros_like(recall)
    association = np.zeros_like(recall)
    tolerance = config["tolerance_pixels"]
    for i, ref in enumerate(refs):
        distance = _distance(ref)
        for j, pred in enumerate(preds):
            precision[i, j] = float(np.mean(distance[pred] <= tolerance))
    for j, (pred, candidate) in enumerate(zip(preds, candidates)):
        distance = _distance(pred)
        support_distance = _distance(labels == candidate["component_id"])
        for i, ref in enumerate(refs):
            recall[i, j] = float(np.mean(distance[ref] <= tolerance))
            association[i, j] = float(np.mean(support_distance[ref] <= tolerance))
    eligible = (recall >= config["minimum_reference_coverage"]) & (
        precision >= config["minimum_prediction_coverage"]
    )
    graph = [
        sorted(np.flatnonzero(row).tolist(), key=lambda j: (-recall[i, j] - precision[i, j], j))
        for i, row in enumerate(eligible)
    ]
    owners = {}

    def augment(i, seen):
        for j in graph[i]:
            if j not in seen:
                seen.add(j)
                if j not in owners or augment(owners[j], seen):
                    owners[j] = i
                    return True
        return False

    for i in range(len(refs)):
        augment(i, set())
    matched = {i: j for j, i in owners.items()}
    associated = association >= config["association_reference_coverage"]
    per_reference = [
        dict(
            line_id=a["line_id"],
            reference_index=i,
            candidate_id=candidates[matched[i]]["candidate_id"] if i in matched else None,
            matched=i in matched,
            reference_coverage=float(recall[i, matched[i]]) if i in matched else None,
            prediction_coverage=float(precision[i, matched[i]]) if i in matched else None,
            best_reference_coverage=float(recall[i].max()) if candidates else 0.0,
            associated_candidates=[
                candidates[j]["candidate_id"] for j in np.flatnonzero(associated[i])
            ],
        )
        for i, a in enumerate(annotations)
    ]
    per_candidate = [
        dict(
            candidate_id=c["candidate_id"],
            matched_line_id=annotations[owners[j]]["line_id"] if j in owners else None,
            associated_line_ids=[
                annotations[i]["line_id"] for i in np.flatnonzero(associated[:, j])
            ],
            merge_flag=bool(associated[:, j].sum() > 1),
        )
        for j, c in enumerate(candidates)
    ]
    return dict(
        protocol="local_symmetric_centerline_coverage_v1",
        split="train",
        reference_lines=len(refs),
        candidates=len(preds),
        matched_lines=len(matched),
        omitted_lines=len(refs) - len(matched),
        unassigned_candidates=len(preds) - len(matched),
        merged_candidates=int(np.sum(associated.sum(axis=0) > 1)),
        fragmented_references=int(np.sum(associated.sum(axis=1) > 1)),
        reference_match_rate=len(matched) / len(refs) if refs else None,
        per_reference=per_reference,
        per_candidate=per_candidate,
        config=config,
        reference_coverage_matrix=recall.tolist(),
        prediction_coverage_matrix=precision.tolist(),
        component_association_matrix=association.tolist(),
        note="Diagnóstico geométrico na ROI de treino. Omissão = referência sem par um a um; "
        "fusões e fragmentos são associações geométricas.",
    )
