"""Avaliação comparativa das pipelines E0–E4 com o reconhecedor HTR fixo.

Para cada página, todas as pipelines geram e reconhecem suas linhas antes da
leitura das anotações. Depois, cada referência é pareada a no máximo um
candidato (IoU de caixas >= limiar) e o CER/WER é calculado sobre todas as
referências; referências sem par contam como deleção completa.
"""

import csv
import json
import time
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from .baseline_pilot import extract_baselines
from .classical import ClassicalConfig, rectangle_geometry, segment
from .columns import image_barriers
from .io import sha256
from .line_matching import match_annotated_boxes
from .metrics import aggregate_text, text_counts
from .pagexml import parse_pagexml
from .postprocessing import filter_candidates, guard_candidates
from .recognition import build_predicted_lines, recognize_lines


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def log_event(run, event, **fields):
    with (run / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(time=now(), event=event, **fields)) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def line_geometry(line):
    return dict(candidate_id=line.id, polygon=line.polygon, baseline=line.baseline, bbox=line.bbox)


def attempt(directory, operation):
    """Executa uma condição página/método e registra estado, avisos e falhas."""
    directory.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    captured = []
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = operation()
        state = dict(status="complete", seconds=time.perf_counter() - start, **result)
    except Exception as error:
        state = dict(
            status="failed",
            seconds=time.perf_counter() - start,
            error_type=type(error).__name__,
            error=str(error),
        )
        write_json(directory / "failure.json", dict(traceback=traceback.format_exc()))
    write_json(
        directory / "warnings.json",
        [dict(category=w.category.__name__, message=str(w.message)) for w in captured],
    )
    write_json(directory / "state.json", state)
    return state


def classical_candidates(gray, config):
    """E1: caixas clássicas convertidas em baselines no sistema original."""
    seg = segment(gray, ClassicalConfig(**config))
    inverse = cv2.invertAffineTransform(np.array(seg["transform"]))
    candidates = []
    for i, box in enumerate(seg["boxes"]):
        line = rectangle_geometry(box, f"c_{i:04d}")
        raw = cv2.transform(np.array([line.baseline], float), inverse)[0]
        points = np.clip(raw, [0, 0], [gray.shape[1] - 1, gray.shape[0] - 1])
        require(np.all(np.diff(points[:, 0]) > 0), "Baseline não monotônica")
        candidates.append(
            dict(
                candidate_id=line.id,
                baseline=points.tolist(),
                bbox=box,
                bbox_frame="deskewed",
                initial_clip_max_pixels=float(np.abs(raw - points).max()),
            )
        )
    geometry = dict(transform=seg["transform"], angle=seg["applied_angle"], boxes=seg["boxes"])
    return dict(candidates=candidates, method="classical_original_frame_surrogate"), geometry


def predict_condition(method, gray, rgb, probability, models, plan, out):
    """Gera geometria e reconhece as linhas de um método em uma página."""
    image = Image.fromarray(gray)
    valid = np.ones_like(gray, np.uint8)
    if method == "E0_native":
        lines, raw = models.native.segment(rgb)
        geometry = [line_geometry(line) for line in lines]
        write_json(out / "native_geometry.json", raw)
        raw_count, excluded = len(lines), 0
    else:
        if method == "E1":
            extraction, classical = classical_candidates(gray, plan["classical_config"])
            write_json(out / "classical_geometry.json", classical)
        else:
            require(probability is not None, "Mapa de probabilidade da U-Net indisponível")
            allowed = valid
            if method == "E3":
                allowed, diagnostics = image_barriers(gray, plan["column_config"])
                write_json(out / "columns.json", diagnostics)
                Image.fromarray((1 - allowed) * 255).save(out / "barriers.png")
            extraction, _ = extract_baselines(probability, allowed, plan["baseline_config"])
        write_json(out / "raw_extraction.json", extraction)
        post = plan["common_postprocessing"]
        filtered, edge_audit, _, support = filter_candidates(gray, extraction, post["edge_filter"])
        write_json(out / "edge_audit.json", edge_audit)
        write_json(out / "support_diagnostic.json", support)
        guarded, guard_audit = guard_candidates(filtered["candidates"], gray.shape)
        write_json(out / "guard_audit.json", guard_audit)
        write_json(out / "crop_candidates.json", guarded)
        # Nenhum candidato é um resultado válido: todas as referências viram omissões.
        if guarded:
            lines, geometry = build_predicted_lines(
                image, valid, guarded, post["polygons"], models.polygonizer
            )
        else:
            lines, geometry = [], []
        require(all(Polygon(g["polygon"]).is_valid for g in geometry), "Polígono inválido")
        raw_count = len(extraction["candidates"])
        excluded = raw_count - len(guarded)
    write_json(out / "geometry.json", geometry)
    predictions = recognize_lines(models.backend, image, lines, out / "recognition")
    require(
        [p["candidate_id"] for p in predictions] == [g["candidate_id"] for g in geometry],
        "Ordem dos candidatos divergente",
    )
    return dict(recognized=len(predictions), raw_candidates=raw_count, excluded=excluded)


def references(page):
    require(bool(page.lines), "Página sem linhas de referência")
    for line in page.lines:
        require(line.transcription is not None, "Missing reference transcription")
        polygon = Polygon(line.polygon) if len(line.polygon) >= 3 else None
        require(
            polygon is not None and polygon.is_valid and polygon.area > 0,
            "Polígono de referência inválido",
        )
    return [
        dict(
            line_id=line.id,
            reference_raw=line.transcription,
            polygon=line.polygon,
            baseline=line.baseline,
        )
        for line in page.lines
    ]


def score(refs, predictions, geometry, method, threshold):
    """Pareia referências e candidatos apenas pela geometria e conta as edições."""
    require(len(predictions) == len(geometry), "Número de predições e geometrias diverge")
    require(
        len({p["candidate_id"] for p in predictions}) == len(predictions),
        "Candidato duplicado",
    )
    if method == "E4_control":
        require(
            [p["candidate_id"] for p in predictions] == [r["line_id"] for r in refs],
            "E4 identity mismatch",
        )
        indices = list(range(len(refs)))
        matching = {"policy": "identity"}
    else:
        boxes = []
        for ref in refs:
            xy = np.array(ref["polygon"])
            boxes.append([*xy.min(0), *(xy.max(0) + 1)])
        matching = match_annotated_boxes(boxes, [g["bbox"] for g in geometry], threshold)
        indices = [r["candidate_index"] for r in matching["per_reference"]]
    rows, used = [], set()
    for ref, index in zip(refs, indices):
        if index is not None:
            used.add(index)
        hypothesis = predictions[index]["prediction_raw"] if index is not None else ""
        rows.append(
            dict(
                line_id=ref["line_id"],
                candidate_id=predictions[index]["candidate_id"] if index is not None else None,
                reference_raw=ref["reference_raw"],
                prediction_raw=hypothesis,
                status="matched" if index is not None else "omitted_annotation",
                **text_counts(ref["reference_raw"], hypothesis),
            )
        )
    unassigned = [p for i, p in enumerate(predictions) if i not in used]
    return rows, matching, unassigned


def summarize(results, rows_by_method, methods, page_count):
    """Agrega por método; só há métrica global quando todas as páginas foram pontuadas."""
    summary = {}
    for method in methods:
        entries = [r for r in results if r["method"] == method]
        complete = len(entries) == page_count and all(e["metrics"] is not None for e in entries)
        planned = (
            sum(e["reference_lines"] for e in entries)
            if all(e["reference_lines"] is not None for e in entries)
            else None
        )
        summary[method] = dict(
            complete=complete,
            planned_pages=page_count,
            completed_pages=sum(e["metrics"] is not None for e in entries),
            reference_lines_planned=planned,
            reference_lines_scored=len(rows_by_method[method]),
            metrics=aggregate_text(rows_by_method[method]) if complete else None,
            matched=sum(e["matched"] for e in entries) if complete else None,
            omitted=sum(e["omitted"] for e in entries) if complete else None,
            unassigned_candidates=sum(e["unassigned"] for e in entries) if complete else None,
        )
    return summary


@dataclass
class Models:
    """Modelos usados na avaliação.

    backend: reconhecedor com `recognize_line(image, line)` e `details`.
    native: segmentador E0 com `segment(rgb) -> (lines, raw)` e `details`.
    probability: função `gray -> (probabilidade, cobertura)` da U-Net.
    polygonizer: poligonizador das baselines; None usa o do Kraken.
    """

    backend: object
    native: object
    probability: Callable
    polygonizer: Callable | None = None
    recognizer_sha256: str | None = None


def load_models(root, plan):
    import torch

    from .kraken_backend import KrakenBackend
    from .native import NativeSegmenter
    from .unet_pilot import SmallUNet, predict_page

    seed, threads = plan["htr_config"]["seed"], plan["htr_config"]["threads"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(threads)

    paths = plan["models"]
    backend = KrakenBackend(root / paths["recognizer"], plan["htr_config"])
    blla = paths.get("native_blla")
    if blla is None:
        import kraken

        blla = Path(kraken.__file__).parent / "blla.mlmodel"
    native = NativeSegmenter(root / blla, plan["native_segmentation_config"])
    payload = torch.load(root / paths["pilot_unet"], map_location="cpu", weights_only=True)
    unet = SmallUNet(payload["base_channels"])
    unet.load_state_dict(payload["model_state"])
    unet.eval()

    def probability(gray):
        result, _, _, coverage = predict_page(unet, gray, plan["neural_inference"])
        return result, coverage

    return Models(
        backend=backend,
        native=native,
        probability=probability,
        recognizer_sha256=sha256(root / paths["recognizer"]),
    )


def select_pages(manifest, split=None, page_ids=None):
    with Path(manifest).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if page_ids:
        rows = [r for r in rows if r["page_id"] in set(page_ids)]
    elif split:
        rows = [r for r in rows if r["split"] == split]
    return sorted(rows, key=lambda r: (r["document_id"], r["page_id"]))


def predict_all(run, manifest, selected, plan, models):
    """Fase 1: todas as pipelines previstas, sem acesso às anotações."""
    pending = {}
    for row in selected:
        page_id = row["page_id"]
        folder = run / page_id
        folder.mkdir()
        log_event(run, "image_decode_start", page_id=page_id)
        with Image.open(manifest.parent / row["image_path"]) as original:
            rgb = original.convert("RGB")
            gray = np.array(original.convert("L"))
        require(gray.shape == (int(row["height"]), int(row["width"])), "Dimensões divergentes")
        Image.fromarray(gray).save(folder / "image.png")
        probability = None
        try:
            probability, coverage = models.probability(gray)
            require(
                probability.shape == gray.shape
                and np.isfinite(probability).all()
                and np.all(coverage > 0),
                "Cobertura de probabilidade inválida",
            )
            np.savez_compressed(folder / "probability.npz", probability=probability)
        except Exception as error:
            probability = None
            write_json(
                folder / "neural_failure.json",
                dict(error=str(error), traceback=traceback.format_exc()),
            )
        for method in plan["predicted_method_order"]:
            out = folder / method
            pending[page_id, method] = attempt(
                out,
                lambda: predict_condition(method, gray, rgb, probability, models, plan, out),
            )
            log_event(
                run,
                "prediction_attempt_finished",
                page_id=page_id,
                method=method,
                status=pending[page_id, method]["status"],
            )
            print(page_id, method, pending[page_id, method]["status"], flush=True)
    return pending


def load_references(folder, row, manifest):
    page = parse_pagexml(manifest.parent / row["pagexml_path"])
    require(
        (page.width, page.height) == (int(row["width"]), int(row["height"])),
        "Dimensões da referência divergentes",
    )
    refs = references(page)
    write_json(folder / "references.json", refs)
    return page, refs


def run_control(folder, out, page, backend):
    """E4: reconhecimento sobre a geometria anotada."""
    image = Image.open(folder / "image.png").convert("L")
    write_json(out / "geometry.json", [line_geometry(line) for line in page.lines])
    predictions = recognize_lines(backend, image, list(page.lines), out / "recognition")
    return dict(recognized=len(predictions), raw_candidates=len(predictions), excluded=0)


def score_condition(out, refs, method, plan, item, base):
    lines = (out / "recognition/predictions.jsonl").read_text().splitlines()
    predictions = [json.loads(s) for s in lines]
    geometry = read_json(out / "geometry.json")
    scored, matching, unused = score(
        refs, predictions, geometry, method, plan["matching"]["threshold"]
    )
    for record in scored:
        record.update(base)
    write_json(out / "per_reference.json", scored)
    write_json(out / "matching.json", matching)
    write_json(out / "unassigned.json", unused)
    matched = sum(r["candidate_id"] is not None for r in scored)
    item.update(
        metrics=aggregate_text(scored),
        matched=matched,
        omitted=len(refs) - matched,
        unassigned=len(unused),
    )
    return scored


def score_all(run, manifest, selected, plan, models, pending):
    """Fase 2: leitura das anotações, controle E4 e pontuação."""
    results = []
    rows_by_method = {m: [] for m in plan["methods"]}
    for row in selected:
        page_id = row["page_id"]
        folder = run / page_id
        log_event(run, "reference_parse_start", page_id=page_id)
        page = refs = reference_error = None
        try:
            page, refs = load_references(folder, row, manifest)
        except Exception as error:
            reference_error = str(error)
            write_json(
                folder / "reference_failure.json",
                dict(error=reference_error, traceback=traceback.format_exc()),
            )
        for method in plan["methods"]:
            out = folder / method
            if method == "E4_control":

                def control():
                    require(reference_error is None, reference_error)
                    return run_control(folder, out, page, models.backend)

                state = attempt(out, control)
            else:
                state = pending[page_id, method]
            base = dict(page_id=page_id, document_id=row["document_id"], method=method)
            item = dict(
                base,
                reference_lines=len(refs) if refs is not None else None,
                state=state,
                metrics=None,
                matched=None,
                omitted=None,
                unassigned=None,
            )
            if refs is not None and state["status"] == "complete":
                try:
                    rows_by_method[method] += score_condition(out, refs, method, plan, item, base)
                except Exception as error:
                    item["scoring_failure"] = str(error)
                    write_json(
                        out / "scoring_failure.json",
                        dict(error=str(error), traceback=traceback.format_exc()),
                    )
            if reference_error is not None:
                item["reference_failure"] = reference_error
            results.append(item)
    return results, rows_by_method


def execute(root, run, selected, plan, models=None):
    """Roda todas as condições página/método e grava os resultados em `run`."""
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "status.json", dict(status="running"))
    write_json(run / "plan.json", plan)
    write_json(run / "selection.json", selected)
    models = models or load_models(root, plan)
    write_json(
        run / "backend.json",
        dict(
            htr=getattr(models.backend, "details", None),
            native=getattr(models.native, "details", None),
        ),
    )
    manifest = root / plan["manifest_path"]
    start = time.perf_counter()
    pending = predict_all(run, manifest, selected, plan, models)
    results, rows_by_method = score_all(run, manifest, selected, plan, models, pending)
    summary = summarize(results, rows_by_method, list(plan["methods"]), len(selected))
    write_json(run / "pages.json", results)
    write_json(run / "summary.json", summary)
    complete = all(r["complete"] for r in summary.values())
    write_json(
        run / "status.json",
        dict(
            status="complete" if complete else "complete_with_failures",
            conditions=len(results),
            conditions_scored=sum(r["metrics"] is not None for r in results),
            seconds=time.perf_counter() - start,
            recognizer_sha256=models.recognizer_sha256,
        ),
    )
    return summary
