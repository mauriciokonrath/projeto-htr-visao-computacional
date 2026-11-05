"""Comparação pareada, linha a linha, entre duas execuções do reconhecedor."""

import json
from collections import defaultdict
from pathlib import Path

import yaml

from .io import read_csv
from .metrics import aggregate_text, normalize, text_counts

COUNTS = ("char_errors", "reference_chars", "word_errors", "reference_words")


def checked_rows(rows):
    indexed = {}
    for raw in rows:
        row = dict(raw)
        key = (row["page_id"], row["line_id"])
        if key in indexed:
            raise ValueError("Linha duplicada.")
        if row["split"] != "train" or row["status"] != "recognized":
            raise ValueError("Comparação exige todas as linhas reconhecidas do treino.")
        if any(normalize(row[k + "_raw"]) != row[k] for k in ("reference", "prediction")):
            raise ValueError("Texto normalizado diverge do original.")
        counts = text_counts(row["reference"], row["prediction"])
        if any(int(row[k]) != counts[k] for k in COUNTS):
            raise ValueError("Contagens divergem das transcrições.")
        for rate in ("cer", "wer"):
            value = None if row[rate] in (None, "") else float(row[rate])
            if value != counts[rate]:
                raise ValueError("Taxa diverge das transcrições.")
        row.update(counts)
        indexed[key] = row
    if not indexed:
        raise ValueError("Run vazio.")
    return indexed


def load_run(path):
    path = Path(path)
    metrics = json.loads((path / "metrics.json").read_text())
    status = json.loads((path / "status.json").read_text())
    provenance = json.loads((path / "provenance.json").read_text())
    config = yaml.safe_load((path / "config.yaml").read_text())
    rows = list(checked_rows(read_csv(path / "per_line.csv")).values())
    totals = aggregate_text(rows)
    if status["status"] != "complete" or metrics["split"] != "train":
        raise ValueError("Run deve estar completo e ser do treino.")
    if metrics["lines"] != len(rows) or any(metrics[k] != totals[k] for k in totals):
        raise ValueError("Totais do run inconsistentes.")
    if metrics["pages"] != len({r["page_id"] for r in rows}):
        raise ValueError("Número de páginas inconsistente.")
    return dict(rows=rows, metrics=metrics, provenance=provenance, config=config)


def compare_lines(baseline_rows, variant_rows, require_identical=False):
    baseline, variant = checked_rows(baseline_rows), checked_rows(variant_rows)
    if baseline.keys() != variant.keys():
        raise ValueError("Universos de linhas diferentes; omissões não são permitidas.")
    paired = []
    grouped = defaultdict(lambda: ([], []))
    for key in sorted(baseline):
        a, b = baseline[key], variant[key]
        for field in ("document_id", "reference_raw", "reference", "reading_index"):
            if str(a[field]) != str(b[field]):
                raise ValueError(f"Pareamento diverge em {field}.")
        if require_identical and any(
            a[k] != b[k] for k in ("prediction_raw", "prediction", "crop_sha256")
        ):
            raise ValueError("Controle não reproduz exatamente predição e crop originais.")
        paired.append(
            dict(
                page_id=key[0],
                line_id=key[1],
                document_id=a["document_id"],
                reference=a["reference"],
                baseline_prediction=a["prediction"],
                variant_prediction=b["prediction"],
                reference_chars=a["reference_chars"],
                reference_words=a["reference_words"],
                baseline_char_errors=a["char_errors"],
                variant_char_errors=b["char_errors"],
                baseline_word_errors=a["word_errors"],
                variant_word_errors=b["word_errors"],
                delta_char_errors=b["char_errors"] - a["char_errors"],
                delta_word_errors=b["word_errors"] - a["word_errors"],
                baseline_cer=a["cer"],
                variant_cer=b["cer"],
            )
        )
        grouped[a["document_id"]][0].append(a)
        grouped[a["document_id"]][1].append(b)
    documents = []
    for doc, (aa, bb) in sorted(grouped.items()):
        a, b = aggregate_text(aa), aggregate_text(bb)
        documents.append(
            dict(
                document_id=doc,
                lines=len(aa),
                **{f"baseline_{k}": v for k, v in a.items()},
                **{f"variant_{k}": v for k, v in b.items()},
                delta_cer_pp=100 * (b["cer"] - a["cer"]) if a["cer"] is not None else None,
                delta_wer_pp=100 * (b["wer"] - a["wer"]) if a["wer"] is not None else None,
            )
        )
    a, b = aggregate_text(list(baseline.values())), aggregate_text(list(variant.values()))
    summary = dict(
        lines=len(paired),
        documents=len(documents),
        baseline=a,
        variant=b,
        delta_cer_pp=100 * (b["cer"] - a["cer"]) if a["cer"] is not None else None,
        delta_wer_pp=100 * (b["wer"] - a["wer"]) if a["wer"] is not None else None,
        improved_lines=sum(r["delta_char_errors"] < 0 for r in paired),
        worsened_lines=sum(r["delta_char_errors"] > 0 for r in paired),
        tied_lines=sum(r["delta_char_errors"] == 0 for r in paired),
        identical_predictions=sum(
            a["prediction"] == variant[k]["prediction"] for k, a in baseline.items()
        ),
        direction="variante menos controle; negativo favorece a variante",
        scope="comparação pareada descritiva nas páginas de treino",
    )
    return summary, paired, documents
