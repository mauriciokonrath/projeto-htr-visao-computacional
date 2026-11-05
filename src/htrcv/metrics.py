"""CER/WER com normalização Unicode e médias micro por corpus."""

import unicodedata

import numpy as np


def normalize(text: str) -> str:
    """NFC e espaços colapsados; mantém acentos, pontuação e caixa."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def edit_distance(reference, hypothesis) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    row = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        previous, row = row, [i]
        for j, right in enumerate(hypothesis, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
    return row[-1]


def text_counts(reference: str, hypothesis: str) -> dict:
    reference, hypothesis = normalize(reference), normalize(hypothesis)
    chars, words = len(reference), len(reference.split())
    char_errors = edit_distance(reference, hypothesis)
    word_errors = edit_distance(reference.split(), hypothesis.split())
    return dict(
        char_errors=char_errors,
        reference_chars=chars,
        word_errors=word_errors,
        reference_words=words,
        cer=char_errors / chars if chars else None,
        wer=word_errors / words if words else None,
    )


def aggregate_text(rows) -> dict:
    totals = {
        key: sum(row[key] for row in rows)
        for key in ("char_errors", "reference_chars", "word_errors", "reference_words")
    }
    totals["cer"] = (
        totals["char_errors"] / totals["reference_chars"] if totals["reference_chars"] else None
    )
    totals["wer"] = (
        totals["word_errors"] / totals["reference_words"] if totals["reference_words"] else None
    )
    return totals


def segmentation_counts(reference, prediction):
    if reference.shape != prediction.shape:
        raise ValueError("Máscaras com dimensões diferentes.")
    reference, prediction = reference.astype(bool), prediction.astype(bool)
    tp = int(np.count_nonzero(reference & prediction))
    fp = int(np.count_nonzero(~reference & prediction))
    fn = int(np.count_nonzero(reference & ~prediction))
    # Vazio/vazio é acerto; predição vazia para referência não vazia tem recall 0.
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        iou=tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        dice=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        precision=tp / (tp + fp) if tp + fp else float(fn == 0),
        recall=tp / (tp + fn) if tp + fn else float(fp == 0),
    )
