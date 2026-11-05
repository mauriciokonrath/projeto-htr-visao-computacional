"""Divisão treino/validação/teste por documento (e por escriba, se houver)."""

import math
import random


def grouped_splits(rows, seed=42, ratios=(0.70, 0.15, 0.15)):
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("Três proporções positivas devem somar 1.")
    if any(not row.get("document_id") for row in rows):
        raise ValueError("document_id obrigatório; não inferir a partir de linhas.")
    documents = sorted({row["document_id"] for row in rows})
    parent = {d: d for d in documents}

    def root(d):
        while parent[d] != d:
            parent[d] = parent[parent[d]]
            d = parent[d]
        return d

    writers = {}
    for row in rows:
        writer, doc = row.get("writer_id"), row["document_id"]
        if writer:
            if writer in writers:
                parent[root(doc)] = root(writers[writer])
            writers[writer] = doc
    groups = sorted({root(d) for d in documents})
    if len(groups) < 3:
        raise ValueError("São necessários ao menos 3 grupos independentes para train/val/test.")
    random.Random(seed).shuffle(groups)
    n = len(groups)
    train_n = max(1, min(n - 2, int(n * ratios[0])))
    val_n = max(1, min(n - train_n - 1, int(n * ratios[1])))
    assignment = {
        g: "train" if i < train_n else "val" if i < train_n + val_n else "test"
        for i, g in enumerate(groups)
    }
    return {doc: assignment[root(doc)] for doc in documents}
