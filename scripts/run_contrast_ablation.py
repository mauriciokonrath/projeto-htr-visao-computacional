"""Repete o controle E4 e avalia CLAHE nas páginas de treino."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml

from htrcv.ablation import compare_lines, load_run
from htrcv.io import save_json, sha256, write_csv
from htrcv.kraken_backend import load_htr_config
from htrcv.oracle import run_oracle
from htrcv.preprocessing import validate_preprocessing


def verify_crops(run_path, rows):
    for row in rows:
        if sha256(run_path / row["crop_path"]) != row["crop_sha256"]:
            raise ValueError("Crop do run anterior foi alterado.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--htr-config", type=Path, required=True)
    parser.add_argument("--preprocessing", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = load_run(args.baseline)
    config, model = load_htr_config(args.htr_config, args.model)
    treatment = yaml.safe_load(args.preprocessing.read_text())
    validate_preprocessing(treatment)
    if treatment is None:
        raise ValueError("Forneça um tratamento CLAHE explícito.")
    if baseline["metrics"]["experiment"] != "E4" or baseline["metrics"].get("image_preprocessing"):
        raise ValueError("Baseline deve ser E4 sem tratamento.")
    if (
        config != baseline["config"]
        or config["model_sha256"] != baseline["metrics"]["model_sha256"]
    ):
        raise ValueError("Configuração HTR/modelo diverge do baseline.")
    if sha256(args.manifest) != baseline["provenance"]["manifest_sha256"]:
        raise ValueError("Manifesto diverge do baseline.")
    verify_crops(args.baseline, baseline["rows"])
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "status.json", dict(status="running"))
    save_json(
        args.output / "plan.json",
        dict(
            declared_at_utc=datetime.now(timezone.utc).isoformat(),
            split="train",
            experiment="E4 contrast ablation",
            treatment=treatment,
            stage="página inteira em tons de cinza, antes da extração das linhas",
            baseline_per_line_sha256=sha256(args.baseline / "per_line.csv"),
            baseline_provenance_sha256=sha256(args.baseline / "provenance.json"),
            manifest_sha256=sha256(args.manifest),
            model_sha256=config["model_sha256"],
            preprocessing_config_sha256=sha256(args.preprocessing),
            runner_sha256=sha256(__file__),
            selection="todas as linhas anotadas do treino",
            decision_rule="CER/WER do corpus e mudanças pareadas por documento e por linha",
        ),
    )
    try:
        print("Controle: repetindo E4 original.", flush=True)
        run_oracle(args.manifest, args.htr_config, args.output / "control", "train", model)
        control = load_run(args.output / "control")
        compare_lines(baseline["rows"], control["rows"], require_identical=True)
        # Geometria e normalização devem coincidir entre as execuções.
        for field in ("sources", "normalization", "evaluation_scope"):
            if baseline["provenance"][field] != control["provenance"][field]:
                raise ValueError(f"Controle diverge em {field}.")
        save_json(
            args.output / "control_check.json",
            dict(
                exact_predictions_and_crop_hashes=True,
                lines=len(control["rows"]),
                all_counts_equal=True,
                source_and_protocol_equal=True,
            ),
        )
        print("Controle reproduzido. Iniciando CLAHE.", flush=True)
        run_oracle(
            args.manifest,
            args.htr_config,
            args.output / "clahe",
            "train",
            model,
            preprocessing=treatment,
        )
        variant = load_run(args.output / "clahe")
        for field in (
            "sources",
            "normalization",
            "evaluation_scope",
            "manifest_sha256",
            "model_sha256",
        ):
            if control["provenance"][field] != variant["provenance"][field]:
                raise ValueError(f"Variante diverge em {field}.")
        summary, paired, documents = compare_lines(control["rows"], variant["rows"])
        summary["treatment"] = treatment
        save_json(args.output / "comparison.json", summary)
        write_csv(args.output / "paired_lines.csv", paired)
        write_csv(args.output / "paired_documents.csv", documents)
        save_json(args.output / "status.json", dict(status="complete", lines=summary["lines"]))
        print(summary, flush=True)
    except Exception as error:
        save_json(
            args.output / "status.json",
            dict(status="failed", error_type=type(error).__name__, error=str(error)),
        )
        raise


if __name__ == "__main__":
    main()
