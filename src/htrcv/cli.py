import argparse
import json
from pathlib import Path

from .oracle import run_oracle
from .workflow import evaluate_transcriptions, prepare, run_classical


def main():
    parser = argparse.ArgumentParser(
        description="Experimentos reprodutíveis para manuscritos históricos."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser(
        "prepare", help="Importar PAGE XML, gerar máscaras, recortes GT e splits."
    )
    p.add_argument("--raw", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--dataset", default="iforal")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--document-map",
        type=Path,
        help="CSV pagexml_path,document_id,writer_id; caminhos relativos a --raw.",
    )
    p = commands.add_parser(
        "classical", help="Executar segmentação clássica; CER/WER ficam nulos sem HTR."
    )
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--config", default=Path("configs/classical.yaml"), type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument(
        "--coverage-review",
        type=Path,
        help="CSV revisado de cobertura GT; sem ele IoU/Dice ficam nulos.",
    )
    p = commands.add_parser(
        "evaluate-text", help="CER/WER de transcrições reais fornecidas em CSV page_id,text."
    )
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p = commands.add_parser("oracle", help="E4: Kraken fixo sobre baselines/polígonos GT.")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--config", default=Path("configs/htr.yaml"), type=Path)
    p.add_argument(
        "--model",
        type=Path,
        help="Local alternativo do mesmo checkpoint; hash permanece obrigatório.",
    )
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.raw, args.output, args.dataset, args.seed, args.document_map)
        elif args.command == "classical":
            result = run_classical(
                args.manifest, args.config, args.output, args.split, args.coverage_review
            )
        elif args.command == "oracle":
            result = run_oracle(args.manifest, args.config, args.output, args.split, args.model)
        else:
            result = evaluate_transcriptions(args.reference, args.predictions, args.output)
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Erro: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
