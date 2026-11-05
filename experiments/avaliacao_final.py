"""Executa a comparação E0–E4 nas páginas de teste do iForal.

Uso, a partir da raiz do repositório:

    python experiments/avaliacao_final.py --output runs/avaliacao_final
    python experiments/avaliacao_final.py --pages p_e7466ba2bd7782e4 --output runs/dev
"""

import argparse
import json
from pathlib import Path

from htrcv.final_evaluation import execute, read_json, select_pages

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/avaliacao_final.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pages", nargs="+", help="IDs de página (padrão: split do config)")
    args = parser.parse_args()

    plan = read_json(args.config)
    selected = select_pages(ROOT / plan["manifest_path"], plan.get("split"), args.pages)
    if not selected:
        parser.error("Nenhuma página selecionada.")
    summary = execute(ROOT, args.output.resolve(), selected, plan)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
