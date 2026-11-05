"""Gera as faixas de baseline usadas como alvo da U-Net piloto."""

import argparse
import json
from pathlib import Path

from htrcv.baseline_pilot import prepare_baseline_pilot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_baseline_pilot(args.source, args.config, args.output), indent=2))
