"""Treina a U-Net piloto em uma ROI de treino."""

import argparse
import json
from pathlib import Path

from htrcv.unet_pilot import run_pilot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(run_pilot(args.dataset, args.config, args.output), ensure_ascii=False, indent=2)
    )
