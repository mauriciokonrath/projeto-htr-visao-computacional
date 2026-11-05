"""Prepara a ROI de treino da U-Net e a máscara de pixels válidos."""

import argparse
import json
from pathlib import Path

from htrcv.pilot_data import prepare_pilot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=192)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_pilot(args.manifest, args.review, args.output, args.tile_size, args.stride),
            ensure_ascii=False,
            indent=2,
        )
    )
