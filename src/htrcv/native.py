"""E0: segmentação nativa BLLA do Kraken."""

from importlib import metadata

import numpy as np

from .io import sha256
from .kraken_backend import validate_line_geometry
from .pagexml import Line


class NativeSegmenter:
    def __init__(self, model, config):
        import torch
        from kraken.lib import vgsl

        if metadata.version("kraken") != config["backend_version"]:
            raise ValueError("Versão Kraken divergente.")
        if sha256(model) != config["model_sha256"]:
            raise ValueError("Segmentador alterado antes do carregamento.")
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        torch.set_num_threads(config["threads"])
        self.config = config
        self.model = vgsl.TorchVGSLModel.load_model(str(model))
        self.details = dict(
            model_type=self.model.model_type,
            input_shape=list(self.model.input),
            model_sha256=sha256(model),
            kraken=metadata.version("kraken"),
        )

    def segment(self, image):
        import torch
        from kraken import blla

        with torch.inference_mode():
            result = blla.segment(
                image,
                model=self.model,
                device=self.config["device"],
                text_direction=self.config["text_direction"],
                mask=None,
                raise_on_error=True,
                autocast=False,
            )
        if result.type != "baselines":
            raise ValueError("Segmentador não devolveu baselines.")
        lines = []
        raw = []
        for i, item in enumerate(result.lines):
            line = Line(
                f"e0_{i:04d}",
                "native_prediction",
                tuple(tuple(int(v) for v in xy) for xy in item.boundary),
                tuple(tuple(int(v) for v in xy) for xy in item.baseline),
                None,
            )
            # Geometria nativa usada sem correções.
            validate_line_geometry(line, image.width, image.height)
            lines.append(line)
            raw.append(
                dict(
                    native_id=item.id,
                    candidate_index=i,
                    candidate_id=line.id,
                    polygon=line.polygon,
                    baseline=line.baseline,
                    box=line.bbox,
                )
            )
        return lines, raw
