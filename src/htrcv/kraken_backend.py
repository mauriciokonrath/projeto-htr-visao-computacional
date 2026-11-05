"""Adaptador para o reconhecedor do Kraken 5.3.0."""

import logging
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .io import sha256


@dataclass(frozen=True)
class Recognition:
    text: str
    crop: Image.Image
    mean_confidence: float | None


def load_htr_config(path, model_override=None):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "backend",
        "backend_version",
        "model_path",
        "model_source",
        "model_sha256",
        "device",
        "threads",
        "seed",
        "padding",
        "crop_protocol",
        "bidi_reordering",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"Config HTR deve conter exatamente {sorted(required)}.")
    if config["backend"] != "kraken" or config["backend_version"] != "5.3.0":
        raise ValueError("Backend validado: kraken 5.3.0.")
    if config["crop_protocol"] not in {
        "kraken_gt_baseline",
        "kraken_native_baseline",
        "kraken_predicted_unet_baseline",
        "kraken_predicted_rectangle_horizontal",
        "kraken_predicted_ink_path",
        "kraken_predicted_guarded_ink_path",
    }:
        raise ValueError("Protocolo de recorte não suportado.")
    if config["device"] not in {"cpu", "mps"}:
        raise ValueError("Escolha device cpu ou mps explicitamente.")
    if type(config["bidi_reordering"]) is not bool:
        raise ValueError("bidi_reordering deve ser booleano.")
    for key, minimum in (("threads", 1), ("padding", 0), ("seed", 0)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} deve ser inteiro >= {minimum}.")
    digest = config["model_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("model_sha256 deve ser um SHA-256 hexadecimal.")
    model = (
        Path(model_override).resolve()
        if model_override
        else (path.parent / config["model_path"]).resolve()
    )
    if not model.is_file():
        raise FileNotFoundError(f"Modelo ausente: {model}. Execute scripts/download_traprinq.py.")
    if sha256(model) != digest:
        raise ValueError("Hash do checkpoint diverge da configuração; reconhecimento recusado.")
    return config, model


def validate_line_geometry(line, width, height):
    if len(line.polygon) < 3 or len(line.baseline) < 2:
        raise ValueError(f"Linha {line.id}: polígono e baseline são obrigatórios.")
    if any(not (0 <= x < width and 0 <= y < height) for x, y in (*line.polygon, *line.baseline)):
        raise ValueError(f"Linha {line.id}: geometria fora dos pixels da imagem.")
    length = np.linalg.norm(np.diff(np.asarray(line.baseline, dtype=float), axis=0), axis=1).sum()
    if length < 5:
        raise ValueError(f"Linha {line.id}: baseline menor que 5 pixels.")


class _WarningCapture(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class KrakenBackend:
    def __init__(self, model_path, config):
        # Import tardio: o ambiente sem torch/kraken continua funcional.
        import torch
        from kraken.lib import models

        if version("kraken") != config["backend_version"]:
            raise ValueError("Versão Kraken diferente da versão fixada no config.")
        if sha256(model_path) != config["model_sha256"]:
            raise ValueError("Checkpoint alterado antes do carregamento.")
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        torch.set_num_threads(config["threads"])
        if config["device"] == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS indisponível neste processo. Escolha CPU explicitamente.")
        self.config = config
        self.model = models.load_any(str(model_path), train=False, device=config["device"])
        if self.model.seg_type != "baselines" or self.model.nn.one_channel_mode != "L":
            raise ValueError("Esta integração exige modelo de baselines com entrada grayscale.")
        self.details = dict(
            kraken=version("kraken"),
            torch=torch.__version__,
            device=config["device"],
            input_shape=list(self.model.nn.input),
            segmentation_type=self.model.seg_type,
            legacy_polygons=bool(self.model.nn.use_legacy_polygons),
            mps_available=torch.backends.mps.is_available(),
        )

    def recognize_line(self, image, line):
        import torch
        from kraken.containers import BaselineLine, Segmentation
        from kraken.rpred import rpred

        validate_line_geometry(line, image.width, image.height)
        boundary = list(line.polygon)
        if boundary[0] != boundary[-1]:
            boundary.append(boundary[0])
        geometry = BaselineLine(id=line.id, baseline=list(line.baseline), boundary=boundary)
        bounds = Segmentation(
            type="baselines",
            imagename=None,
            text_direction="horizontal-lr",
            script_detection=False,
            lines=[geometry],
        )
        capture = _WarningCapture()
        logger = logging.getLogger("kraken.rpred")
        logger.addHandler(capture)
        try:
            with torch.inference_mode():
                iterator = rpred(
                    self.model,
                    image.convert("L"),
                    bounds,
                    pad=self.config["padding"],
                    bidi_reordering=self.config["bidi_reordering"],
                )
                record = next(iterator)
            # O Kraken pode devolver registro vazio quando a extração falha;
            # isso é tratado como erro, diferente de uma predição vazia válida.
            if capture.messages:
                raise RuntimeError("Kraken não concluiu a linha: " + "; ".join(capture.messages))
            if record.id != line.id or not hasattr(iterator, "box"):
                raise RuntimeError("Registro/crop não corresponde à linha solicitada.")
            confidence = float(np.mean(record.confidences)) if record.confidences else None
            return Recognition(record.prediction, iterator.box.copy(), confidence)
        finally:
            logger.removeHandler(capture)
