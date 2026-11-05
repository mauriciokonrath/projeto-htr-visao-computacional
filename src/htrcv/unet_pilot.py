"""U-Net pequena para detecção de baselines, treinada em uma ROI."""

import json
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from .io import load_gray, save_image, save_json, sha256, source_hashes, write_csv
from .metrics import segmentation_counts
from .pilot_data import load_pilot, padded_tile, reconstruct_tiles, tile_origins


class Block(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(
            nn.Conv2d(incoming, outgoing, 3, padding=1),
            nn.GroupNorm(2, outgoing),
            nn.ReLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1),
            nn.GroupNorm(2, outgoing),
            nn.ReLU(),
        )


class SmallUNet(nn.Module):
    """Três níveis de codificação/decodificação com skip connections."""

    def __init__(self, base_channels=8):
        super().__init__()
        if type(base_channels) is not int or base_channels < 2 or base_channels % 2:
            raise ValueError("base_channels precisa ser inteiro par >= 2.")
        c = base_channels
        self.enc1, self.enc2, self.enc3 = Block(1, c), Block(c, 2 * c), Block(2 * c, 4 * c)
        self.bridge = Block(4 * c, 8 * c)
        self.dec3, self.dec2, self.dec1 = Block(12 * c, 4 * c), Block(6 * c, 2 * c), Block(3 * c, c)
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        z = self.bridge(F.max_pool2d(e3, 2))
        for skip, decoder in ((e3, self.dec3), (e2, self.dec2), (e1, self.dec1)):
            z = decoder(
                torch.cat(
                    (
                        F.interpolate(
                            z, size=skip.shape[-2:], mode="bilinear", align_corners=False
                        ),
                        skip,
                    ),
                    dim=1,
                )
            )
        return self.head(z)


def masked_loss(logits, target, valid):
    """BCE balanceada + Dice suave, apenas nos pixels válidos."""
    if logits.shape != target.shape or target.shape != valid.shape:
        raise ValueError("Logits, alvo e máscara precisam ter a mesma forma.")
    mask = valid.bool()
    if not mask.any():
        raise ValueError("Batch sem pixels válidos.")
    z, y = logits[mask], target[mask]
    if (
        not torch.isfinite(z).all()
        or not torch.isfinite(y).all()
        or not ((y == 0) | (y == 1)).all()
    ):
        raise ValueError("Logits/alvos válidos inválidos.")
    raw = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    positive, negative = y > 0, y == 0
    bce = (
        (raw[positive].mean() + raw[negative].mean()) / 2
        if positive.any() and negative.any()
        else raw.mean()
    )
    probability = z.sigmoid()
    dice_loss = 1 - (2 * (probability * y).sum() + 1) / (probability.sum() + y.sum() + 1)
    return 0.5 * bce + 0.5 * dice_loss


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    required = {
        "device",
        "threads",
        "seed",
        "steps",
        "batch_size",
        "base_channels",
        "learning_rate",
        "weight_decay",
        "evaluate_every",
        "threshold",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("Configuração do piloto inválida.")
    if config["device"] not in {"cpu", "mps"} or config["threshold"] != 0.5:
        raise ValueError("Dispositivo inválido ou limiar diferente do protocolo 0.5.")
    for key in ("threads", "steps", "batch_size", "base_channels", "evaluate_every"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("Parâmetro inteiro inválido: " + key)
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("Seed inválida.")
    if not (0 < config["learning_rate"] < 1 and 0 <= config["weight_decay"] < 1):
        raise ValueError("Taxa de aprendizado/regularização inválida.")
    return config


def score_probability(probability, target, valid, threshold=0.5):
    mask = valid > 0
    if not mask.any() or not np.isfinite(probability[mask]).all():
        raise ValueError("Predição inválida ou ROI vazia.")
    truth = target[mask] > 0
    predicted = probability[mask] >= threshold
    counts = segmentation_counts(truth, predicted)
    tn = int(np.count_nonzero(~truth & ~predicted))
    specificity = tn / (tn + counts["fp"]) if tn + counts["fp"] else None
    loss = masked_loss(
        torch.logit(torch.from_numpy(np.clip(probability[mask], 1e-6, 1 - 1e-6))),
        torch.from_numpy(truth.astype(np.float32)),
        torch.ones(len(truth), dtype=torch.bool),
    )
    return dict(
        **counts,
        tn=tn,
        specificity=specificity,
        balanced_accuracy=(counts["recall"] + specificity) / 2 if specificity is not None else None,
        loss=float(loss),
        valid_pixels=int(mask.sum()),
        predicted_positive_pixels=int(predicted.sum()),
    )


def predict_reconstructed(model, images, rows, arrays, batch_size, device):
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for i in range(0, len(images), batch_size):
            probs = model(images[i : i + batch_size].to(device)).sigmoid().cpu().numpy()[:, 0]
            probabilities.extend(probs)
    return reconstruct_tiles(probabilities, rows, arrays["image"].shape, arrays["valid"])


def run_pilot(dataset, config_path, output):
    dataset, output = Path(dataset), Path(output)
    meta, rows, arrays = load_pilot(dataset)
    config = load_config(config_path)
    if config["device"] == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS indisponível; use device cpu.")
    if meta["tile_size"] < 16:
        raise ValueError("Tile pequeno demais para três níveis U-Net.")
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "status.json", dict(status="running", phase="training"))
    save_json(
        output / "plan.json",
        dict(
            declared_at_utc=datetime.now(timezone.utc).isoformat(),
            purpose="U-Net piloto treinada em uma ROI de treino",
            target=meta.get("target_description", "união dos polígonos de linha"),
            loss="0,5 BCE balanceada + 0,5 Dice suave, nos pixels válidos",
            checkpoint="checkpoint do último passo, sem augmentation",
            reconstruction="média das probabilidades sobrepostas, limiar 0,5",
            input="tons de cinza / 255",
            config=config,
        ),
    )
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    save_json(
        output / "provenance.json",
        dict(
            dataset_sha256=sha256(dataset / "dataset.json"),
            dataset=meta,
            source_hashes=source_hashes(),
            torch=torch.__version__,
            mps_available=torch.backends.mps.is_available(),
            actual_device=config["device"],
        ),
    )
    (output / "environment.txt").write_text(
        "\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()))
        + "\n"
    )
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(config["seed"])
    images = torch.from_numpy(
        np.stack([load_gray(dataset / r["image_path"]) for r in rows])[:, None].astype(np.float32)
        / 255
    )
    targets = torch.from_numpy(
        np.stack([load_gray(dataset / r["target_path"]) for r in rows])[:, None].astype(np.float32)
        / 255
    )
    valid = torch.from_numpy(
        np.stack([load_gray(dataset / r["valid_path"]) for r in rows])[:, None] > 0
    )
    started, history, step = time.perf_counter(), [], 0
    try:
        model = SmallUNet(config["base_channels"]).to(config["device"])
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        initial, _ = predict_reconstructed(
            model, images, rows, arrays, config["batch_size"], config["device"]
        )
        initial_metrics = score_probability(initial, arrays["target"], arrays["valid"])
        history.append(dict(step=0, **initial_metrics))
        np.save(output / "initial_probability.npy", initial)
        save_json(
            output / "baselines.json",
            {
                "random_initialization": initial_metrics,
                "all_foreground": score_probability(
                    np.ones_like(initial), arrays["target"], arrays["valid"]
                ),
                "all_background": score_probability(
                    np.zeros_like(initial), arrays["target"], arrays["valid"]
                ),
            },
        )
        with (output / "training.jsonl").open("w") as journal:
            for step in range(1, config["steps"] + 1):
                model.train()
                indices = rng.choice(
                    len(rows), size=config["batch_size"], replace=len(rows) < config["batch_size"]
                )
                optimizer.zero_grad(set_to_none=True)
                logits = model(images[indices].to(config["device"]))
                loss = masked_loss(
                    logits,
                    targets[indices].to(config["device"]),
                    valid[indices].to(config["device"]),
                )
                loss.backward()
                if not all(
                    p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
                ):
                    raise RuntimeError("Gradiente não finito.")
                optimizer.step()
                journal.write(
                    json.dumps(
                        dict(
                            step=step,
                            batch_loss=float(loss.detach()),
                            tiles=[rows[i]["tile_id"] for i in indices],
                        )
                    )
                    + "\n"
                )
                journal.flush()
                if step % config["evaluate_every"] == 0 or step == config["steps"]:
                    probability, counts = predict_reconstructed(
                        model, images, rows, arrays, config["batch_size"], config["device"]
                    )
                    score = score_probability(probability, arrays["target"], arrays["valid"])
                    history.append(dict(step=step, **score))
                    write_csv(output / "history.csv", history)
                    print(
                        f"step {step}/{config['steps']}: loss={score['loss']:.4f}, IoU={score['iou']:.4f}, Dice={score['dice']:.4f}",
                        flush=True,
                    )
        checkpoint = output / "final.pt"
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(
            dict(
                model_state=cpu_state,
                base_channels=config["base_channels"],
                config=config,
                dataset_sha256=sha256(dataset / "dataset.json"),
                step=step,
            ),
            checkpoint,
        )
        # Recarrega o checkpoint salvo e confere a saída reconstruída.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        restored = SmallUNet(payload["base_channels"]).to(config["device"])
        restored.load_state_dict(payload["model_state"])
        reloaded, _ = predict_reconstructed(
            restored, images, rows, arrays, config["batch_size"], config["device"]
        )
        if not np.array_equal(probability, reloaded):
            raise RuntimeError("Checkpoint não reproduz a predição final.")
        np.save(output / "final_probability.npy", probability)
        save_image(
            output / "prediction.png",
            ((probability >= 0.5) & (arrays["valid"] > 0)).astype(np.uint8) * 255,
        )
        save_image(output / "tile_coverage.png", counts.astype(np.uint16))
        save_image(output / "valid.png", arrays["valid"])
        result = dict(
            experiment="E2_unet_piloto",
            target_kind=meta.get("target_kind", "line_envelopes"),
            split="train",
            pages=1,
            annotated_lines=meta["annotated_lines"],
            tiles=len(rows),
            steps=step,
            parameters=sum(p.numel() for p in model.parameters()),
            initial=initial_metrics,
            final=score,
            checkpoint_sha256=sha256(checkpoint),
            checkpoint_reload_exact=True,
            elapsed_seconds=time.perf_counter() - started,
            actual_device=config["device"],
            generalization_evaluated=False,
            htr_evaluated=False,
            full_page_metrics=None,
            scope="métricas na mesma ROI usada no treino",
        )
        save_json(output / "metrics.json", result)
        save_json(output / "status.json", dict(status="complete", steps=step))
        return result
    except Exception as error:
        save_json(
            output / "status.json",
            dict(status="failed", step=step, error_type=type(error).__name__, error=str(error)),
        )
        raise


def predict_page(model, gray, plan):
    """Inferência em ladrilhos sobre a página inteira; média nas sobreposições."""
    import torch

    size = plan["tile_size"]
    rows = [
        dict(x=x, y=y)
        for y in tile_origins(gray.shape[0], size, plan["stride"])
        for x in tile_origins(gray.shape[1], size, plan["stride"])
    ]
    images = torch.from_numpy(
        np.stack([padded_tile(gray, r["x"], r["y"], size, 255) for r in rows])[:, None].astype(
            np.float32
        )
        / 255
    )
    valid = np.ones_like(gray, np.uint8)
    probability, coverage = predict_reconstructed(
        model, images, rows, dict(image=gray, valid=valid), plan["batch_size"], "cpu"
    )
    return probability, valid, rows, coverage
