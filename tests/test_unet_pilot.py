import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from test_pilot_data import pilot_fixture

from htrcv.pilot_data import prepare_pilot

torch = pytest.importorskip("torch")
u = importlib.import_module("htrcv.unet_pilot")


def test_ignored_pixels_do_not_affect_loss_or_gradient():
    z = torch.tensor([0.2, -0.4, 3.0, -2.0], requires_grad=True)
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])
    valid = torch.tensor([True, True, False, False])
    loss = u.masked_loss(z, target, valid)
    loss.backward()
    assert torch.equal(z.grad[2:], torch.zeros(2))
    changed = torch.tensor([0.2, -0.4, -999.0, 999.0], requires_grad=True)
    other = u.masked_loss(changed, torch.tensor([1.0, 0.0, 0.0, 1.0]), valid)
    other.backward()
    assert torch.equal(loss, other) and torch.equal(z.grad, changed.grad)
    assert z.grad[:2].abs().sum() > 0
    with pytest.raises(ValueError, match="sem pixels"):
        u.masked_loss(z, target, torch.zeros(4, dtype=torch.bool))


def test_foreground_and_background_baselines_ignore_unknown_pixels():
    target = np.array([[255, 0], [255, 0]], np.uint8)
    valid = np.array([[255, 255], [0, 0]], np.uint8)
    a = u.score_probability(np.ones((2, 2), np.float32), target, valid)
    b = u.score_probability(np.array([[1.0, 1.0], [0.0, 0.0]], np.float32), target, valid)
    assert a == b and a["iou"] == 0.5 and a["balanced_accuracy"] == 0.5


def test_unet_shape_backward_and_checkpoint_smoke(tmp_path):
    model = u.SmallUNet(2)
    x = torch.rand(2, 1, 33, 35)
    out = model(x)
    assert out.shape == x.shape
    u.masked_loss(out, torch.zeros_like(out), torch.ones_like(out, dtype=torch.bool)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    manifest, review = pilot_fixture(tmp_path)
    data = tmp_path / "data"
    prepare_pilot(manifest, review, data, tile_size=32, stride=24)
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/pilot_unet.yaml").read_text())
    config.update(steps=2, base_channels=2, evaluate_every=1, batch_size=2, threads=1)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    result = u.run_pilot(data, path, tmp_path / "run")
    assert result["steps"] == 2 and result["checkpoint_reload_exact"]
    assert not result["generalization_evaluated"] and not result["htr_evaluated"]
    assert json.loads((tmp_path / "run/status.json").read_text())["status"] == "complete"
