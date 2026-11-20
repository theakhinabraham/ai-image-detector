"""Shared fixtures. Tests never need the real datasets or trained checkpoints:
they build a tiny UNTRAINED SmallCNN checkpoint in a temp folder."""
import io

import pytest
import torch
import yaml
from PIL import Image

from src.models import SmallCNN

CFG = {
    "run_name": "testcnn", "seed": 0,
    "data": {"img_size": 32, "mean": [0.5] * 3, "std": [0.5] * 3, "num_workers": 0},
    "model": {"name": "smallcnn", "dropout": 0.3},
}


@pytest.fixture
def app_root(tmp_path):
    """A fake repo root with one checkpoint, a calibration file and an app config."""
    torch.manual_seed(0)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "reports").mkdir()
    torch.save({"state_dict": SmallCNN().state_dict(), "config": CFG, "epoch": 0},
               tmp_path / "checkpoints" / "testcnn.pt")
    (tmp_path / "reports" / "calibration_testcnn.json").write_text('{"temperature": 1.5}')
    app_cfg = {"models": [{"name": "testcnn", "label": "Test CNN",
                           "checkpoint": "checkpoints/testcnn.pt",
                           "calibration": "reports/calibration_testcnn.json",
                           "metrics": "reports/metrics_testcnn.json"}]}
    (tmp_path / "app.yaml").write_text(yaml.safe_dump(app_cfg))
    return tmp_path


def image_bytes(fmt="PNG", size=(64, 48), **save_kwargs) -> bytes:
    img = Image.new("RGB", size)
    for x in range(size[0]):  # a gradient, so the image is not perfectly flat
        for y in range(size[1]):
            img.putpixel((x, y), (x * 4 % 256, y * 5 % 256, (x + y) % 256))
    buf = io.BytesIO()
    img.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()
