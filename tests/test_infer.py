import io

import numpy as np
import pytest
import torch
from PIL import Image

from src.infer import ImageError, Predictor, estimate_jpeg_quality, load_image_safe, verdict_band
from tests.conftest import image_bytes


def test_load_image_safe_strips_exif_and_returns_rgb():
    img = Image.new("RGBA", (40, 30), (10, 20, 30, 128))
    exif = Image.Exif()
    exif[0x010F] = "SecretCameraMaker"   # EXIF 'Make' tag
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", exif=exif)
    clean, info = load_image_safe(buf.getvalue())
    assert clean.mode == "RGB" and clean.size == (40, 30)
    assert not clean.info.get("exif") and len(clean.getexif()) == 0
    assert info["format"] == "JPEG"


@pytest.mark.parametrize("data", [b"not an image at all", image_bytes("PNG")[:60]])
def test_corrupt_files_rejected(data):
    with pytest.raises(ImageError):
        load_image_safe(data)


def test_wrong_real_format_rejected():
    with pytest.raises(ImageError, match="Unsupported"):
        load_image_safe(image_bytes("GIF"))


def test_too_many_pixels_rejected():
    buf = io.BytesIO()
    Image.new("L", (6000, 5000)).save(buf, format="PNG")  # 30 MP > 25 MP limit
    with pytest.raises(ImageError, match="too large"):
        load_image_safe(buf.getvalue())


def test_jpeg_quality_estimate_is_close():
    for q in (30, 50, 90):
        with Image.open(io.BytesIO(image_bytes("JPEG", quality=q))) as img:
            assert abs(estimate_jpeg_quality(img) - q) <= 3


def test_verdict_bands():
    assert verdict_band(0.1) == "likely real"
    assert verdict_band(0.35) == "uncertain" and verdict_band(0.65) == "uncertain"
    assert verdict_band(0.9) == "likely AI-generated"


def test_predictor_output(app_root):
    p = Predictor("testcnn", app_root / "checkpoints/testcnn.pt", app_root / "reports/calibration_testcnn.json")
    out = p.predict(image_bytes("PNG", size=(300, 200)))
    assert 0.0 <= out["probability_fake"] <= 1.0
    assert out["calibrated"] and out["temperature"] == 1.5
    assert any("downscaled" in w for w in out["warnings"])
    assert any("central square" in w for w in out["warnings"])


def test_preprocessing_shape_and_dtype(app_root):
    p = Predictor("testcnn", app_root / "checkpoints/testcnn.pt", app_root / "reports/calibration_testcnn.json")
    img, _ = load_image_safe(image_bytes("WEBP", size=(500, 300)))
    x = p.transform(img)
    assert x.shape == (3, 32, 32) and x.dtype == torch.float32


def test_heavy_compression_warning_through_full_pipeline(app_root):
    """Regression test: the quality must survive load_image_safe (exif_transpose copies the image)."""
    _, info = load_image_safe(image_bytes("JPEG", size=(100, 100), quality=25))
    assert info["jpeg_quality"] is not None and abs(info["jpeg_quality"] - 25) <= 3
    p = Predictor("testcnn", app_root / "checkpoints/testcnn.pt", app_root / "reports/calibration_testcnn.json")
    out = p.predict(image_bytes("JPEG", size=(100, 100), quality=25))
    assert any("heavily JPEG-compressed" in w for w in out["warnings"])
