"""Inference on a single uploaded image: safe loading, preprocessing, calibrated
probability of "AI-generated", Grad-CAM heatmap, and honest warnings.

Used by the web app, and from the command line:
  python -I -m src.infer --model smallcnn path/to/image.jpg
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.data import build_transforms, load_config
from src.gradcam import GradCAM, heatmap_rgba, to_png_bytes
from src.models import load_checkpoint

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_PIXELS = 25_000_000   # 25 megapixels; bigger is refused (decompression-bomb guard)
DISPLAY_SIZE = 256        # size of the image + heatmap shown in the UI


class ImageError(ValueError):
    """Raised for files we refuse to process; the message is safe to show to users."""


def load_image_safe(data: bytes) -> tuple[Image.Image, dict]:
    """Bytes -> clean RGB image + facts about the original file.

    Steps and why:
    - Image.open is lazy: it reads only the header, so we can check the format and
      pixel count BEFORE decoding. A tiny file can claim to be 50,000 x 50,000
      pixels and exhaust memory when decoded (a "decompression bomb").
    - verify() checks the file structure; a corrupt/truncated file fails here.
    - exif_transpose applies the phone's rotation flag so photos are upright.
    - Rebuilding the image from its pixel array drops ALL metadata (EXIF, GPS
      location, camera serials). We never keep or store the upload.
    """
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt, (w, h) = probe.format, probe.size
            if fmt not in ALLOWED_FORMATS:
                raise ImageError(f"Unsupported image format ({fmt}). Use JPEG, PNG or WebP.")
            if w * h > MAX_PIXELS:
                raise ImageError(f"Image is too large ({w}x{h} pixels). Maximum is 25 megapixels.")
            probe.verify()
        with Image.open(io.BytesIO(data)) as img:
            # Read the JPEG quality BEFORE exif_transpose: that returns a copy
            # which no longer carries the file's quantisation tables.
            jpeg_quality = estimate_jpeg_quality(img) if fmt == "JPEG" else None
            img = ImageOps.exif_transpose(img)
            rgb = np.asarray(img.convert("RGB"))
    except ImageError:
        raise
    except Exception as exc:  # Pillow raises many different errors for bad files
        raise ImageError("The file could not be read as an image. It may be corrupt.") from exc
    return Image.fromarray(rgb), {"format": fmt, "width": w, "height": h, "jpeg_quality": jpeg_quality}


# Standard JPEG luminance quantisation table at quality 50 (from the JPEG spec, Annex K).
_STD_LUMA_Q50 = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99])


def estimate_jpeg_quality(img: Image.Image) -> int | None:
    """Rough JPEG quality (1-100) from the file's luminance quantisation table.

    Encoders scale the standard table by a factor that depends on quality:
    scale = 5000/q (q < 50) or 200 - 2q (q >= 50). We invert that formula.
    It is an estimate (some encoders use custom tables), used only for a warning.
    """
    tables = getattr(img, "quantization", None)
    if not tables or 0 not in tables:
        return None
    scale = 100 * np.mean(np.array(tables[0], dtype=float)) / np.mean(_STD_LUMA_Q50)
    q = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return int(np.clip(round(q), 1, 100))


def center_square(img: Image.Image) -> Image.Image:
    """The same region the model sees: resize-shorter-side + centre-crop keeps the central square."""
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def verdict_band(p_fake: float) -> str:
    """Three bands instead of a hard yes/no: 0.35-0.65 is reported as uncertain."""
    if p_fake < 0.35:
        return "likely real"
    if p_fake > 0.65:
        return "likely AI-generated"
    return "uncertain"


class Predictor:
    """One loaded model + its temperature. Thread-safe (the app may serve requests in parallel)."""

    def __init__(self, name: str, checkpoint: str | Path, calibration: str | Path,
                 trained_on: str = "a limited set of generators"):
        self.name = name
        self.trained_on = trained_on
        self.model, self.cfg = load_checkpoint(checkpoint, "cpu")
        d = self.cfg["data"]
        self.img_size = d["img_size"]
        self.transform = build_transforms(d["img_size"], d["mean"], d["std"], train=False)
        self.gradcam = GradCAM(self.model, self.model.gradcam_layer())
        self.temperature, self.calibrated = 1.0, False
        if Path(calibration).exists():
            self.temperature = json.loads(Path(calibration).read_text())["temperature"]
            self.calibrated = True
        self._lock = threading.Lock()  # hooks store state on the object, so one request at a time

    def predict(self, data: bytes) -> dict:
        start = time.perf_counter()
        img, info = load_image_safe(data)
        x = self.transform(img).unsqueeze(0)
        with self._lock:
            cam, logit = self.gradcam(x)
        p_fake = float(1 / (1 + np.exp(-logit / self.temperature)))

        shown = center_square(img).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BILINEAR)
        heat = heatmap_rgba(cam, (DISPLAY_SIZE, DISPLAY_SIZE))
        return {
            "model": self.name,
            "probability_fake": p_fake,
            "verdict": verdict_band(p_fake),
            "temperature": self.temperature,
            "calibrated": self.calibrated,
            "image_png": to_png_bytes(shown),       # what the model looked at (centre square)
            "heatmap_png": to_png_bytes(heat),
            "original_size": [info["width"], info["height"]],
            "latency_ms": (time.perf_counter() - start) * 1000,
            "warnings": self.warnings(info),
        }

    def warnings(self, info: dict) -> list[str]:
        w, h = info["width"], info["height"]
        out = []
        if min(w, h) > self.img_size:
            out.append(f"The image was downscaled from {w}x{h} to {self.img_size}x{self.img_size} "
                       "pixels for this model, so fine detail is lost.")
        if w != h:
            out.append("Only the central square of the image was analysed.")
        q = info["jpeg_quality"]
        if q is not None and q < 60:
            out.append(f"The image looks heavily JPEG-compressed (estimated quality {q}). "
                       "Compression removes the traces the model relies on, so results are less reliable.")
        if not self.calibrated:
            out.append("No calibration file was found; the probability is uncalibrated.")
        out.append(f"The model was trained on a limited set of generators ({self.trained_on}); "
                   "it may not recognise images from other or newer generators.")
        return out


def load_predictors(app_config: str | Path = "configs/app.yaml", root: str | Path = ".") -> dict[str, Predictor]:
    """Load every model listed in the app config whose checkpoint exists.
    Paths in the config are relative to `root` (the repo folder)."""
    root = Path(root)
    preds = {}
    for m in load_config(app_config)["models"]:
        if (root / m["checkpoint"]).exists():
            preds[m["name"]] = Predictor(m["name"], root / m["checkpoint"], root / m["calibration"],
                                         m.get("trained_on", "a limited set of generators"))
    return preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--model", default="resnet18_mix")
    parser.add_argument("--app-config", default="configs/app.yaml")
    args = parser.parse_args()
    pred = load_predictors(args.app_config)[args.model]
    out = pred.predict(Path(args.image).read_bytes())
    out.pop("image_png"), out.pop("heatmap_png")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
