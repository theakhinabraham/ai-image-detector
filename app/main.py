"""FastAPI web app. Binds to 127.0.0.1 only; uploads are processed in memory and never stored.

Run:  python -I -m app.main            (then open http://127.0.0.1:8000)

Endpoints:
  GET  /          the single-page UI (app/static/index.html)
  GET  /health    liveness check + which models are loaded
  GET  /models    loaded models with their test metrics, read from reports/*.json
  POST /predict   multipart form: file=<image>, model=<name>
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.data import load_config
from src.infer import ImageError, load_predictors

ROOT = Path(__file__).resolve().parents[1]   # repo folder; all config paths are relative to it
MAX_BYTES = 5 * 1024 * 1024                  # 5 MB upload limit
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def model_cards(app_cfg: dict, root: Path, loaded: set[str]) -> list[dict]:
    """Facts shown in the UI for each model, read from metric files (never hardcoded)."""
    cards = []
    for m in app_cfg["models"]:
        if m["name"] not in loaded:
            continue
        metrics = _read_json(root / m["metrics"]) or {}
        calib = _read_json(root / m["calibration"]) or {}
        cocoai = _read_json(root / "reports" / f"cocoai_{m['name']}.json") or {}
        cards.append({
            "name": m["name"], "label": m["label"],
            "input_size": metrics.get("input_size"),
            "cifake_test": {k: metrics.get(k) for k in ("accuracy", "f1", "roc_auc", "n")} if metrics else None,
            "cocoai_test": {k: cocoai["overall"].get(k) for k in ("accuracy", "roc_auc", "n")}
                           | {"real_photos_correct": cocoai.get("real_photos_correct")} if cocoai else None,
            "ece_after_calibration": calib.get("test_ece_after"),
        })
    return cards


def create_app(app_config: str | Path = ROOT / "configs" / "app.yaml", root: Path = ROOT) -> FastAPI:
    app_cfg = load_config(app_config)
    predictors = load_predictors(app_config, root)
    app = FastAPI(title="AI-generated image detector", docs_url=None, redoc_url=None)

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "models_loaded": sorted(predictors)}

    @app.get("/models")
    def models():
        return {"models": model_cards(app_cfg, root, set(predictors))}

    @app.post("/predict")
    def predict(file: UploadFile = File(...), model: str = Form("")):
        # A plain `def` endpoint runs in a worker thread, so slow model work
        # does not block other requests.
        if not predictors:
            raise HTTPException(503, "No trained models are available. Train a model first (see README).")
        name = model or next(iter(predictors))
        if name not in predictors:
            raise HTTPException(400, f"Unknown model '{name}'.")
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(415, "Please upload a JPEG, PNG or WebP image.")
        data = file.file.read(MAX_BYTES + 1)  # read at most 5 MB + 1 byte: never more into memory
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "The image is larger than 5 MB. Please upload a smaller file.")
        if not data:
            raise HTTPException(400, "The uploaded file is empty.")
        try:
            out = predictors[name].predict(data)
        except ImageError as exc:   # corrupt file, wrong real format, too many pixels
            raise HTTPException(422, str(exc))
        finally:
            del data                # nothing is written to disk; drop our reference to the bytes
        for key in ("image_png", "heatmap_png"):
            out[key] = base64.b64encode(out[key]).decode("ascii")
        return out

    return app


app = create_app(os.environ.get("APP_CONFIG", ROOT / "configs" / "app.yaml"))

if __name__ == "__main__":
    import uvicorn

    # 127.0.0.1 = this computer only; the app is not reachable from the network.
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
