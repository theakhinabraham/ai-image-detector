import base64

import pytest
from fastapi.testclient import TestClient

from app.main import MAX_BYTES, create_app
from tests.conftest import image_bytes


@pytest.fixture
def client(app_root):
    return TestClient(create_app(app_root / "app.yaml", app_root))


def test_health_and_models(client):
    assert client.get("/health").json() == {"status": "ok", "models_loaded": ["testcnn"]}
    models = client.get("/models").json()["models"]
    assert models[0]["name"] == "testcnn" and models[0]["cifake_test"] is None  # no metrics file -> nothing invented


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "not proof" in r.text


def test_predict_valid_image(client):
    r = client.post("/predict", files={"file": ("x.png", image_bytes("PNG"), "image/png")},
                    data={"model": "testcnn"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"model", "probability_fake", "verdict", "heatmap_png", "latency_ms", "warnings"}
    assert body["verdict"] in {"likely real", "uncertain", "likely AI-generated"}
    assert base64.b64decode(body["heatmap_png"])[:8] == b"\x89PNG\r\n\x1a\n"
    assert any("limited set of generators" in w for w in body["warnings"])


def test_predict_wrong_type(client):
    r = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_predict_oversize(client):
    r = client.post("/predict", files={"file": ("big.jpg", b"\xff" * (MAX_BYTES + 1), "image/jpeg")})
    assert r.status_code == 413


def test_predict_corrupt(client):
    r = client.post("/predict", files={"file": ("bad.png", b"\x89PNG garbage", "image/png")})
    assert r.status_code == 422 and "corrupt" in r.json()["detail"]


def test_predict_disguised_gif(client):
    r = client.post("/predict", files={"file": ("x.png", image_bytes("GIF"), "image/png")})
    assert r.status_code == 422


def test_unknown_model(client):
    r = client.post("/predict", files={"file": ("x.png", image_bytes("PNG"), "image/png")},
                    data={"model": "nope"})
    assert r.status_code == 400
