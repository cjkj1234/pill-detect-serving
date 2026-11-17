import pytest
from fastapi.testclient import TestClient

from pill_serving import DISCLAIMER
from pill_serving.api import create_app
from pill_serving.config import Settings

from .conftest import CLASS_NAMES


@pytest.fixture
def client(model_dir):
    settings = Settings(model_path=model_dir / "model.onnx", class_mapping_path=model_dir / "class_mapping.json")
    with TestClient(create_app(settings)) as c:  # with 블록이 lifespan(모델 로드)을 실행한다
        yield c


def test_health_reports_model_loaded(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True, "detail": None}


def test_model_info(client):
    info = client.get("/model").json()
    assert info["model_file"] == "model.onnx"
    assert info["input_size"] == 640
    assert info["num_classes"] == 3
    assert info["max_detections"] == 4
    assert info["class_agnostic_nms"] is True


def test_predict_returns_named_detections(client, png_bytes):
    response = client.post("/predict", files={"file": ("pill.png", png_bytes, "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert body["image"] == {"width": 976, "height": 1280}
    assert [d["name"] for d in body["detections"]] == [CLASS_NAMES[0], CLASS_NAMES[2], CLASS_NAMES[0]]
    assert body["detections"][0]["category_id"] == 1000
    assert body["detections"][0]["box"] == pytest.approx({"x1": 61.0, "y1": 80.0, "x2": 91.5, "y2": 120.0})
    assert len(body["detections"][0]["candidates"]) == 3
    assert body["timing"]["total_ms"] >= body["timing"]["inference_ms"]
    assert body["disclaimer"] == DISCLAIMER


def test_predict_rejects_non_image(client):
    response = client.post("/predict", files={"file": ("x.png", b"hello", "image/png")})
    assert response.status_code == 400


def test_predict_rejects_too_large_file(model_dir, png_bytes):
    settings = Settings(
        model_path=model_dir / "model.onnx",
        class_mapping_path=model_dir / "class_mapping.json",
        max_upload_mb=0.001,
    )
    with TestClient(create_app(settings)) as c:
        response = c.post("/predict", files={"file": ("pill.png", png_bytes, "image/png")})
    assert response.status_code == 413


def test_server_starts_without_model_and_explains_why(tmp_path, model_dir, png_bytes):
    settings = Settings(model_path=tmp_path / "missing.onnx", class_mapping_path=model_dir / "class_mapping.json")
    with TestClient(create_app(settings)) as c:
        health = c.get("/health").json()
        predict = c.post("/predict", files={"file": ("pill.png", png_bytes, "image/png")})

    assert health["model_loaded"] is False
    assert "missing.onnx" in health["detail"]
    assert predict.status_code == 503
