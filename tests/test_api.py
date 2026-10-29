"""HTTP contract: health, a real prediction, and the two validation failures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from greener.api import create_app

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "sample_request.json"


@pytest.fixture
def client(artifact_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_PATH", str(artifact_path))
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_health_without_a_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    app = create_app()
    with TestClient(app) as test_client:
        response = test_client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert "make train" in body["detail"]


def test_health_reports_the_loaded_model(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_features"] == 364
    assert body["n_components"] == 36
    assert body["train_r2"] > 0.9


def test_sample_request_returns_a_bench_time(client: TestClient, artifact) -> None:
    sample = json.loads(SAMPLE.read_text())
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    body = response.json()
    direct = artifact.predict_record(sample)
    assert body["ID"] == sample["ID"]
    assert body["predicted_seconds"] == pytest.approx(direct.predicted_seconds)
    assert body["unknown_categoricals"] == direct.unknown_categoricals
    assert 50 < body["predicted_seconds"] < 200
    # The first public test row uses X5 = "t", a level the training file never saw.
    assert "X5" in body["unknown_categoricals"]


def test_missing_feature_is_422(client: TestClient) -> None:
    response = client.post("/predict", json={"ID": 1, "X0": "k"})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "X10" in detail["missing"]
    assert "X0" not in detail["missing"]


def test_non_binary_option_is_422(client: TestClient) -> None:
    sample = json.loads(SAMPLE.read_text())
    sample["X10"] = 5
    response = client.post("/predict", json=sample)
    assert response.status_code == 422
    assert response.json()["detail"]["column"] == "X10"


def test_schema_describes_the_loaded_model(client: TestClient) -> None:
    response = client.get("/schema")
    assert response.status_code == 200
    body = response.json()
    assert body["categorical_columns"] == ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]
    assert "k" in body["categorical_levels"]["X0"]
    assert "X10" in body["binary_columns"]
    assert "X11" not in body["binary_columns"]
    assert "X11" in body["dropped_columns"]
    assert body["n_features"] == 364
    assert body["n_components"] == 36
    assert 90 < body["y_mean"] < 110
    assert body["y_std"] > 0


def test_schema_without_a_model_is_503(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    app = create_app()
    with TestClient(app) as test_client:
        response = test_client.get("/schema")
    assert response.status_code == 503
    assert "make train" in response.json()["detail"]


def test_predict_without_a_model_is_503(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    app = create_app()
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json={"ID": 1})
    assert response.status_code == 503
