"""Smoke tests for the API. We do not load a trained model here; we verify that
the API returns 503 when no artifacts are present and that schemas validate."""

from fastapi.testclient import TestClient

from sentinel_stream.serving.api import app


def test_health_endpoint():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_predict_returns_503_without_model(monkeypatch, tmp_path):
    # Force the lifespan to look at an empty artifacts dir.
    monkeypatch.setenv("SENTINEL_ARTIFACTS_DIR", str(tmp_path / "missing"))
    with TestClient(app) as client:
        r = client.post(
            "/predict",
            json={"temperature": 70.0, "pressure": 101.0, "vibration": 0.1, "humidity": 45.0},
        )
        assert r.status_code == 503
