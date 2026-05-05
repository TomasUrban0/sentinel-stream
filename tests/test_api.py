"""Smoke tests for the API. We do not load a trained model here; we verify that
the API returns 503 when no artifacts are present and that schemas validate."""

from fastapi.testclient import TestClient

from sentinel_stream.serving.api import app


def _payload(value: float = 1.0) -> dict[str, float]:
    return {
        "accelerometer_1_rms": value,
        "accelerometer_2_rms": value,
        "current": value,
        "pressure": value,
        "temperature": value,
        "thermocouple": value,
        "voltage": value,
        "volume_flow_rate": value,
    }


def test_health_endpoint():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_predict_returns_503_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("SENTINEL_ARTIFACTS_DIR", str(tmp_path / "missing"))
    with TestClient(app) as client:
        r = client.post("/predict", json=_payload())
        assert r.status_code == 503
