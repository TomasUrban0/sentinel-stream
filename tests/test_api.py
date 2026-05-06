"""Smoke tests for the API. We do not load a trained model here; we verify that
the API returns 503 when no artifacts are present and that schemas validate."""

from fastapi.testclient import TestClient

from sentinel_stream.serving.api import app


def _payload() -> dict[str, object]:
    return {
        "type": "L",
        "air_temperature_k": 298.5,
        "process_temperature_k": 308.5,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 100.0,
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


def test_predict_rejects_invalid_type():
    with TestClient(app) as client:
        bad = _payload() | {"type": "X"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422
