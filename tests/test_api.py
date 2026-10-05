"""Checks that the web layer is wired to the service layer correctly.
Requires fastapi and httpx (pip install -r requirements.txt)."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import config  # noqa: E402
from app_factory import create_app  # noqa: E402
from helpers import FakePredictor, make_state, new_record, rain  # noqa: E402


def _client(**kwargs):
    return TestClient(create_app(make_state(**kwargs), serve_frontend=False))


def test_health():
    assert _client().get("/health").json() == {"status": "ok"}


def test_status_warns_when_todays_data_is_missing():
    body = _client().get("/status").json()
    assert body["ready"] is False
    assert "Today's data (2026-01-01)" in body["message"]


def test_predict_is_blocked_with_a_clear_message_before_data_is_added():
    client = _client(predictor=FakePredictor())
    response = client.post("/predict", json={"rainfall_forecast": rain(5)})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "DATA_NOT_CURRENT"


def test_add_data_then_predict():
    client = _client(predictor=FakePredictor(delta=0.5))
    saved = client.post("/observations", json=new_record(storage=106.0))
    assert saved.status_code == 200
    assert saved.json()["status"]["ready"] is True
    result = client.post("/predict", json={"target_date": "2026-01-02", "rainfall_forecast": rain(5)})
    assert result.status_code == 200
    assert result.json()["predicted_storage_mcm"] == 106.5


def test_validation_errors_use_the_same_shape():
    client = _client()
    response = client.post("/observations", json=new_record(day="2026-01-03"))
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "DATE_OUT_OF_ORDER"


def test_history_and_observations_endpoints():
    client = _client()
    assert len(client.get("/history?days=10").json()) == 10
    assert len(client.get("/observations?days=5").json()) == 5


def test_pin_header_is_required_when_configured():
    client = _client(pin="1234")
    assert client.post("/observations", json=new_record()).status_code == 401
    ok = client.post("/observations", json=new_record(), headers={"X-Entry-PIN": "1234"})
    assert ok.status_code == 200


def test_frontend_is_served_from_the_same_application():
    if not config.STATIC_DIR.exists():
        pytest.skip("static folder not present")
    client = TestClient(create_app(make_state(), serve_frontend=True))
    assert client.get("/").status_code == 200


def test_chained_prediction_through_the_api():
    client = _client(predictor=FakePredictor(delta=1.5))
    client.post("/observations", json=new_record(day="2026-01-01", storage=100.0, value=4.0))
    predicted = client.post("/predict", json={"rainfall_forecast": rain(5)}).json()
    accepted = client.post("/demo/accept-prediction",
                           json={"storage_mcm": predicted["predicted_storage_mcm"], "rainfall": rain(5)})
    assert accepted.status_code == 200
    assert accepted.json()["status"]["prediction_target"] == "2026-01-03"


def test_accept_prediction_blocked_outside_demo_mode():
    client = TestClient(create_app(make_state(demo=False), serve_frontend=False))
    response = client.post("/demo/accept-prediction", json={"storage_mcm": 100.0, "rainfall": rain(1)})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "NOT_DEMO_MODE"
