from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid():
    payload = {
        "age_years": 4.0,
        "km_driven": 25000,
        "engine_cc": 150,
        "owners": 1,
        "service_records": 5,
        "accidents": 0,
        "brand_tier": "mid",
        "city_zone": "Central",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)


def test_predict_rejects_age_over_30():
    payload = {
        "age_years": 31.0,
        "km_driven": 25000,
        "engine_cc": 150,
        "owners": 1,
        "service_records": 5,
        "accidents": 0,
        "brand_tier": "mid",
        "city_zone": "Central",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_negative_service_records():
    payload = {
        "age_years": 4.0,
        "km_driven": 25000,
        "engine_cc": 150,
        "owners": 1,
        "service_records": -1,
        "accidents": 0,
        "brand_tier": "mid",
        "city_zone": "Central",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422