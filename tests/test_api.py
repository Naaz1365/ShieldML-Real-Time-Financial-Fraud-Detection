
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

SAMPLE_TRANSACTION = {
    "amount": 1500,
    "hour": 2,
    "account_age_days": 30,
    "transactions_1h": 8,
    "transactions_24h": 20,
    "avg_amount": 500,
    "distance_km": 250,
    "failed_attempts": 3,
    "merchant_risk": 1,
    "device_change": 1,
    "international": 1,
    "previous_fraud_flags": 1,
}


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_endpoint():
    response = client.post(
        "/predict",
        json=SAMPLE_TRANSACTION,
    )

    assert response.status_code == 200

    result = response.json()

    assert "fraud_risk_score" in result
    assert "prediction" in result
    assert 0 <= result["fraud_risk_score"] <= 100
    assert result["prediction"] in [
        "HIGH_RISK",
        "LOWER_RISK",
    ]


def test_explain_endpoint():
    response = client.post(
        "/explain",
        json=SAMPLE_TRANSACTION,
    )

    assert response.status_code == 200

    result = response.json()

    assert "explanations" in result
    assert "note" in result
    assert len(result["explanations"]) > 0


def test_invalid_transaction():
    response = client.post(
        "/predict",
        json={"amount": 1500},
    )

    assert response.status_code == 422
