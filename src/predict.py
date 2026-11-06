
from pathlib import Path

import joblib
import pandas as pd

from src.features import engineer_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "fraud_ensemble.joblib"
FEATURE_PATH = ROOT / "models" / "ensemble_feature_columns.joblib"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

BASE_FEATURES = [
    "amount",
    "hour",
    "account_age_days",
    "transactions_1h",
    "transactions_24h",
    "avg_amount",
    "distance_km",
    "failed_attempts",
    "merchant_risk",
    "device_change",
    "international",
    "previous_fraud_flags",
]


def predict_transaction(transaction: dict) -> dict:
    """Score a transaction using the ShieldML demo model."""

    data = pd.DataFrame([transaction])

    missing = [col for col in BASE_FEATURES if col not in data.columns]
    if missing:
        raise ValueError(f"Missing transaction features: {missing}")

    data = data[BASE_FEATURES].apply(pd.to_numeric, errors="coerce")

    if data.isnull().any().any():
        raise ValueError("All 12 transaction features must be numeric.")

    data = engineer_features(data)
    data = data.reindex(columns=feature_columns)

    probability = float(model.predict_proba(data)[0, 1])
    risk_score = round(probability * 100, 2)
    
    if risk_score < 30:
        prediction = "LOW_RISK"
    elif risk_score < 66:
        prediction = "MEDIUM_RISK"
    else:
        prediction = "HIGH_RISK"

    return {
        "fraud_risk_score": risk_score,
        "prediction": prediction,
        "disclaimer": (
            "Demo model trained on synthetic data; "
            "not suitable for real financial decisions."
        ),
    }


if __name__ == "__main__":
    sample = {
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

    print(predict_transaction(sample))
