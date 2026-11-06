
from pathlib import Path

import joblib
import pandas as pd
import shap

from src.features import engineer_features

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "fraud_ensemble.joblib"
FEATURES_PATH = ROOT / "models" / "ensemble_feature_columns.joblib"
DATA_PATH = ROOT / "data" / "transactions_demo.csv"

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


def explain_transaction(transaction: dict) -> dict:

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    input_df = pd.DataFrame([transaction])

    missing = [
        col for col in BASE_FEATURES
        if col not in input_df.columns
    ]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    input_df = input_df[BASE_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    if input_df.isnull().any().any():
        raise ValueError("All transaction features must be numeric.")

    input_df = engineer_features(input_df)
    input_df = input_df.reindex(columns=feature_columns)

    background_df = pd.read_csv(DATA_PATH)
    background_df = background_df[BASE_FEATURES]
    background_df = engineer_features(background_df)
    background_df = background_df.reindex(columns=feature_columns)

    background_df = background_df.sample(
        n=min(50, len(background_df)),
        random_state=42,
    )

    explainer = shap.Explainer(
        model.predict_proba,
        background_df,
        algorithm="permutation",
    )

    shap_values = explainer(
        input_df,
        max_evals=2 * len(feature_columns) + 1,
    )

    values = shap_values.values

    if values.ndim == 3:
        values = values[0, :, 1]
    elif values.ndim == 2:
        values = values[0]

    impacts = [
        {
            "feature": feature,
            "impact": round(float(value), 6),
        }
        for feature, value in zip(feature_columns, values)
    ]

    impacts.sort(
        key=lambda item: abs(item["impact"]),
        reverse=True,
    )

    return {
        "explanations": impacts,
        "note": (
            "SHAP values explain the ensemble model's output "
            "relative to background data; they do not prove fraud."
        ),
    }
