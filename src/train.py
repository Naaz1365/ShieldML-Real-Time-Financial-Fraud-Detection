
from pathlib import Path
import json

import pandas as pd
import joblib

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from features import engineer_features


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. GENERATE SYNTHETIC DATASET
# ==========================================

print("Generating synthetic transaction dataset...")

X, y = make_classification(
    n_samples=20000,
    n_features=12,
    n_informative=8,
    n_redundant=2,
    weights=[0.98, 0.02],
    class_sep=1.0,
    random_state=42,
)

columns = [
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

df = pd.DataFrame(X, columns=columns)


# ==========================================
# 3. MAKE DEMO FEATURES MORE REALISTIC
# ==========================================

df["amount"] = (
    df["amount"] * 100 + 500
).clip(lower=1)

df["avg_amount"] = (
    df["avg_amount"] * 100 + 500
).clip(lower=1)

df["hour"] = (
    (df["hour"] * 3 + 12) % 24
).astype(int)

df["transactions_1h"] = (
    df["transactions_1h"] * 3 + 5
).clip(lower=0)

df["transactions_24h"] = (
    df["transactions_24h"] * 10 + 20
).clip(lower=1)

df["is_fraud"] = y


# Save dataset
df.to_csv(
    DATA_DIR / "transactions_demo.csv",
    index=False,
)

print("Dataset generated successfully.")
print("Total transactions:", len(df))


# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================

print("\nApplying feature engineering...")

X = df.drop(columns=["is_fraud"])

X = engineer_features(X)

y = df["is_fraud"]

feature_columns = list(X.columns)

print("Feature count:", len(feature_columns))


# ==========================================
# 5. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 6. RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

print("\nTraining ShieldML Random Forest Model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 7. MODEL PREDICTIONS
# ==========================================

probabilities = model.predict_proba(X_test)[:, 1]

threshold = 0.5



predictions = (
    probabilities >= threshold
).astype(int)


# ==========================================
# 8. MODEL EVALUATION
# ==========================================

print("\n========== MODEL EVALUATION ==========")

accuracy = accuracy_score(
    y_test,
    predictions,
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test,
    probabilities,
)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1],
)

report = classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Normal", "Fraud"],
    zero_division=0,
    output_dict=True,
)


# ==========================================
# 9. DISPLAY EVALUATION RESULTS
# ==========================================

print("\nAccuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Fraud"],
        zero_division=0,
    )
)


# ==========================================
# 10. SAVE TRAINED MODEL
# ==========================================

joblib.dump(
    model,
    MODEL_DIR / "fraud_model.joblib",
)

joblib.dump(
    feature_columns,
    MODEL_DIR / "feature_columns.joblib",
)

print("\nTrained model saved successfully.")


# ==========================================
# 11. SAVE EVALUATION METRICS
# ==========================================

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "threshold": float(threshold),
    "test_samples": int(len(y_test)),
    "feature_count": int(len(feature_columns)),
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "training_data": "Synthetic demo data",
}

with open(
    MODEL_DIR / "metrics.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics,
        file,
        indent=4,
    )


# ==========================================
# 12. FINAL STATUS
# ==========================================

print("\n========== SHIELDML TRAINING SUMMARY ==========")

print("Model          : Random Forest")
print("Training Data  : Synthetic Dataset")
print("Features       :", len(feature_columns))
print("Training Size  :", len(X_train))
print("Testing Size   :", len(X_test))
print("Model File     : models/fraud_model.joblib")
print("Metrics File   : models/metrics.json")

print("\nTraining and evaluation completed successfully!")

print(
    "\nNOTE: This model uses synthetic demo data only. "
    "Its evaluation metrics do not establish real-world fraud detection performance."
)
