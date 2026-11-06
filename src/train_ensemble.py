
from pathlib import Path
import json

import pandas as pd
import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import fbeta_score

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.features import engineer_features


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

df["amount"] = (df["amount"] * 100 + 500).clip(lower=1)
df["avg_amount"] = (df["avg_amount"] * 100 + 500).clip(lower=1)
df["hour"] = ((df["hour"] * 3 + 12) % 24).astype(int)
df["transactions_1h"] = (df["transactions_1h"] * 3 + 5).clip(lower=0)
df["transactions_24h"] = (df["transactions_24h"] * 10 + 20).clip(lower=1)
df["is_fraud"] = y

df.to_csv(DATA_DIR / "transactions_demo.csv", index=False)


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

print("\nApplying feature engineering...")

X = engineer_features(df.drop(columns=["is_fraud"]))
y = df["is_fraud"]

feature_columns = list(X.columns)

print("Feature count:", len(feature_columns))


# ==========================================
# 4. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

fraud_count = int((y_train == 1).sum())
normal_count = int((y_train == 0).sum())

scale_weight = normal_count / fraud_count

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Fraud class weight:", round(scale_weight, 2))


# ==========================================
# 5. DEFINE THREE MODELS
# ==========================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

lgbm_model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    class_weight="balanced",
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1,
    n_jobs=-1,
)


# ==========================================
# 6. SOFT VOTING ENSEMBLE
# ==========================================

ensemble_model = VotingClassifier(
    estimators=[
        ("random_forest", rf_model),
        ("xgboost", xgb_model),
        ("lightgbm", lgbm_model),
    ],
    voting="soft",
    n_jobs=1,
)

print("\nSplitting training data for validation...")

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train,
)

print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_fit_resampled, y_fit_resampled = smote.fit_resample(
    X_fit,
    y_fit,
)

print("Training ensemble for threshold optimization...")

ensemble_model.fit(X_fit_resampled, y_fit_resampled)

val_probabilities = ensemble_model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.05, 0.96, 0.01)

scores = [
    fbeta_score(
        y_val,
        (val_probabilities >= t).astype(int),
        beta=2,
        zero_division=0,
    )
    for t in thresholds
]

best_threshold = float(thresholds[np.argmax(scores)])

print("Best F2 Threshold:", round(best_threshold, 3))
print("Validation F2 Score:", round(max(scores), 4))

print("\nRetraining final ensemble...")

X_train_resampled, y_train_resampled = SMOTE(
    random_state=42
).fit_resample(X_train, y_train)

ensemble_model.fit(X_train_resampled, y_train_resampled)

print("Final ensemble training completed!")


# ==========================================
# 7. MODEL EVALUATION
# ==========================================
def evaluate_model(name, model, threshold=0.5):

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Fraud"],
            zero_division=0,
            output_dict=True,
        ),
    }

    print(f"\n========== {name} ==========")
    print("Accuracy :", round(metrics["accuracy"], 4))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall   :", round(metrics["recall"], 4))
    print("F1-Score :", round(metrics["f1_score"], 4))
    print("ROC-AUC  :", round(metrics["roc_auc"], 4))

    return metrics


# ==========================================
# 8. EVALUATE ALL MODELS
# ==========================================

print("\nEvaluating individual models and ensemble...")

model_results = {}

for name, fitted_model in ensemble_model.named_estimators_.items():

    model_results[name] = evaluate_model(
        name.upper(),
        fitted_model,
    )

model_results["soft_voting_ensemble"] = evaluate_model(
    "SOFT VOTING ENSEMBLE",
    ensemble_model,
    threshold=best_threshold,
)



# ==========================================
# 9. SAVE MODELS
# ==========================================

joblib.dump(
    ensemble_model,
    MODEL_DIR / "fraud_ensemble.joblib",
)

for name, fitted_model in ensemble_model.named_estimators_.items():

    joblib.dump(
        fitted_model,
        MODEL_DIR / f"{name}_model.joblib",
    )

joblib.dump(
    feature_columns,
    MODEL_DIR / "ensemble_feature_columns.joblib",
)


# ==========================================
# 10. SAVE ENSEMBLE METRICS
# ==========================================

ensemble_metrics = dict(model_results["soft_voting_ensemble"])

ensemble_metrics.update({
    "model_name": "Soft Voting Ensemble",
    "individual_models": model_results,
    "test_samples": int(len(y_test)),
    "feature_count": len(feature_columns),
    "training_data": "Synthetic demo data",
})

with open(
    MODEL_DIR / "ensemble_metrics.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(ensemble_metrics, file, indent=4)


# ==========================================
# 11. FINAL STATUS
# ==========================================

print("\n========== SHIELDML ENSEMBLE SUMMARY ==========")

print("Models: Random Forest + XGBoost + LightGBM")
print("Ensemble: Soft Voting")
print("Features:", len(feature_columns))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nEnsemble saved:")
print("models/fraud_ensemble.joblib")

print("\nMetrics saved:")
print("models/ensemble_metrics.json")

print("\nExisting fraud_model.joblib has NOT been modified.")

print("\nNOTE: This is an educational prototype trained on")
print("synthetic data. It is not validated for real financial decisions.")
