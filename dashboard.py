from pathlib import Path
import json

import requests
import streamlit as st
import pandas as pd


from src.database import (
    init_db,
    save_transaction,
    get_transactions,
    clear_transactions,
)

# INITIALIZE DATABASE
init_db()

API_URL = "http://127.0.0.1:8000"

# PAGE CONFIGURATION
st.set_page_config(
    page_title="ShieldML | Fraud Analytics",
    page_icon="🛡️",
    layout="wide",
)

# HEADER
st.title("🛡️ ShieldML")
st.subheader("Real-Time Financial Fraud Detection System")

st.caption(
    "Machine Learning • Explainable AI • Transaction Analytics"
)

st.info(
    "Educational prototype using synthetic training data. "
    "Risk scores are not verified real-world fraud probabilities. "
    "Do not use this prototype for actual financial decisions."
)

# SIDEBAR
with st.sidebar:
    st.header("Transaction Details")

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=1500.0,
        step=100.0,
    )

    hour = st.slider(
        "Transaction Hour", 0, 23, 2
    )

    account_age = st.number_input(
        "Account Age (days)", min_value=0, value=30
    )

    tx_1h = st.number_input(
        "Transactions in 1 Hour", min_value=0, value=8
    )

    tx_24h = st.number_input(
        "Transactions in 24 Hours", min_value=0, value=20
    )

    avg_amount = st.number_input(
        "Average Transaction Amount",
        min_value=0.0,
        value=500.0,
        step=100.0,
    )

    distance = st.number_input(
        "Distance (km)", min_value=0.0, value=250.0
    )

    failed = st.number_input(
        "Failed Attempts", min_value=0, value=3
    )

    merchant_risk = st.slider(
        "Merchant Risk", 0.0, 1.0, 1.0
    )

    device_change = st.selectbox(
        "New Device",
        [0, 1],
        format_func=lambda x: "Yes" if x else "No",
    )

    international = st.selectbox(
        "International Transaction",
        [0, 1],
        format_func=lambda x: "Yes" if x else "No",
    )

    previous_flags = st.number_input(
        "Previous Fraud Flags", min_value=0, value=1
    )

    analyze = st.button(
        "Analyze Transaction",
        type="primary",
        width="stretch",
    )


# TRANSACTION ANALYSIS
if analyze:

    payload = {
        "amount": amount,
        "hour": hour,
        "account_age_days": account_age,
        "transactions_1h": tx_1h,
        "transactions_24h": tx_24h,
        "avg_amount": avg_amount,
        "distance_km": distance,
        "failed_attempts": failed,
        "merchant_risk": merchant_risk,
        "device_change": device_change,
        "international": international,
        "previous_fraud_flags": previous_flags,
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10,
        )

        response.raise_for_status()
        result = response.json()

        risk_score = float(result["fraud_risk_score"])
        prediction = str(result["prediction"])

        save_transaction(
            amount=amount,
            risk_score=risk_score,
            prediction=prediction,
        )

        st.success("Transaction analyzed and saved successfully!")

        st.header("Transaction Result")

        col1, col2 = st.columns(2)

        col1.metric(
            "Demo Risk Score",
            f"{risk_score:.2f}/100",
        )

        col2.metric(
            "Prediction",
            prediction,
        )

        st.caption(result.get("disclaimer", ""))

        # SHAP EXPLAINABILITY
        st.divider()
        st.header("Explainable AI (SHAP)")
        st.subheader("Why did the model make this prediction?")

        try:
            with st.spinner("Generating model explanation..."):

                explanation_response = requests.post(
                    f"{API_URL}/explain",
                    json=payload,
                    timeout=30,
                )

                explanation_response.raise_for_status()
                explanation = explanation_response.json()

            impacts = pd.DataFrame(
                explanation.get("explanations", [])
            )

            st.caption(explanation.get("note", ""))

            if (
                impacts.empty
                or not {"feature", "impact"}.issubset(
                    impacts.columns
                )
            ):

                st.warning(
                    "No usable feature explanations were returned."
                )

            else:

                impacts["impact"] = pd.to_numeric(
                    impacts["impact"],
                    errors="coerce",
                )

                impacts = impacts.dropna(
                    subset=["impact"]
                ).copy()

                if impacts.empty:

                    st.warning(
                        "No valid SHAP impact values returned."
                    )

                else:

                    impacts["Direction"] = impacts["impact"].apply(
                        lambda value:
                            "Increases model output" if value > 0
                            else "Decreases model output" if value < 0
                            else "No impact"
                    )

                    st.caption(
                        "Positive SHAP values increase model output "
                        "for the fraud class; negative values decrease it. "
                        "These explanations are not proof of fraud."
                    )

                    st.subheader("Top 5 Feature Impacts")

                    top_impacts = impacts.copy()

                    top_impacts["Absolute Impact"] = (
                        top_impacts["impact"].abs()
                    )

                    top_impacts = top_impacts.sort_values(
                        "Absolute Impact",
                        ascending=False,
                    ).head(5)

                    chart_impacts = (
                        top_impacts.sort_values("impact")
                        .set_index("feature")
                    )

                    st.bar_chart(
                        chart_impacts[["impact"]]
                    )

                    st.subheader("Top 5 Feature Impact Details")

                    display_top = top_impacts[
                        ["feature", "impact", "Direction"]
                    ].rename(
                        columns={
                            "feature": "Feature",
                            "impact": "SHAP Impact",
                        }
                    )

                    st.dataframe(
                        display_top,
                        width="stretch",
                        hide_index=True,
                    )

                    with st.expander("View All Feature Impacts"):

                        all_impacts = impacts[
                            ["feature", "impact", "Direction"]
                        ].rename(
                            columns={
                                "feature": "Feature",
                                "impact": "SHAP Impact",
                            }
                        )

                        st.dataframe(
                            all_impacts,
                            width="stretch",
                            hide_index=True,
                        )

        except requests.RequestException as error:

            st.warning(
                "Prediction succeeded, but SHAP explanation failed."
            )

            st.error(str(error))

        except (ValueError, TypeError, KeyError) as error:

            st.warning(
                "SHAP explanation could not be displayed."
            )

            st.error(str(error))

    except requests.RequestException as error:

        st.error(f"Prediction API request failed: {error}")

        st.info(
            "Check that FastAPI is running at http://127.0.0.1:8000."
        )

    except (KeyError, ValueError, TypeError) as error:

        st.error(f"Invalid prediction response: {error}")

    except Exception as error:

        st.error(f"Unexpected error: {error}")


# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.divider()
st.header("Model Performance Evaluation")

METRICS_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "ensemble_metrics.json"
)

if METRICS_PATH.exists():

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    st.caption(
        "Evaluation results from the synthetic test dataset."
    )

    # PERFORMANCE METRICS
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Accuracy",
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{metrics['precision'] * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{metrics['recall'] * 100:.2f}%"
    )

    col4.metric(
        "F1-Score",
        f"{metrics['f1_score'] * 100:.2f}%"
    )

    col5.metric(
        "ROC-AUC",
        f"{metrics['roc_auc'] * 100:.2f}%"
    )

    # CONFUSION MATRIX
    st.subheader("Confusion Matrix")

    cm = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual Normal", "Actual Fraud"],
        columns=["Predicted Normal", "Predicted Fraud"]
    )

    st.dataframe(
        cm,
        width="stretch"
    )

    st.bar_chart(cm)

    # CLASSIFICATION REPORT
    st.subheader("Classification Report")

    report = pd.DataFrame(
        metrics["classification_report"]
    ).T

    st.dataframe(
        report,
        width="stretch"
    )

    # MODEL INFORMATION
    st.subheader("Model Information")

    info_col1, info_col2, info_col3 = st.columns(3)

    info_col1.metric(
        "Test Samples",
        metrics["test_samples"]
    )

    info_col2.metric(
        "Features",
        metrics["feature_count"]
    )

    info_col3.metric(
        "Decision Threshold",
        metrics["threshold"]
    )

    st.info(
        "Model evaluation uses synthetic demo data. "
        "These results do not represent verified real-world "
        "financial fraud detection performance."
    )

else:

    st.warning(
        "Model metrics not found. Run python src/train.py first."
    )

# ANALYTICS OVERVIEW
st.divider()
st.header("Analytics Overview")

history = pd.DataFrame(get_transactions())

if history.empty:

    st.write("No transactions recorded yet.")

    st.caption(
        "Enter transaction details and click Analyze Transaction."
    )

else:

    history = history.rename(
        columns={
            "id": "ID",
            "created_at": "Time",
            "amount": "Amount",
            "risk_score": "Risk Score",
            "prediction": "Prediction",
        }
    )

    # FIX: NORMALIZE PREDICTION LABELS
    history["Prediction"] = (
        history["Prediction"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({
            "LOWER_RISK": "LOW_RISK"
        })
    )

    # TRANSACTION COUNTS
    total = len(history)

    low_risk = int(
        (history["Prediction"] == "LOW_RISK").sum()
    )

    medium_risk = int(
        (history["Prediction"] == "MEDIUM_RISK").sum()
    )

    high_risk = int(
        (history["Prediction"] == "HIGH_RISK").sum()
    )

    average_score = float(
        history["Risk Score"].mean()
    )

    # RISK CLASSIFICATION SUMMARY
    st.subheader("Risk Classification Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Transactions", total)
    col2.metric("Low Risk", low_risk)
    col3.metric("Medium Risk", medium_risk)
    col4.metric("High Risk", high_risk)
    col5.metric(
        "Average Risk Score",
        f"{average_score:.2f}/100"
    )

    # TRANSACTION HISTORY
    st.subheader("Transaction History")

    st.dataframe(
        history,
        width="stretch",
        hide_index=True,
    )

    # RISK SCORE CHART
    st.subheader("Risk Score by Transaction")

    chart_data = history.sort_values("ID").copy()
    chart_data = chart_data.set_index("ID")

    st.line_chart(
        chart_data[["Risk Score"]]
    )

    # PREDICTION DISTRIBUTION
    st.subheader("Prediction Distribution")

    distribution = history["Prediction"].value_counts()

    st.bar_chart(distribution)

    # EXPORT DATA
    st.subheader("Export Data")

    csv_data = history.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Transaction History (CSV)",
        data=csv_data,
        file_name="shieldml_transaction_history.csv",
        mime="text/csv",
        width="stretch",
    )

    # DATABASE MANAGEMENT
    with st.expander("Database Management"):

        st.warning(
            "Deleting history permanently removes all saved "
            "demo transaction records."
        )

        confirm_delete = st.checkbox(
            "I confirm that I want to delete all transaction history."
        )

        if st.button(
            "Delete All History",
            width="stretch"
        ):

            if confirm_delete:

                clear_transactions()

                st.success("Transaction history deleted.")

                st.rerun()

            else:

                st.error(
                    "Please confirm deletion first."
                )


# FOOTER
st.divider()

st.caption(
    "ShieldML | Educational Prototype | Synthetic-Data Model"
)
