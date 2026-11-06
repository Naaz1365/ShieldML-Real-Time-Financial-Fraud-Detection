# ShieldML — Real-Time Financial Fraud Detection System

ShieldML is a machine learning-based prototype designed to analyze financial transaction data and estimate fraud risk in real time.

## Features

- Real-time transaction risk scoring
- Fraud risk classification: Low, Medium, High
- Ensemble machine learning model
- Random Forest, XGBoost and LightGBM
- SMOTE for handling class imbalance
- SHAP-based model explanations
- FastAPI prediction API
- Streamlit interactive dashboard
- Transaction history and analytics
- API response-time benchmarking

## Technologies Used

- Python
- Scikit-learn
- XGBoost
- LightGBM
- Imbalanced-learn
- SHAP
- FastAPI
- Streamlit
- Pandas
- NumPy
- Joblib

## Project Structure

```text
ShieldML/
├── data/
├── models/
├── src/
│   ├── api.py
│   ├── predict.py
│   ├── features.py
│   └── train_ensemble.py
├── tests/
│   └── benchmark_api.py
├── dashboard.py
└── README.md
```

## How to Run

### 1. Activate Virtual Environment

```bash
.venv\Scripts\activate
```

### 2. Start FastAPI

```bash
python -m uvicorn src.api:app --reload
```

API URL: `http://127.0.0.1:8000`

API Documentation: `http://127.0.0.1:8000/docs`

### 3. Start Streamlit Dashboard

Open another terminal and run:

```bash
python -m streamlit run dashboard.py
```

Dashboard URL: `http://localhost:8501`

## Model Performance

| Metric | Result |
|---|---:|
| Accuracy | 97.62% |
| Precision | 53.12% |
| Recall | 50.50% |
| F1-Score | 51.78% |
| ROC-AUC | 84.65% |
| Decision Threshold | 0.66 |

## API Performance Benchmark

The API was tested using 100 consecutive requests to the `/predict` endpoint.

| Metric | Result |
|---|---:|
| Total Requests | 100 |
| Average Response Time | 49.11 ms |
| Minimum Response Time | 44.69 ms |
| Maximum Response Time | 131.11 ms |
| Target Response Time | Below 50 ms |

The average response time achieved the target of less than 50 ms in this local benchmark.

## Disclaimer

ShieldML is an educational prototype trained and tested using synthetic transaction data. Its results are not validated for real-world financial decisions or production fraud detection.