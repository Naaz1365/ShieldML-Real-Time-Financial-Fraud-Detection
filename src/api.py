
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import predict_transaction
from src.explain import explain_transaction

app = FastAPI(
    title="ShieldML Fraud Detection API",
    description="Demo API for transaction risk scoring",
    version="1.0.0",
)


class Transaction(BaseModel):
    amount: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    account_age_days: float = Field(ge=0)
    transactions_1h: float = Field(ge=0)
    transactions_24h: float = Field(ge=0) 
    avg_amount: float = Field(ge=0)
    distance_km: float = Field(ge=0)
    failed_attempts: float = Field(ge=0)
    merchant_risk: float = Field(ge=0, le=1)
    device_change: int = Field(ge=0, le=1)
    international: int = Field(ge=0, le=1)
    previous_fraud_flags: float = Field(ge=0)


@app.get("/")
def home():
    return {
        "project": "ShieldML",
        "status": "running",
        "message": "Financial Fraud Detection Demo API",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(transaction: Transaction):
    try:
        return predict_transaction(transaction.model_dump())
    except (ValueError, TypeError) as error:
        raise HTTPException(
            status_code=400, detail=str(error)
        ) from error

@app.post("/explain")
def explain(payload: Transaction):
    try:
        transaction = payload.model_dump()
        return explain_transaction(transaction)
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=400, detail=str(error))