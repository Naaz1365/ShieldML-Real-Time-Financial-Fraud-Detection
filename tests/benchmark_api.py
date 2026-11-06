import time
import requests

URL = "http://127.0.0.1:8000/predict"

payload = {
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
    "previous_fraud_flags": 1
}

times = []

for i in range(100):
    start = time.perf_counter()

    response = requests.post(URL, json=payload)
    response.raise_for_status()

    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)

print("Average Response Time:", round(sum(times) / len(times), 2), "ms")
print("Minimum:", round(min(times), 2), "ms")
print("Maximum:", round(max(times), 2), "ms")