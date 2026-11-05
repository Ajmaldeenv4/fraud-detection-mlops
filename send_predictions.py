"""Send sample transactions to a running API to generate metrics and drift data."""

import os
import random
import time

import requests

url = os.environ.get("API_URL", "http://localhost:8000") + "/predict"
for i in range(int(os.environ.get("N_REQUESTS", "50"))):
    payload = {
        "Time": random.uniform(0, 172800),
        "Amount": random.uniform(1, 500),
        **{f"V{j}": random.gauss(0, 1) for j in range(1, 29)},
    }
    r = requests.post(url, json=payload, timeout=5)
    r.raise_for_status()
    result = r.json()
    print(f"Request {i + 1}: {result['decision']} ({result['fraud_probability']:.3f})")
    time.sleep(float(os.environ.get("DELAY_SECONDS", "0.3")))
