import requests, random, time

url = "http://localhost:8000/predict"
for i in range(50):
    payload = {
        "Time": random.uniform(0, 172800),
        "Amount": random.uniform(1, 500),
        **{f"V{j}": random.gauss(0, 1) for j in range(1, 29)}
    }
    r = requests.post(url, json=payload)
    print(f'Request {i+1}: {r.json()["decision"]} ({r.json()["fraud_probability"]:.3f})')
    time.sleep(0.3)
