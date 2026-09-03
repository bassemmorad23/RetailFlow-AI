"""Quick test: fire 25 requests, see when rate limit kicks in."""
import requests

url = "http://localhost:8000/chat"
payload = {
    "conversation_id": "rt_test",
    "customer_id": "rt_test",
    "store_id": "store_002",
    "text": "hi",
    "channel": "web",
}

for i in range(1, 10):
    try:
        r = requests.post(url, json=payload, timeout=150)
        print(f"Request {i:2d}: {r.status_code}")
    except Exception as e:
        print(f"Request {i:2d}: ERROR {e}")