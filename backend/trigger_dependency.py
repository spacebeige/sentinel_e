import requests
import uuid

API_URL = "http://127.0.0.1:8000/api"
HEADERS = {
    "x-debug-user": str(uuid.uuid4()),
    "x-debug-email": "oomkaragarkhed0710@gmail.com",
    "Content-Type": "application/json"
}

res = requests.post(f"{API_URL}/v2/chat", headers=HEADERS, json={"title": "Test Chat"})
print("Create Chat Response:", res.status_code, res.text)
