import requests
import json
import uuid

API_URL = "http://127.0.0.1:8000/api"
HEADERS = {
    "x-debug-user": str(uuid.uuid4()),
    "x-debug-email": "oomkaragarkhed0710@gmail.com",
    "Content-Type": "application/json"
}

print("1. Creating chat...")
res = requests.post(f"{API_URL}/v2/chat", headers=HEADERS, json={"title": "Test Chat"})
print("Create Chat Response:", res.status_code, res.text)

if res.status_code == 200:
    chat_id = res.json()["data"]["id"]
    
    print("\n2. Restoring chat...")
    res_restore = requests.get(f"{API_URL}/v2/chat/{chat_id}", headers=HEADERS)
    print("Restore Chat Response:", res_restore.status_code, res_restore.text)
    
    print("\n3. Fetching history (with typo)...")
    hist = requests.get(f"{API_URL}/v2/chat/history", headers=HEADERS)
    print("History Response:", hist.status_code, hist.text)
