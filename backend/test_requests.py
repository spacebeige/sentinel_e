import requests
import json
API_URL = "http://127.0.0.1:8000/api"
HEADERS = {
    "x-debug-user": "cd4ee2f4-7894-4bc3-a9c1-a26c20dbf0d7",
    "x-debug-email": "oomkaragarkhed0710@gmail.com",
    "Content-Type": "application/json"
}
res = requests.get(f"{API_URL}/v2/chat/7e42cb02-28de-4d03-9622-f0c26fce2ec3", headers=HEADERS)
print("Detail:", res.status_code, res.text)
hist = requests.get(f"{API_URL}/v2/chat/history", headers=HEADERS)
print("History:", hist.status_code, hist.text)
