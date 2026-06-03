import requests
import json
import uuid
API_URL = "http://127.0.0.1:8000/api"
HEADERS = {
    "x-debug-user": "cd4ee2f4-7894-4bc3-a9c1-a26c20dbf0d7",
    "x-debug-email": "oomkaragarkhed0710@gmail.com",
    "Content-Type": "application/json"
}

# Add a message to the chat
chat_id = "7e42cb02-28de-4d03-9622-f0c26fce2ec3"
msg_payload = {"role": "user", "content": "Hello world!"}
res_msg = requests.post(f"{API_URL}/v2/chat/{chat_id}/message", headers=HEADERS, params={"role": "user", "content": "Hello world!"})
print("Added message:", res_msg.status_code, res_msg.text)

# Try get history
hist = requests.get(f"{API_URL}/v2/history", headers=HEADERS)
print("History:", hist.status_code, hist.text[:200])

