from fastapi.testclient import TestClient
from main import app
import uuid
import traceback

client = TestClient(app)

print("--- TRIGGERING BLOCKER 1 (Chat Persistence) ---")
try:
    headers = {
        "x-debug-user": str(uuid.uuid4()),
        "x-debug-email": "oomkaragarkhed0710@gmail.com",
        "Content-Type": "application/json"
    }
    response = client.post("/api/v2/chat", headers=headers, json={"title": "Test Chat"})
    print("Response Status:", response.status_code)
    print("Response JSON:", response.json())
except Exception as e:
    traceback.print_exc()

print("\n--- TRIGGERING BLOCKER 2 (Debate Mode) ---")
try:
    # Need a valid user id, so we use the one we know works
    valid_headers = {
        "x-debug-user": "cd4ee2f4-7894-4bc3-a9c1-a26c20dbf0d7",
        "x-debug-email": "oomkaragarkhed0710@gmail.com",
        "Content-Type": "application/json"
    }
    response = client.post("/api/mco/run", headers=valid_headers, data={
        "text": "Debate this",
        "mode": "debate",
        "rounds": 3,
        "sub_mode": "default"
    })
    print("Response Status:", response.status_code)
    print("Response Text:", response.text[:200])
except Exception as e:
    traceback.print_exc()

print("\n--- TRIGGERING BLOCKER 3 (Admin Stats) ---")
try:
    # Admin stats might need a valid admin token, but we bypass auth with headers
    response = client.get("/api/admin/system/stats", headers=valid_headers)
    print("Response Status:", response.status_code)
    print("Response Text:", response.text[:200])
except Exception as e:
    traceback.print_exc()
