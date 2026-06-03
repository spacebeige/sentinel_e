from fastapi.testclient import TestClient
from main import app
import traceback

client = TestClient(app)

print("\n--- FETCHING /api/history ---")
try:
    valid_headers = {
        "x-debug-user": "cd4ee2f4-7894-4bc3-a9c1-a26c20dbf0d7",
        "x-debug-email": "oomkaragarkhed0710@gmail.com",
        "Content-Type": "application/json"
    }
    response = client.get("/api/history", headers=valid_headers)
    print("Response Status:", response.status_code)
    print("Response Text:", response.text)
except Exception as e:
    traceback.print_exc()

