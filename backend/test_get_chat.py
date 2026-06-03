from fastapi.testclient import TestClient
from main import app
import traceback

client = TestClient(app)

print("\n--- FETCHING CHAT DETAIL ---")
try:
    valid_headers = {
        "x-debug-user": "cd4ee2f4-7894-4bc3-a9c1-a26c20dbf0d7",
        "x-debug-email": "oomkaragarkhed0710@gmail.com",
        "Content-Type": "application/json"
    }
    # Using the chat_id from the user's test
    chat_id = "7e42cb02-28de-4d03-9622-f0c26fce2ec3"
    response = client.get(f"/api/v2/chat/{chat_id}", headers=valid_headers)
    print("Response Status:", response.status_code)
    print("Response Text:", response.text)
except Exception as e:
    traceback.print_exc()

