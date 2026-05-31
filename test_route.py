from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
response = client.post("/api/v2/analytics/events", json={"event_type": "TEST", "event_data": {}})
print("STATUS:", response.status_code)
print("BODY:", response.text)
