from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

res = client.options("/health", headers={
    "Origin": "http://localhost:5174",
    "Access-Control-Request-Method": "GET"
})

print("Status:", res.status_code)
print("Headers:", res.headers)
print("Body:", res.text)
