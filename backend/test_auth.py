import asyncio
import os
import json
import jwt
from fastapi.testclient import TestClient
from dotenv import load_dotenv

load_dotenv("backend/.env", override=True)
secret = os.getenv("SUPABASE_JWT_SECRET", "missing")
print("SECRET LENGTH:", len(secret))
# Supabase anon tokens have no 'iss' usually or have the supabase url as iss
token = jwt.encode({
    "iss": "supabase",
    "sub": "00000000-0000-0000-0000-000000000000",
    "role": "authenticated",
    "email": "test@example.com"
}, secret, algorithm="HS256")

from main import app
client = TestClient(app)
res = client.get("/api/v2/history", headers={"Authorization": f"Bearer {token}"})
print("HISTORY STATUS:", res.status_code)
if res.status_code != 200:
    print("HISTORY ERROR:", res.json())

mco_res = client.post("/api/mco/run", headers={"Authorization": f"Bearer {token}"}, json={"query": "hello"})
print("MCO STATUS:", mco_res.status_code)
print("MCO RESPONSE:", json.dumps(mco_res.json(), indent=2)[:500])
