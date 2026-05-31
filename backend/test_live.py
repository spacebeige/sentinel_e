import requests
import time
import subprocess
import os
import signal
import jwt
from dotenv import load_dotenv

load_dotenv(".env", override=True)
secret = os.getenv("SUPABASE_JWT_SECRET", "12345678901234567890123456789012") # fallback so it doesn't crash on boot

token = jwt.encode({
    "iss": "supabase",
    "sub": "00000000-0000-0000-0000-000000000000",
    "role": "authenticated",
    "aud": "authenticated",
    "email": "test@example.com"
}, secret, algorithm="HS256")

# Start server
p = subprocess.Popen(["/Users/ashwinagarkhed/sentinel_e/.venv/bin/python", "-m", "uvicorn", "main:app", "--port", "8002"], cwd="/Users/ashwinagarkhed/sentinel_e/backend")
time.sleep(5) # wait for startup

try:
    res = requests.get("http://127.0.0.1:8002/api/v2/history", headers={"Authorization": f"Bearer {token}"})
    print("HISTORY STATUS:", res.status_code)
    print("HISTORY BODY:", res.text)
    
    mco_res = requests.post("http://127.0.0.1:8002/api/mco/run", headers={"Authorization": f"Bearer {token}"}, json={"query": "hello", "selected_model": "llama31-8b"})
    print("MCO STATUS:", mco_res.status_code)
    print("MCO BODY:", mco_res.text)
except Exception as e:
    print("ERROR:", e)
finally:
    os.kill(p.pid, signal.SIGTERM)
