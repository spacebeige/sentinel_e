from dotenv import load_dotenv
load_dotenv()
import os, jwt, requests

secret = os.environ.get("SUPABASE_JWT_SECRET")
token = jwt.encode({
    "iss": "supabase",
    "sub": "845d7582-aab0-4bc4-8d46-3811f814bb26",
    "role": "authenticated",
    "aud": "authenticated",
    "email": "test@example.com"
}, secret, algorithm="HS256")

headers = {"Authorization": "Bearer " + token}

health = requests.get("https://sentinel-e-evo.onrender.com/health").json()
print("HEALTH:", health.get("status"), "version=" + str(health.get("version")))

hist = requests.get("https://sentinel-e-evo.onrender.com/api/v2/history", headers=headers)
print("HISTORY:", hist.status_code)

mco = requests.post("https://sentinel-e-evo.onrender.com/api/mco/run", headers=headers,
                    json={"query": "hello", "selected_model": "llama31-8b"})
print("MCO:", mco.status_code)

data = mco.json()
answer = data.get("aggregated_answer") or data.get("formatted_output") or (data.get("data") or {}).get("priority_answer", "")
is_fallback = data.get("fallback", False)
print("FALLBACK=" + str(is_fallback))
print("ANSWER_LEN=" + str(len(answer)))
print("ANSWER_PREVIEW=" + answer[:150])
