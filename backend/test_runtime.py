import json
import os

os.environ["ENABLE_ROUTING_DEBUG"] = "true"
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "dummy")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "dummy")

from fastapi.testclient import TestClient
from main import app
from gateway.auth_v2 import get_optional_user

def override_get_optional_user():
    return {"id": "test_id", "user_id": "00000000-0000-0000-0000-000000000000", "email": "test@example.com"}

app.dependency_overrides[get_optional_user] = override_get_optional_user



def run_test(client, name, payload):
    print(f"\n{'='*50}\nTEST: {name}\n{'='*50}")
    print(f"Payload: {json.dumps(payload)}")
    response = client.post("/api/mco/run", json=payload)
    if response.status_code == 200:
        data = response.json()
        if "routing_debug" in data:
            print("routing_debug:")
            print(json.dumps(data["routing_debug"], indent=2))
        else:
            print("ERROR: routing_debug not found in response")
            print(json.dumps(data, indent=2)[:500])
    else:
        print(f"ERROR: Status {response.status_code}")
        print(response.text)

def main():
    with TestClient(app) as client:
        # Test 1 - Standard Mode
        run_test(client, "Standard Mode", {
            "query": "hello",
            "mode": "standard"
        })

        # Test 2 - Debate Mode
        run_test(client, "Debate Mode", {
            "query": "Compare AGI timelines from optimistic and skeptical viewpoints.",
            "mode": "experimental",
            "sub_mode": "debate"
        })

        # Test 3 - Glass Mode
        run_test(client, "Glass Mode", {
            "query": "Explain how this system works internally.",
            "mode": "experimental",
            "sub_mode": "glass"
        })

        # Test 4 - Evidence Mode
        run_test(client, "Evidence Mode", {
            "query": "What is the evidence for dark matter?",
            "mode": "experimental",
            "sub_mode": "evidence"
        })

        # Test 5 - Synthesis Mode
        run_test(client, "Synthesis Mode", {
            "query": "Summarize the theories of quantum gravity.",
            "mode": "experimental",
            "sub_mode": "synthesis"
        })

        # Test 6 - Retrieval
        run_test(client, "Retrieval", {
            "query": "When is Monaco GP 2026?",
            "force_retrieval": True
        })

if __name__ == "__main__":
    main()
