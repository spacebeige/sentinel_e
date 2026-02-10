import requests
import os
from dotenv import load_dotenv

load_dotenv("backend/.env")

def test_requests():
    key = os.getenv("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    try:
        print("Sending request via requests...")
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        print(f"Status: {resp.status_code}")
        print(f"Body: {resp.text[:100]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_requests()
