import requests
import json
import os
from datetime import datetime

API_URL = "http://127.0.0.1:8000"

def run_api_tests():
    print("Running API Validation Tests...")
    results = {}
    
    # Check Health
    try:
        health = requests.get(f"{API_URL}/api/health")
        print(f"Health check: {health.status_code}")
        results["health"] = health.status_code
    except Exception as e:
        print(f"Server not running: {e}")
        return
        
    # Since we don't have a supabase JWT, we can't test authenticated routes easily without a mock token or real token.
    # But wait, Sentinel-E might allow local dev bypass or we can generate a JWT using the backend's secret if we know it.
    
    # 1. Test Admin Stats
    # Even if it returns 401/403, we capture the payload.
    admin = requests.get(f"{API_URL}/api/admin/system/stats")
    results["admin_stats"] = {"status": admin.status_code, "body": admin.text}
    print(f"Admin Stats: {admin.status_code}")
    
    with open("api_validation.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_api_tests()
