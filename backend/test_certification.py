import asyncio
import os
import json
os.environ["ENVIRONMENT"] = "development"
os.environ["SUPABASE_JWT_SECRET"] = ""
from fastapi.testclient import TestClient
import uuid

def run_tests():
    from main import app
    from database.connection_v2 import engine
    import logging
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    print("=== SCENARIO 1: True Pro Mode ===")
    test_user_id = "52c702e1-abca-4dc2-be37-959dace4fe03"
    test_email = "test1@sentinel.com"
    
    headers = {
        "X-Debug-User": test_user_id,
        "X-Debug-Email": test_email,
        "X-Auth-Provider": "supabase"
    }

    with TestClient(app) as client:
        # Generate run
        payload = {
        "query": "Analyze the future of open-source AI governance.",
        "mode": "analytical",
        "sub_mode": "pro",
        "runtime_preferences": {"depth": "high"}
    }
    
        # We use the MCO run endpoint
        print("Executing Pro Mode...")
        res = client.post("/api/mco/run", json=payload, headers=headers)
        print("Status:", res.status_code)
        
        mco_data = res.json()
        print("MCO DATA:", mco_data)
        metadata = mco_data.get("metadata", {})
        omega = mco_data.get("omega_metadata", {})
        
        print("Debate Executed:", "debate_result" in omega)
        print("Evidence Executed:", "forensic_result" in omega)
        print("Glass Executed:", "audit_result" in omega)
        print("Synthesis Executed:", "synthesis_result" in omega)

        chat_id = mco_data.get("chat_id")
        print(f"\nCreated Chat ID: {chat_id}")
        
        print("\n=== SCENARIO 2: Artifact Persistence ===")
        print("Fetching Chat History...")
        res2 = client.get(f"/api/v2/chat/{chat_id}", headers=headers)
        chat_history = res2.json().get("data", {})
        
        chat_metadata = chat_history.get("metadata", {}).get("machine", {})
        print("Artifacts in chat history:")
        print("  - debate_result:", "debate_result" in chat_metadata)
        print("  - forensic_result:", "forensic_result" in chat_metadata)
        print("  - audit_result:", "audit_result" in chat_metadata)
        print("  - synthesis_result:", "synthesis_result" in chat_metadata)
        print("  - visualizations:", "visualizations" in chat_metadata)
        
        print("\n=== SCENARIO 3: Visualization Persistence ===")
        viz = chat_metadata.get("visualizations", {})
        print("Visualizations dict keys:", list(viz.keys()))
        print("Heatmap generated:", "heatmap_png" in viz)
        print("Conflict graph generated:", "conflict_graph_png" in viz)
        
        print("\n=== SCENARIO 4: History Search ===")
        res_search = client.get(f"/api/v2/user/search?q=open-source", headers=headers)
        search_data = res_search.json().get("data", {})
        print("Search Results Count:", search_data.get("count"))
        if search_data.get("count", 0) > 0:
            match = search_data["results"][0]
            print("Match Content snippet:", match.get("content")[:50])
            print("Match Mode:", match.get("mode"))

        print("\n=== SCENARIO 5: Analytics ===")
        res_analytics = client.get(f"/api/v2/user/analytics", headers=headers)
        analytics_data = res_analytics.json().get("data", {})
        print("Total Sessions:", analytics_data.get("total_sessions"))
        print("Total Messages:", analytics_data.get("total_messages"))
        print("Mode Usage:", analytics_data.get("mode_usage"))
        print("Model Usage:", analytics_data.get("model_usage"))

        print("\n=== SCENARIO 6: End-to-End Recovery ===")
        print("Simulating Logout -> Login -> Open Conversation")
        res_recover = client.get(f"/api/v2/chat/{chat_id}", headers=headers)
        rec_chat = res_recover.json().get("data", {})
        rec_meta = rec_chat.get("metadata", {}).get("machine", {})
        
        print("Messages restored:", len(rec_chat.get("messages", [])) > 0)
        print("Selected Model restored:", rec_meta.get("winning_model") is not None)
        print("Selected Mode restored:", rec_meta.get("mode") is not None)
        print("Artifacts restored:", "debate_result" in rec_meta)
        print("Visualizations restored:", "visualizations" in rec_meta)

if __name__ == "__main__":
    import asyncio
    async def init():
        from database.connection_v2 import init_db
        await init_db()
    asyncio.run(init())
    run_tests()
