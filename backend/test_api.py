import asyncio
import os
os.environ["ENVIRONMENT"] = "development"
os.environ["SUPABASE_JWT_SECRET"] = ""
from fastapi.testclient import TestClient
import uuid

def run_tests():
    from main import app
    from database.connection_v2 import engine
    from database.models_v2 import Base
    import logging
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    client = TestClient(app)

    print("--- TEST 1: CONVERSATION CREATION ---")
    test_user_id = "52c702e1-abca-4dc2-be37-959dace4fe03"
    test_email = "test1@sentinel.com"
    
    headers = {
        "X-Debug-User": test_user_id,
        "X-Debug-Email": test_email,
        "X-Auth-Provider": "supabase"
    }

    # Create session
    sess_res = client.post("/api/v2/sessions", json={"client": "web", "ip_address": "127.0.0.1"}, headers=headers)
    print("Session created:", sess_res.status_code, sess_res.json())

    # Create conversation
    res = client.post("/api/v2/conversations", json={
        "title": "Test Chat",
        "mode": "conversational",
        "engine": "claude-3"
    }, headers=headers)
    print("Create Chat:", res.status_code, res.json())
    chat_data = res.json()["data"]
    chat_id = chat_data["id"]

    # Verify conversation retrievable
    res2 = client.get("/api/v2/conversations", headers=headers)
    print("List Chats (Test 1):", len(res2.json()["data"]), "chats found")
    
    print("\n--- TEST 2: MESSAGE PERSISTENCE ---")
    # Send 25 messages
    for i in range(25):
        msg_res = client.post("/api/v2/messages", json={
            "conversation_id": chat_id,
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Test message {i}"
        }, headers=headers)
        if msg_res.status_code != 201:
            print("Failed to add message", i, msg_res.json())

    # Retrieve messages
    msgs_res = client.get(f"/api/v2/messages?conversation_id={chat_id}", headers=headers)
    messages = msgs_res.json()["data"]
    print("Retrieved messages count:", len(messages))

    print("\n--- TEST 3: DUPLICATE STORAGE AUDIT ---")
    # Ensure exact 25 messages exist
    print("Duplicate check pass:", len(messages) == 25)

    print("\n--- TEST 4: LOGOUT RECOVERY (Simulate re-auth) ---")
    res_recover = client.get("/api/v2/conversations", headers=headers)
    print("Recovered chats:", len(res_recover.json()["data"]))
    
    print("\n--- TEST 5: CONVERSATION RENAME ---")
    patch_res = client.patch(f"/api/v2/conversations/{chat_id}", json={"title": "Renamed Chat"}, headers=headers)
    print("Rename status:", patch_res.status_code, patch_res.json()["data"]["title"])

    print("\n--- TEST 6 & 7: CONVERSATION ARCHIVE/DELETE ---")
    del_res = client.delete(f"/api/v2/conversations/{chat_id}", headers=headers)
    print("Delete status:", del_res.status_code)
    
    list_res = client.get("/api/v2/conversations", headers=headers)
    print("Visible chats after delete:", len(list_res.json()["data"]))

    print("\n--- TEST 8: USER ISOLATION ---")
    test_user_id_2 = "b4e8d350-f966-4a9f-863a-cc5eb1f86820"
    headers2 = {
        "X-Debug-User": test_user_id_2,
        "X-Debug-Email": "test2@sentinel.com",
        "X-Auth-Provider": "supabase"
    }
    
    iso_res = client.get("/api/v2/conversations", headers=headers2)
    print("User 2 chats:", len(iso_res.json()["data"]))
    
    iso_msg = client.get(f"/api/v2/messages?conversation_id={chat_id}", headers=headers2)
    print("User 2 fetch user 1 chat:", iso_msg.status_code, iso_msg.json())

    print("\nTests complete.")

if __name__ == "__main__":
    # Ensure db initialization first
    import asyncio
    async def init():
        from database.connection_v2 import init_db
        await init_db()
    asyncio.run(init())
    run_tests()
