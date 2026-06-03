import json
import asyncio
from fastapi.testclient import TestClient
from main import app
from database.session import SessionLocal
from database.models import User, Chat, ChatMessage, UserSettings, UserProfile
from sqlalchemy import text

client = TestClient(app)

def run_audit():
    results = {}
    
    # DB Audit
    print("Running DB Audit...")
    db = SessionLocal()
    try:
        users = db.query(User).count()
        chats = db.query(Chat).count()
        messages = db.query(ChatMessage).count()
        settings = db.query(UserSettings).count()
        
        # Test foreign keys
        orphan_chats = db.execute(text("SELECT count(*) FROM chats WHERE user_id NOT IN (SELECT id FROM users)")).scalar()
        orphan_msgs = db.execute(text("SELECT count(*) FROM chat_messages WHERE chat_id NOT IN (SELECT id FROM chats)")).scalar()
        
        results['database'] = {
            'users_count': users,
            'chats_count': chats,
            'messages_count': messages,
            'settings_count': settings,
            'orphan_chats': orphan_chats,
            'orphan_messages': orphan_msgs
        }
        
        # Get a test user
        user = db.query(User).first()
        if user:
            # Generate a mock token for this user
            from utils.auth import create_access_token
            from datetime import timedelta
            token = create_access_token(data={"sub": str(user.id)}, expires_delta=timedelta(hours=1))
            headers = {"Authorization": f"Bearer {token}"}
            
            # 1. Profile Validation
            print("Running Profile Validation...")
            prof_res = client.get("/api/v2/user/profile", headers=headers)
            results['profile'] = prof_res.json()
            
            # 2. Settings Validation
            print("Running Settings Validation...")
            set_res = client.get("/api/v2/user/settings", headers=headers)
            results['settings'] = set_res.json()
            
            # 3. Chat History
            print("Running Chat History...")
            hist_res = client.get("/api/v2/chat/history", headers=headers)
            hist_data = hist_res.json()
            results['chat_history'] = hist_data
            
            # 4. Chat Restoration
            print("Running Chat Restoration...")
            if hist_data.get('chats') and len(hist_data['chats']) > 0:
                chat_id = hist_data['chats'][0]['id']
                chat_res = client.get(f"/api/v2/chat/{chat_id}", headers=headers)
                results['chat_restore'] = chat_res.json()
            
            # 5. Admin System Stats
            print("Running Admin System Stats...")
            admin_res = client.get("/api/admin/system/stats", headers=headers)
            results['admin_stats'] = {'status': admin_res.status_code, 'data': admin_res.json()}
            
    finally:
        db.close()
        
    with open("audit_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
        
if __name__ == "__main__":
    run_audit()
    print("Audit Complete.")
