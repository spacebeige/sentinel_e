import asyncio
import json
from database.connection_v2 import async_session_factory
from sqlalchemy import text

async def main():
    async with async_session_factory() as session:
        # Get counts
        users = (await session.execute(text("SELECT COUNT(*) FROM users"))).scalar()
        chats = (await session.execute(text("SELECT COUNT(*) FROM chats"))).scalar()
        messages = (await session.execute(text("SELECT COUNT(*) FROM messages"))).scalar()
        settings = (await session.execute(text("SELECT COUNT(*) FROM user_settings"))).scalar()
        
        # Foreign keys check
        orphan_chats = (await session.execute(text("SELECT count(*) FROM chats WHERE user_id NOT IN (SELECT id FROM users)"))).scalar()
        orphan_messages = (await session.execute(text("SELECT count(*) FROM messages WHERE chat_id NOT IN (SELECT id FROM chats)"))).scalar()
        
        # Sample records
        user_record = (await session.execute(text("SELECT id, email, role FROM users LIMIT 1"))).mappings().first()
        chat_record = (await session.execute(text("SELECT id, title, mode, selected_model FROM chats LIMIT 1"))).mappings().first()
        
        results = {
            "counts": {
                "users": users,
                "chats": chats,
                "messages": messages,
                "settings": settings
            },
            "integrity": {
                "orphan_chats": orphan_chats,
                "orphan_messages": orphan_messages
            },
            "samples": {
                "user": dict(user_record) if user_record else None,
                "chat": dict(chat_record) if chat_record else None
            }
        }
        
        print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(main())
