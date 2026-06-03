import asyncio
from database.connection_v2 import async_session_factory
from database.crud import create_chat
from uuid import uuid4
import traceback
import logging

logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

async def main():
    async with async_session_factory() as db:
        try:
            # We pass a string user_id that is NOT a valid UUID to trigger DataError 
            user_id_str = "not-a-valid-uuid"
            chat = await create_chat(db, chat_name="Test Chat", mode="conversational", user_id=user_id_str)
            print("Chat created successfully:", chat.id)
        except Exception as e:
            print("=== STACK TRACE BLOCKER 1 ===")
            traceback.print_exc()

asyncio.run(main())
