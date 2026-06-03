import asyncio
from database.connection_v2 import async_session_factory
from database.crud import create_chat
from uuid import uuid4
import traceback
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

async def main():
    async with async_session_factory() as db:
        try:
            # We pass a string user_id (not a UUID object) to trigger DataError 
            # or 'invalid input for query argument' in asyncpg if no cast is done
            user_id_str = "cd4ee2f4-7894-4bc3-a9c1-a26c20dbf0d7"
            chat = await create_chat(db, chat_name="Test Chat", mode="conversational", user_id=user_id_str)
            print("Chat created successfully:", chat.id)
        except Exception as e:
            print("=== STACK TRACE BLOCKER 1 ===")
            traceback.print_exc()

asyncio.run(main())
