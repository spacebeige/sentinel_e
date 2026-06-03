import asyncio
from database.connection_v2 import async_session_factory
from database.crud import update_chat_metadata
from uuid import uuid4
import traceback

async def main():
    async with async_session_factory() as db:
        try:
            # We call update_chat_metadata with priority_answer and rounds exactly like in backend/main.py
            chat_id = uuid4()
            metadata = {"version": "test"}
            await update_chat_metadata(
                db, 
                chat_id, 
                priority_answer="Test answer", 
                machine_metadata=metadata, 
                rounds=3
            )
        except Exception as e:
            print("=== STACK TRACE BLOCKER 2 ===")
            traceback.print_exc()

asyncio.run(main())
