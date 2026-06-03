import asyncio
from database.connection_v2 import async_session_factory
from api.endpoints_v2 import create_new_chat
from fastapi import Request
import traceback
import logging

logging.basicConfig(level=logging.DEBUG)

async def main():
    async with async_session_factory() as db:
        try:
            # We mock the payload dependency (user, user_id, db)
            user_id = "cd4ee2f4-7894-4bc3-a9c1-a26c20dbf0d7"
            payload = ({"id": user_id}, user_id, db)
            print("1. Calling create_new_chat")
            res = await create_new_chat(title="Test", mode="conversational", payload=payload)
            print("Create Response:", res.body)
        except Exception as e:
            print("=== STACK TRACE BLOCKER 1 ===")
            traceback.print_exc()

asyncio.run(main())
