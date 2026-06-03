import asyncio
from database.connection_v2 import async_session_factory
from database.models import User
from sqlalchemy.future import select
import traceback

async def main():
    async with async_session_factory() as db:
        try:
            print("Executing select(User)...")
            users_result = await db.execute(select(User))
        except Exception as e:
            print("=== STACK TRACE BLOCKER 3 ===")
            traceback.print_exc()

asyncio.run(main())
