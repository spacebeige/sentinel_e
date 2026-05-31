import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath("."))

from database.connection_v2 import async_session_factory
from sqlalchemy import text

async def verify_neon():
    print("Testing Neon DB Connection...")
    try:
        async with async_session_factory() as session:
            result = await session.execute(text("SELECT 1"))
            val = result.scalar()
            print(f"Neon Connection SUCCESS! SELECT 1 returned: {val}")
    except Exception as e:
        print(f"Neon Connection FAILED: {type(e).__name__} - {e}")

if __name__ == "__main__":
    asyncio.run(verify_neon())
