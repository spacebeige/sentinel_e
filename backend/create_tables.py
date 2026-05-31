import asyncio
import os
from database.connection_v2 import engine
from database.models_v2 import Base

async def init():
    print("Creating tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Done")

if __name__ == "__main__":
    asyncio.run(init())
