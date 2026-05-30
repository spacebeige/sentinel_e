import asyncio
from database.connection_v2 import SessionLocal, engine
from sqlalchemy import text

async def main():
    async with engine.begin() as conn:
        result = await conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))
        print([r[0] for r in result])
        
        result2 = await conn.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='users'"))
        print([r for r in result2])

asyncio.run(main())
