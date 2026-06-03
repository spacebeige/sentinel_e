import asyncio
from sqlalchemy import text
from database.connection_v2 import async_session_factory

async def main():
    async with async_session_factory() as db:
        # Check chats schema
        res = await db.execute(text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'chats';
        """))
        print("CHATS columns:")
        for row in res.fetchall():
            print(row)
            
        res2 = await db.execute(text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'messages';
        """))
        print("\nMESSAGES columns:")
        for row in res2.fetchall():
            print(row)

asyncio.run(main())
