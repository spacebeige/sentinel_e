import asyncio
import asyncpg

async def try_conn(password):
    url = f"postgresql://postgres.kyqoygozcxxsmlkkraub:{password}@aws-0-ap-south-1.pooler.supabase.com:6543/postgres?sslmode=require"
    print(f"Trying password: {password[:5]}...")
    try:
        conn = await asyncpg.connect(url, timeout=5)
        print(f"SUCCESS with password: {password}!")
        await conn.close()
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False

async def main():
    import os
    test_password = os.getenv("SUPABASE_TEST_PASSWORD")
    if not test_password:
        print("Please set SUPABASE_TEST_PASSWORD environment variable.")
        return
    passwords = [test_password]
    for p in passwords:
        if await try_conn(p):
            break

asyncio.run(main())
