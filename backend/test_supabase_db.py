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
    passwords = [
        "***REMOVED***",
        "password123",
        "sentinel-e-dev-secret-change-in-production-a3f8b2c1d4e5f6"
    ]
    for p in passwords:
        if await try_conn(p):
            break

asyncio.run(main())
