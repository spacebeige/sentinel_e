import asyncio
import asyncpg
import os

import os

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://localhost/neondb")
if "neondb_owner" in POSTGRES_URL and "npg_" in POSTGRES_URL:
    raise ValueError("Hardcoded production credentials detected. Please use .env")

async def run():
    conn = await asyncpg.connect(POSTGRES_URL)
    with open("/Users/ashwinagarkhed/.gemini/antigravity-ide/brain/80cdf0f4-e8b7-4a36-9bd5-a87afaab1d8c/supabase_migrations.sql", "r") as f:
        sql1 = f.read()
    with open("/Users/ashwinagarkhed/.gemini/antigravity-ide/brain/80cdf0f4-e8b7-4a36-9bd5-a87afaab1d8c/supabase_migrations_v2.sql", "r") as f:
        sql2 = f.read()
    
    try:
        await conn.execute(sql1)
        print("Migration 1 applied.")
    except Exception as e:
        print("Migration 1 error:", e)
        
    try:
        await conn.execute(sql2)
        print("Migration 2 applied.")
    except Exception as e:
        print("Migration 2 error:", e)
        
    await conn.close()

asyncio.run(run())
