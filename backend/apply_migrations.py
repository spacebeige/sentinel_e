import asyncio
import asyncpg
import os

POSTGRES_URL = "postgresql://neondb_owner:***REMOVED***@ep-noisy-morning-a10vt6me-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"

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
