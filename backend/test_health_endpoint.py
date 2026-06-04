import asyncio
from httpx import AsyncClient

async def test_health():
    async with AsyncClient() as client:
        res = await client.get("https://sentinel-e-evo.onrender.com/health")
        print(res.status_code)
        print(res.json())

asyncio.run(test_health())
