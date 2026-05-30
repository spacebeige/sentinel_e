import asyncio
from httpx import AsyncClient

async def test_health():
    async with AsyncClient() as client:
        res = await client.get("http://localhost:8000/health")
        print(res.status_code)
        print(res.json())

asyncio.run(test_health())
