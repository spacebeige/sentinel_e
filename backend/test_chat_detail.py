import asyncio
from api.endpoints_v2 import get_chat_detail
from database.connection_v2 import async_session_factory
import logging

logging.basicConfig(level=logging.DEBUG)

async def main():
    async with async_session_factory() as db:
        try:
            # We use the live user id and chat id from final_acceptance_matrix.md
            res = await get_chat_detail(
                "7e42cb02-28de-4d03-9622-f0c26fce2ec3",
                payload=({}, "cd4ee2f4-7894-4bc3-a9c1-a26c20dbf0d7", db)
            )
            print("Response:", res)
        except Exception as e:
            print("EXCEPTION:", e)
            import traceback
            traceback.print_exc()

asyncio.run(main())
