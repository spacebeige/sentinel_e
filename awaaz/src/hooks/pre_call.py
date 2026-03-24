"""Pre-call hook - fetch existing citizen record from SQLite."""

import logging
import hashlib
import aiosqlite
from typing import Optional

logger = logging.getLogger(__name__)


def hash_phone(ani: str) -> str:
    """Hash phone number for privacy."""
    return hashlib.sha256(ani.encode()).hexdigest()


async def pre_call(session, db_path: str = "awaaz.db") -> None:
    """
    Fetch existing citizen record by phone number.
    Called immediately after answer, before greeting.
    """
    if not session.caller_ani:
        logger.debug("No ANI available for pre-call lookup")
        return

    try:
        async with aiosqlite.connect(db_path) as db:
            ani_hash = hash_phone(session.caller_ani)
            async with db.execute(
                "SELECT lang, accent_region, last_call, total_complaints "
                "FROM citizens WHERE phone_hash = ? LIMIT 1",
                (ani_hash,),
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    session.citizen_record = {
                        "lang": row[0],
                        "accent_region": row[1],
                        "last_call": row[2],
                        "total_complaints": row[3],
                    }
                    logger.info(
                        f"Pre-call lookup: found existing citizen with {row[3]} complaints"
                    )
    except Exception as e:
        logger.error(f"Pre-call lookup error: {e}")
