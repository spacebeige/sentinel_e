"""Post-call hook - create ticket, send SMS, log history."""

import logging
import aiosqlite
import asyncio
from datetime import datetime
from .pre_call import hash_phone, logger as pre_logger

logger = logging.getLogger(__name__)


def generate_ticket_id() -> str:
    """Generate unique ticket ID."""
    from datetime import datetime
    import random

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_suffix = str(random.randint(1000, 9999))
    return f"TKT-{timestamp}-{random_suffix}"


async def post_call(session, db_path: str = "awaaz.db") -> None:
    """
    Called on hangup or state==CLOSING.
    Creates ticket, sends SMS notification, logs call history.
    """
    if not session.ticket_id:
        session.ticket_id = generate_ticket_id()

    logger.info(f"Post-call processing for {session.session_id}")

    # Create ticket in database
    try:
        async with aiosqlite.connect(db_path) as db:
            ani_hash = hash_phone(session.caller_ani)
            await db.execute(
                """
                INSERT INTO tickets
                (ticket_id, session_id, phone_hash, lang, grievance_category,
                 dept_assigned, priority, complaint_summary, state, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,datetime('now'))
                """,
                (
                    session.ticket_id,
                    session.session_id,
                    ani_hash,
                    session.lang,
                    session.grievance_category or "GR-08",
                    session.dept_assigned or "General",
                    session.priority,
                    session.complaint_summary or "",
                    "NEW",
                ),
            )
            await db.commit()
            logger.info(f"Ticket created: {session.ticket_id}")

            # Update citizen record
            await db.execute(
                """
                INSERT INTO citizens (phone_hash, lang, accent_region, last_call, total_complaints)
                VALUES (?, ?, ?, datetime('now'), 1)
                ON CONFLICT(phone_hash) DO UPDATE SET
                    last_call = datetime('now'),
                    total_complaints = total_complaints + 1
                """,
                (ani_hash, session.lang, session.accent_region),
            )
            await db.commit()

    except Exception as e:
        logger.error(f"Failed to create ticket: {e}")

    # Send SMS (non-blocking)
    asyncio.create_task(_send_sms_notification(session))

    # Log call history
    _log_call_history(session)


async def _send_sms_notification(session) -> None:
    """Send SMS to caller with ticket info."""
    try:
        # Placeholder for SMS API integration
        logger.info(f"SMS notification would be sent to {hash_phone(session.caller_ani)[:8]}...")
    except Exception as e:
        logger.error(f"SMS send error: {e}")


def _log_call_history(session) -> None:
    """Log call details for monitoring."""
    duration = __import__("time").time() - session.call_start_ts
    logger.info(
        "Call completed",
        session_id=session.session_id,
        ticket_id=session.ticket_id,
        lang=session.lang,
        turns=session.turn_number,
        state=session.state,
        is_emergency=session.is_emergency,
        duration_s=duration,
        category=session.grievance_category,
    )
