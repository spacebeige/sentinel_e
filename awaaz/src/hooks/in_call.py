"""In-call hook - language detection and emergency check."""

import logging

logger = logging.getLogger(__name__)


async def in_call(session, text: str, model_processor) -> None:
    """
    Called after every STT transcription.
    Checks for emergency. (Language is now handled via API natively)
    """
    if not text:
        return

    try:
        # Emergency check
        from .pre_call import logger as pre_logger  # avoid circular import

        from ..pipeline.nlp import check_emergency

        if check_emergency(text, session.lang, model_processor):
            session.is_emergency = True
            session.state = "EMERGENCY"
            session.priority = "CRITICAL"
            logger.warning(f"Emergency detected in {session.session_id}")

    except Exception as e:
        logger.error(f"In-call hook error: {e}")
