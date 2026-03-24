"""Barge-in gate for AWAAZ -interruptible TTS playback on speech detection."""

import logging

logger = logging.getLogger(__name__)


class BargeInGate:
    """Enables caller interruption during TTS playback."""

    def __init__(self, session):
        self.session = session
        self._playing = False

    async def enable(self):
        """Call when TTS playback starts."""
        self._playing = True
        self.session.barge_in_pending = False
        logger.debug(f"Barge-in enabled for {self.session.session_id}")

    async def disable(self):
        """Call when TTS playback ends."""
        self._playing = False
        logger.debug(f"Barge-in disabled for {self.session.session_id}")

    def on_vad_activity(self, has_speech: bool) -> bool:
        """
        Called by VAD when speech is detected during playback.
        Returns True if playback should stop.
        """
        if self._playing and has_speech:
            self.session.barge_in_pending = True
            logger.info(f"Barge-in triggered for {self.session.session_id}")
            return True
        return False
