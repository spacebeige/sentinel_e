"""Playback manager for AWAAZ - handles TTS audio → μ-law conversion → ARI playback."""

import os
import logging
import asyncio
import audioop
from typing import Optional
from scipy.signal import resample_poly
import numpy as np

logger = logging.getLogger(__name__)


def pcm16_to_ulaw(pcm_bytes: bytes, src_rate: int = 16000, target_rate: int = 8000) -> bytes:
    """
    Resample PCM16 audio and convert to μ-law.

    Args:
        pcm_bytes: Input PCM16 audio data
        src_rate: Source sample rate (default 16000)
        target_rate: Target sample rate (default 8000 for Asterisk)

    Returns:
        μ-law 8kHz mono audio bytes
    """
    # Convert bytes to int16 array
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)

    # Resample if needed
    if src_rate != target_rate:
        ratio = target_rate / src_rate
        audio = resample_poly(audio, target_rate, src_rate)

    # Convert back to int16
    audio = np.clip(audio, -32768, 32767).astype(np.int16)

    # Encode to μ-law
    ulaw = audioop.lin2ulaw(audio.tobytes(), 2)
    return ulaw


class PlaybackManager:
    """Manages TTS audio playback on Asterisk channels."""

    def __init__(self, media_dir: str = "/mnt/asterisk_media/ai-generated"):
        self.media_dir = media_dir
        os.makedirs(media_dir, exist_ok=True)

    async def play_tts_file(
        self,
        session,
        audio_bytes: bytes,
        ari_client,
        src_rate: int = 16000,
    ) -> Optional[str]:
        """
        Play TTS audio on channel.

        Args:
            session: AWAAZSession
            audio_bytes: PCM16 16kHz mono audio from TTS
            ari_client: ARIClient instance
            src_rate: Source sample rate

        Returns:
            Playback ID if successful, else None
        """
        try:
            # Convert PCM16 16kHz to μ-law 8kHz
            ulaw_audio = pcm16_to_ulaw(audio_bytes, src_rate, 8000)

            # Write to file
            filename = f"{session.session_id}_{session.turn_number:03d}.ulaw"
            filepath = os.path.join(self.media_dir, filename)

            with open(filepath, "wb") as f:
                f.write(ulaw_audio)

            logger.info(f"TTS audio written to {filepath}")

            # Play via ARI (note: no file extension in sound: URI)
            sound_file = f"ai-generated/{filename.replace('.ulaw', '')}"
            playback_id = await ari_client.play_sound(session.channel_id, sound_file)

            if playback_id:
                # Calculate sleep duration based on audio bytes
                duration_sec = len(ulaw_audio) / 8000.0
                await asyncio.sleep(duration_sec + 0.5)
                logger.info(f"Playback complete for {session.session_id}")
                return playback_id
            else:
                logger.error("Play sound returned no playback_id")
                return None

        except Exception as e:
            logger.error(f"Error playing TTS audio: {e}", exc_info=True)
            return None
