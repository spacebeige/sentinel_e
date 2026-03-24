"""Voice Activity Detection for AWAAZ - WebRTC VAD + energy fallback."""

import numpy as np
import logging
from typing import Optional
import webrtcvad

logger = logging.getLogger(__name__)


class VADProcessor:
    """Detects speech vs silence using WebRTC VAD."""

    def __init__(
        self,
        aggressiveness: int = 2,
        silence_ms: int = 700,
        max_utterance_s: int = 30,
        sample_rate: int = 16000,
    ):
        """
        Initialize VAD.

        Args:
            aggressiveness: 0-3, higher = more aggressive (default 2 recommended)
            silence_ms: milliseconds of silence to end utterance
            max_utterance_s: max seconds per utterance
            sample_rate: audio sample rate (should be 16000 for Whisper)
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration_ms = 20
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        self.silence_frames = silence_ms // self.frame_duration_ms
        self.max_frames = int(max_utterance_s * sample_rate / self.frame_size)

        self.audio_buffer = []
        self.silence_counter = 0
        self.frame_counter = 0
        self.in_speech = False

    def process_chunk(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """
        Process audio chunk. Returns complete utterance when detected, else None.

        Args:
            audio_bytes: PCM16 16kHz mono audio

        Returns:
            numpy array of PCM16 audio if utterance complete, else None
        """
        # Convert bytes to numpy array (int16)
        audio = np.frombuffer(audio_bytes, dtype=np.int16)

        # Process frame by frame
        for i in range(0, len(audio), self.frame_size):
            if i + self.frame_size > len(audio):
                break

            frame = audio[i : i + self.frame_size]
            frame_bytes = frame.tobytes()

            # Detect speech
            has_speech = self.vad.is_speech(frame_bytes, self.sample_rate)

            if has_speech:
                self.in_speech = True
                self.silence_counter = 0
            else:
                if self.in_speech:
                    self.silence_counter += 1

            self.audio_buffer.append(frame)
            self.frame_counter += 1

            # Check if utterance is complete
            if self.in_speech and self.silence_counter >= self.silence_frames:
                # End of utterance
                utterance = self._get_buffer()
                return utterance
            
            if self.frame_counter >= self.max_frames:
                # Force flush at max duration
                logger.warning("Utterance exceeded max duration, force-flushing")
                utterance = self._get_buffer()
                return utterance

        return None

    def _get_buffer(self) -> np.ndarray:
        """Get current buffer as numpy int16 array and reset."""
        if not self.audio_buffer:
            return np.array([], dtype=np.int16)

        result = np.concatenate(self.audio_buffer)
        self.audio_buffer = []
        self.silence_counter = 0
        self.frame_counter = 0
        self.in_speech = False
        return result

    def reset(self):
        """Reset VAD state (e.g., between calls)."""
        self.audio_buffer = []
        self.silence_counter = 0
        self.frame_counter = 0
        self.in_speech = False
