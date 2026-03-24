"""AudioSocket handler for AWAAZ - TCP server receiving audio from Asterisk."""

import asyncio
import logging
import struct
import uuid as uuid_module
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class AudioSocketHandler:
    """Handles AudioSocket connections from Asterisk."""

    def __init__(self, reader, writer, sessions, audio_queue: asyncio.Queue):
        self.reader = reader
        self.writer = writer
        self.sessions = sessions
        self.audio_queue = audio_queue
        self.session = None
        self.channel_id = None

    async def handle(self):
        """Handle incoming AudioSocket connection."""
        try:
            # First frame: UUID
            uuid_frame = await self._read_frame()
            if not uuid_frame:
                logger.error("Failed to read UUID frame")
                return

            frame_type, frame_data = uuid_frame
            if frame_type != 0x00:
                logger.error(f"Expected UUID frame (0x00), got 0x{frame_type:02x}")
                return

            # UUID is 16 bytes
            channel_uuid = frame_data.hex()
            logger.info(f"AudioSocket connected: {channel_uuid}")

            # Get or create session
            self.session = await self.sessions.get_by_channel(channel_uuid)
            if not self.session:
                self.session = await self.sessions.create(channel_uuid)
            
            self.channel_id = self.session.session_id

            # Send back UUID frame to acknowledge
            await self._send_frame(0x00, frame_data)

            # Listen for audio frames
            await self._audio_loop()

        except Exception as e:
            logger.error(f"AudioSocket error: {e}", exc_info=True)
        finally:
            self.writer.close()
            await self.writer.wait_closed()
            if self.session:
                await self.sessions.close(self.session.session_id)

    async def _audio_loop(self):
        """Read audio frames continuously."""
        while True:
            try:
                frame = await self._read_frame()
                if not frame:
                    logger.info(f"AudioSocket EOF for {self.channel_id}")
                    break

                frame_type, frame_data = frame

                if frame_type == 0x10:  # Audio frame
                    # Queue audio for processing
                    await self.audio_queue.put((self.channel_id, frame_data))
                elif frame_type == 0x00 and len(frame_data) == 0:  # Hangup
                    logger.info(f"Hangup received for {self.channel_id}")
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Audio frame error: {e}")
                break

    async def _read_frame(self) -> Optional[tuple]:
        """
        Read AudioSocket frame.

        Format: [1 byte type][2 bytes big-endian length][N bytes payload]
        Returns: (frame_type, frame_data) or None on EOF
        """
        try:
            # Read header (3 bytes)
            header = await self.reader.readexactly(3)
            if not header:
                return None

            frame_type = header[0]
            frame_len = struct.unpack(">H", header[1:3])[0]

            # Read payload
            if frame_len > 0:
                payload = await self.reader.readexactly(frame_len)
            else:
                payload = b""

            return (frame_type, payload)

        except asyncio.IncompleteReadError:
            return None
        except Exception as e:
            logger.error(f"Frame read error: {e}")
            return None

    async def _send_frame(self, frame_type: int, payload: bytes):
        """Send AudioSocket frame."""
        try:
            frame_len = len(payload)
            header = bytes([frame_type]) + struct.pack(">H", frame_len)
            self.writer.write(header + payload)
            await self.writer.drain()
        except Exception as e:
            logger.error(f"Frame send error: {e}")
