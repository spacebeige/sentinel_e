"""Main orchestrator for AWAAZ - starts ARI listener and AudioSocket server."""

import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Import AWAAZ modules
from src.ari_client import ARIClient
from src.session_store import SessionStore, AWAAZSession
from src.audiosocket_handler import AudioSocketHandler
from src.vad import VADProcessor
from src.playback_manager import PlaybackManager, pcm16_to_ulaw
from src.barge_in_gate import BargeInGate
from src.pipeline.stt import STTProcessor
from src.pipeline.nlp import ModelProcessor, check_emergency, update_session_from_nlp, parse_llm_output
from src.pipeline.tts import TTSProcessor
from src.hooks.pre_call import pre_call, hash_phone
from src.hooks.in_call import in_call
from src.hooks.post_call import post_call

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class AWAAZEngine:
    """Main AWAAZ orchestrator."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.sessions = SessionStore()
        self.audio_queue = asyncio.Queue()
        self.ari = None
        self.playback_mgr = None

        # Pipeline singletons (load once at startup)
        self.vad = None
        self.stt = None
        self.model = None
        self.tts = None

    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration."""
        try:
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing AWAAZ engine...")

        # Initialize pipeline models
        self.vad = VADProcessor(
            aggressiveness=self.config.get("vad", {}).get("aggressiveness", 2),
            silence_ms=self.config.get("vad", {}).get("silence_ms", 700),
        )

        self.stt = STTProcessor(
            model_size=self.config.get("stt", {}).get("model", "small")
        )
        await self.stt.load()


        self.model = ModelProcessor()
        await self.model.load()

        self.tts = TTSProcessor()
        await self.tts.load()

        self.playback_mgr = PlaybackManager(
            media_dir=self.config.get("playback", {}).get("media_dir", "/mnt/asterisk_media/ai-generated")
        )

        # Initialize ARI
        aster_cfg = self.config.get("asterisk", {})
        self.ari = ARIClient(
            username=aster_cfg.get("ari_username", "asterisk"),
            password=aster_cfg.get("ari_password", "asterisk"),
            host=aster_cfg.get("host", "127.0.0.1"),
            ari_port=aster_cfg.get("ari_port", 8088),
            stasis_app=aster_cfg.get("stasis_app", "awaaz"),
        )

        await self.ari.connect()
        logger.info("AWAAZ engine initialized")

    async def run(self):
        """Start AWAAZ services."""
        await self.initialize()

        # Start ARI event listener
        ari_task = asyncio.create_task(self._ari_listener())

        # Start AudioSocket server
        socket_task = asyncio.create_task(self._audiosocket_server())

        # Start call processor worker pool
        processor_tasks = [
            asyncio.create_task(self._call_processor_worker(i))
            for i in range(5)  # 5 concurrent call processors
        ]

        logger.info("[AWAAZ] Ready. Listening on :8090 (AudioSocket) and ARI")

        try:
            await asyncio.gather(ari_task, socket_task, *processor_tasks)
        except KeyboardInterrupt:
            logger.info("Shutdown requested")

    async def _ari_listener(self):
        """Listen to ARI events."""
        async def handler(event_type: str, event: dict):
            if event_type == "StasisStart":
                await self._on_stasis_start(event)
            elif event_type == "StasisEnd":
                await self._on_stasis_end(event)

        await self.ari.listen_events(handler)

    async def _on_stasis_start(self, event: dict):
        """Handle incoming call."""
        channel = event.get("channel", {})
        channel_id = channel.get("id")
        logger.info(f"Incoming call: {channel_id}")

        try:
            # Answer channel
            await self.ari.answer_channel(channel_id)

            # Get caller ANI
            ani = await self.ari.get_caller_ani(channel_id)

            # Create session
            session = await self.sessions.create(channel_id)
            session.caller_ani = ani

            # Run pre-call hook
            await pre_call(session)

            # Play greeting
            await self._play_greeting(session)

        except Exception as e:
            logger.error(f"Stasis start error: {e}", exc_info=True)

    async def _on_stasis_end(self, event: dict):
        """Handle call hangup."""
        channel = event.get("channel", {})
        channel_id = channel.get("id")
        logger.info(f"Call ended: {channel_id}")

        session = await self.sessions.get_by_channel(channel_id)
        if session:
            await post_call(session)
            await self.sessions.remove(session.session_id)

    async def _play_greeting(self, session):
        """Play greeting message."""
        greeting_text = {
            "hi": "नमस्ते। आपकी समस्या का समाधान करने में मदद करने के लिए आपका स्वागत है।",
            "en": "Hello. Welcome to AWAAZ. How can we help you today?",
            "ta": "வணக்கம்.தயவுசெய்து உங்கள் பிரச்சினை பற்றி சொல்லவும்.",
        }

        default_greeting = greeting_text.get(session.lang, greeting_text["en"])
        audio = self.tts.synthesize_to_bytes(default_greeting, session)

        if audio:
            session.turn_number += 1
            await self.playback_mgr.play_tts_file(session, audio, self.ari)

    async def _audiosocket_server(self):
        """Start TCP server for AudioSocket."""
        async def handle_client(reader, writer):
            handler = AudioSocketHandler(reader, writer, self.sessions, self.audio_queue)
            await handler.handle()

        socket_cfg = self.config.get("audiosocket", {})
        host = socket_cfg.get("host", "0.0.0.0")
        port = socket_cfg.get("port", 8090)

        server = await asyncio.start_server(handle_client, host, port)
        logger.info(f"AudioSocket server listening on {host}:{port}")

        async with server:
            await server.serve_forever()

    async def _call_processor_worker(self, worker_id: int):
        """Process audio uttera nces from audio queue."""
        logger.info(f"Call processor worker {worker_id} started")

        while True:
            try:
                session_id, audio_bytes = await self.audio_queue.get()

                session = await self.sessions.get(session_id)
                if not session or not session.is_active:
                    continue

                await self._process_utterance(session, audio_bytes)

            except Exception as e:
                logger.error(f"Processor error: {e}", exc_info=True)

    async def _process_utterance(self, session: AWAAZSession, audio_bytes: bytes):
        """Process single utterance through full pipeline."""
        try:
            # ── VAD + audio buffering ────────────────────────────────────────
            utterance = self.vad.process_chunk(audio_bytes)
            if not utterance or len(utterance) == 0:
                return

            # Save temp audio file
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_wav.write(utterance.tobytes())
            temp_wav.close()

            # ── STT ──────────────────────────────────────────────────────────
            # STT transcribes and natively detects language
            stt_result = await self.stt.transcribe(temp_wav.name, "auto" if session.turn_number == 1 else session.lang)
            text = stt_result.text if stt_result else ""
            
            if stt_result and stt_result.detected_language:
                detected_lang = stt_result.detected_language or "hi"
                # Keep language synchronized with current utterance detection
                # while preserving code-mix behavior for mixed speech.
                if detected_lang in ("hi", "mr", "gu", "ta", "te", "kn", "ml", "pa", "bn") and "eng" in text.lower():
                    session.lang = f"{detected_lang}-en"
                else:
                    session.lang = detected_lang
                session.confidence = stt_result.confidence
                
            if not text:
                logger.warning(f"No transcription for {session.session_id}")
                os.unlink(temp_wav.name)
                return

            logger.info(f"[{session.session_id}] Citizen ({session.lang}): {text[:100]}")

            # ── In-call hook ─────────────────────────────────────────────────
            await in_call(session, text, self.model)

            # ── State update ─────────────────────────────────────────────────
            self._update_state(session, text)

            # ── LLM reply ────────────────────────────────────────────────────
            llm_input_text = (getattr(stt_result, "native_script_text", None) or text)
            raw_reply = await self.model.generate(llm_input_text, session)

            # Parse structured JSON if present
            reply, meta = parse_llm_output(raw_reply)
            if meta:
                update_session_from_nlp(meta, session)

            reply_for_tts = await self.stt.to_native_script_text(reply, session.lang)
            logger.info(f"[{session.session_id}] AWAAZ ({session.lang}): {reply_for_tts[:100]}")

            # ── TTS ──────────────────────────────────────────────────────────
            audio = self.tts.synthesize_to_bytes(reply_for_tts, session)
            if audio:
                session.turn_number += 1
                gate = BargeInGate(session)
                await gate.enable()
                await self.playback_mgr.play_tts_file(session, audio, self.ari)
                await gate.disable()

            # ── Post-call check ──────────────────────────────────────────────
            if session.state in ("CLOSING", "EMERGENCY"):
                await post_call(session)
                await self.ari.hangup_channel(session.channel_id)

            # Update history
            session.history.append({
                "citizen": text,
                "assistant": reply_for_tts,
                "turn": session.turn_number,
            })

        except Exception as e:
            logger.error(f"Utterance processing error: {e}", exc_info=True)
        finally:
            if 'temp_wav' in locals():
                try:
                    os.unlink(temp_wav.name)
                except:
                    pass

    def _update_state(self, session: AWAAZSession, text: str):
        """Update call state machine."""
        if session.is_emergency:
            session.state = "EMERGENCY"
            return

        if session.state == "GREETING":
            session.state = "GATHERING"
        elif session.state == "GATHERING":
            if len(text) > 50:  # Sufficient detail
                session.state = "CONFIRMING"
        elif session.state == "CONFIRMING":
            session.state = "FILING"
        elif session.state == "FILING":
            session.state = "CLOSING"


async def main():
    """Entry point."""
    engine = AWAAZEngine("config.yaml")
    await engine.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
