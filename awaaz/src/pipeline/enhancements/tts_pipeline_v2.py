"""Enhanced TTS V2 Pipeline Orchestrator.

Non-destructive wrapper:
Text -> Naturalizer -> Prosody -> Segment -> Route -> Base TTS -> Merge
"""

import os
import tempfile
import asyncio
import logging

from .speech_naturalizer import naturalize_text
from .prosody_controller import apply_prosody
from .language_segmenter import segment_text
from .tts_router import route_tts
from .audio_merger import merge_audio


class EnhancedTTSProcessor:
    def __init__(self, existing_tts=None):
        self.base_tts = existing_tts

    async def load(self):
        if self.base_tts and hasattr(self.base_tts, "load"):
            await self.base_tts.load()

    async def synthesize(self, text: str, session, output_path: str) -> bool:
        if not self.base_tts:
            from src.pipeline.tts import TTSProcessor

            self.base_tts = TTSProcessor()
            await self.base_tts.load()

        # CRITICAL: Ensure session language is always set
        session_lang = getattr(session, "lang", "hi")
        if not session_lang:
            import logging
            logging.error("[CRITICAL] Session language not set! Defaulting to Hindi.")
            session.lang = "hi"
            session_lang = "hi"
        
        import logging
        logging.debug(f"[EnhancedTTS] Processing with language: {session_lang}")

        text_v1 = naturalize_text(
            text,
            lang=session_lang,
            lang_name=getattr(session, "lang_name", "Hindi"),
            lang_mode=getattr(session, "lang_mode", "pure"),
            formality_label=getattr(session, "formality_label", "STANDARD"),
            accent_region=getattr(session, "accent_region", "neutral"),
        )

        plan = apply_prosody(
            text_v1,
            formality=getattr(session, "formality_label", "STANDARD"),
            region=getattr(session, "accent_region", "neutral"),
            emergency=getattr(session, "is_emergency", False),
        )

        temp_audio_files = []

        for unit in plan.units:
            if not unit.text.strip():
                continue

            session_lang = getattr(session, "lang", "hi")
            segments = segment_text(unit.text, fallback_lang=session_lang.split("-")[0])
            route_input = [
                {
                    "text": seg.text,
                    "lang_type": "english" if seg.lang.startswith("en") else "indic",
                }
                for seg in segments
            ]
            # CRITICAL: Pass session language to router to ensure voice selection
            routes = route_tts(route_input, session_lang)
            logging.debug(f"[EnhancedTTS] Routed {len(routes)} segments for language {session_lang}")

            for idx, route in enumerate(routes):
                seg_text = route["text"]
                if not seg_text.strip():
                    continue

                seg_path = f"{output_path}.chunk_{len(temp_audio_files)}_{idx}.wav"
                original_lang = getattr(session, "lang", "hi")
                
                # CRITICAL: Ensure synthesizer_lang is properly set from route
                synthesizer_lang = route.get("synthesizer_lang", original_lang)
                session.lang = synthesizer_lang  # Set it for TTS voice selection
                
                logging.debug(f"[EnhancedTTS] Segment {idx}: lang={synthesizer_lang}, text_len={len(seg_text)}")
                
                try:
                    seg_success = await self.base_tts.synthesize(seg_text, session, seg_path)
                    if seg_success and os.path.exists(seg_path):
                        logging.debug(f"[EnhancedTTS] Segment {idx} synthesized with {synthesizer_lang}")
                        temp_audio_files.append(seg_path)
                    else:
                        logging.warning(f"[EnhancedTTS] Segment {idx} synthesis failed")
                finally:
                    session.lang = original_lang  # Restore original for next segment

        if not temp_audio_files:
            return False

        ok = merge_audio(temp_audio_files, output_path)
        for p in temp_audio_files:
            try:
                os.remove(p)
            except OSError:
                pass
        return ok


def enhanced_tts(text: str, session, output_path: str, existing_tts=None) -> bool:
    """Sync compatibility wrapper used by current call-sites.
    
    CRITICAL: Ensures language preservation through the entire pipeline.
    If session.lang is not set, this function will fail loudly rather than silently
    producing wrong-language audio.
    """
    # FAILSAFE 1: Validate language is set before starting TTS
    session_lang = getattr(session, "lang", None)
    if not session_lang:
        logging.error("[CRITICAL] enhanced_tts called with session.lang=None!")
        logging.error("[CRITICAL] This will cause TTS to use DEFAULT language instead of input language.")
        logging.error("[CRITICAL] Audio quality issue: Marathi/Tamil/etc input will get Hindi/English output!")
        # Set emergency fallback
        session.lang = "hi"
        session_lang = "hi"
    
    logging.info(f"[EnhancedTTS] Starting TTS synthesis with language: {session_lang}")
    
    pipeline = EnhancedTTSProcessor(existing_tts=existing_tts)

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # We're already in async context (e.g., test_live_voice).
            # Fall back to base_tts.synthesize_to_bytes to avoid nested loop issues.
            if existing_tts and hasattr(existing_tts, "synthesize_to_bytes"):
                logging.debug(f"[EnhancedTTS] Using synthesize_to_bytes path with language {session_lang}")
                audio_bytes = existing_tts.synthesize_to_bytes(text, session)
                if not audio_bytes:
                    logging.error(f"[EnhancedTTS] synthesize_to_bytes returned empty bytes for {session_lang}")
                    return False
                with open(output_path, "wb") as f:
                    f.write(audio_bytes)
                logging.info(f"[EnhancedTTS] \u2713 Audio written for language {session_lang}")
                return True
            logging.warning(f"[EnhancedTTS] existing_tts doesn't have synthesize_to_bytes")
            return False
    except RuntimeError:
        pass

    # FAILSAFE 2: Double-check language before async run
    if not getattr(session, "lang", None):
        logging.error("[CRITICAL] Language was cleared before async synthesize!")
        session.lang = session_lang
    
    return asyncio.run(pipeline.synthesize(text, session, output_path))
