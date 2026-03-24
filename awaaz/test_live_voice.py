#!/usr/bin/env python3
"""
AWAAZ Live Voice Test (inside awaaz folder)
Real-time voice/file input -> STT -> LLM -> TTS playback.

Language integration includes:
- Hinglish (hi-en) code-mix handling
- Marathi (mr)
- Multilingual Indian language routing with safe fallbacks
"""

import os
import sys
import time
import asyncio
import argparse
import re
from dotenv import load_dotenv

from src.pipeline.stt import STTProcessor
from src.pipeline.nlp import ModelProcessor
from src.pipeline.tts import TTSProcessor
from src.session_store import AWAAZSession
import tempfile

try:
    from awaaz_recorder import MicCalibrator, VADEngine, Recorder, save_wav, TARGET_SR
    RECORDER_AVAILABLE = True
except ImportError:
    print("⚠️  awaaz_recorder not available, will skip mic recording")
    TARGET_SR = 16000
    RECORDER_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))


LANGUAGE_SETTINGS = {
    "hi": {"name": "Hindi", "gtts": "hi"},
    "hi-en": {"name": "Hinglish (Hindi-English)", "gtts": "hi"},
    "mr": {"name": "Marathi", "gtts": "mr"},
    "mr-en": {"name": "Marathi-English", "gtts": "mr"},
    "ta": {"name": "Tamil", "gtts": "ta"},
    "te": {"name": "Telugu", "gtts": "te"},
    "kn": {"name": "Kannada", "gtts": "kn"},
    "ml": {"name": "Malayalam", "gtts": "ml"},
    "gu": {"name": "Gujarati", "gtts": "gu"},
    "bn": {"name": "Bengali", "gtts": "bn"},
    "pa": {"name": "Punjabi", "gtts": "pa"},
    "or": {"name": "Odia", "gtts": "or"},
    "as": {"name": "Assamese", "gtts": "bn"},
    "kok": {"name": "Konkani", "gtts": "mr"},
    "kok-en": {"name": "Konkani-English", "gtts": "mr"},
    "bgc": {"name": "Haryanvi", "gtts": "hi"},
    "bgc-en": {"name": "Haryanvi-English", "gtts": "hi"},
    "bho": {"name": "Bhojpuri", "gtts": "hi"},
    "bho-en": {"name": "Bhojpuri-English", "gtts": "hi"},
    "mai": {"name": "Maithili", "gtts": "hi"},
    "mai-en": {"name": "Maithili-English", "gtts": "hi"},
    "awa": {"name": "Awadhi", "gtts": "hi"},
    "awa-en": {"name": "Awadhi-English", "gtts": "hi"},
    "doi": {"name": "Dogri", "gtts": "hi"},
    "doi-en": {"name": "Dogri-English", "gtts": "hi"},
    "mwr": {"name": "Marwadi", "gtts": "hi"},
    "mwr-en": {"name": "Marwadi-English", "gtts": "hi"},
    "pah": {"name": "Pahadi", "gtts": "hi"},
    "pah-en": {"name": "Pahadi-English", "gtts": "hi"},
    "en": {"name": "English", "gtts": "en"},
}


def _expand_mixed_variants(settings: dict) -> dict:
    expanded = dict(settings)
    for code, meta in list(settings.items()):
        if code == "en" or code.endswith("-en"):
            continue
        mixed = f"{code}-en"
        if mixed not in expanded:
            expanded[mixed] = {
                "name": f"{meta['name']}-English",
                "gtts": meta["gtts"],
            }
    return expanded


LANGUAGE_SETTINGS = _expand_mixed_variants(LANGUAGE_SETTINGS)

LANG_ALIASES = {
    "hin": "hi", "hindi": "hi",
    "mar": "mr", "marathi": "mr",
    "eng": "en", "english": "en",
    "kan": "kn", "kannada": "kn",
    "tam": "ta", "tamil": "ta",
    "tel": "te", "telugu": "te",
    "mal": "ml", "malayalam": "ml",
    "guj": "gu", "gujarati": "gu", "gujrati": "gu",
    "pan": "pa", "punjabi": "pa",
    "ben": "bn", "bengali": "bn",
    "odi": "or", "odia": "or",
    "asm": "as", "assamese": "as",
    "kok": "kok", "konkani": "kok",
    "bgc": "bgc", "haryanvi": "bgc",
    "bho": "bho", "bhojpuri": "bho",
    "mai": "mai", "maithili": "mai",
    "awa": "awa", "awadhi": "awa",
    "doi": "doi", "dogri": "doi", "dongri": "doi",
    "mwr": "mwr", "marwadi": "mwr",
    "pah": "pah", "pahadi": "pah", "pahari": "pah"
}


def _normalize_lang_code(lang_code: str) -> str:
    if not lang_code:
        return "hi"
    norm = str(lang_code).strip().lower()
    if norm.startswith("en"):
        return "en"
    return LANG_ALIASES.get(norm, norm)


_COMMON_EN_WORDS = {
    "the", "is", "are", "am", "to", "for", "and", "with", "can", "please",
    "help", "issue", "problem", "service", "name", "number", "address", "today",
    "tomorrow", "urgent", "complaint", "status", "application", "payment", "bill",
}


def _has_english_code_mix(text: str, base_lang: str) -> bool:
    if base_lang == "en":
        return False
    tokens = re.findall(r"[A-Za-z]{2,}", text or "")
    if not tokens:
        return False
    english_hits = sum(1 for t in tokens if t.lower() in _COMMON_EN_WORDS)
    ratio = english_hits / max(len(tokens), 1)
    return english_hits >= 2 and ratio >= 0.2


def _is_mostly_english(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z]{2,}", text or "")
    if len(tokens) < 3:
        return False
    english_hits = sum(1 for t in tokens if t.lower() in _COMMON_EN_WORDS)
    if english_hits >= 2:
        return True
    ascii_chars = sum(1 for ch in (text or "") if ch.isascii() and ch.isalpha())
    alpha_chars = sum(1 for ch in (text or "") if ch.isalpha())
    if alpha_chars == 0:
        return False
    return (ascii_chars / alpha_chars) >= 0.9




def _resolve_transcribe_language(session_lang: str) -> str:
    if session_lang and session_lang.endswith("-en"):
        return "auto"
    return session_lang


async def run_pipeline(audio_file_path: str, session_id: str = None, lang_override: str = None, output_path: str = None):
    """Execute full voice pipeline: STT -> NLP -> TTS
    
    Args:
        audio_file_path: Path to input audio file
        session_id: Optional session ID
        lang_override: Optional language override
        output_path: Optional path to save model response audio (e.g., ./response.wav)
    """
    print("\n" + "="*60)
    print("  AWAAZ Live Voice Pipeline")
    print("="*60)

    # Create session
    session = AWAAZSession(session_id=session_id or "test_" + str(int(time.time())))
    session.lang = "hi"
    session.lang_name = "Hindi"
    session.gtts_lang = "hi"
    session.state = "GREETING"
    
    from src.pipeline.tts import TTSProcessor
    from src.pipeline.enhancements.tts_pipeline_v2 import enhanced_tts
    from src.pipeline.phonetic_converter import PhoneticConverter

    # Initialize modules
    converter = PhoneticConverter()
    # Use Groq Whisper for better Indian language detection
    stt = STTProcessor(preferred_provider="groq_whisper")
    nlp = ModelProcessor(provider="groq", model="llama-3.1-8b-instant")
    base_tts = TTSProcessor(preferred_provider="elevenlabs")

    print("\n[SYSTEM] Initializing pipeline modules...")
    await stt.load()
    await nlp.load()
    await base_tts.load()
    stt_chain = ", ".join(getattr(p, "name", "unknown") for p in getattr(stt, "providers", []))
    print(f"  ✓ STT provider order: {stt_chain}")
    print("  ✓ STT (Groq Whisper Large V3) loaded")
    print(f"  ✓ TTS (preferred={getattr(base_tts, 'preferred_provider', 'elevenlabs')}) ready")
    print("  ✓ LLM (Groq llama-3.1-8b) ready")

    # Step 1: STT (Speech-to-Text)
    print(f"\n[1/3] TRANSCRIPTION (Multi-provider STT)")
    print(f"  Input: {audio_file_path}")
    stt_start = time.time()

    stt_result = await stt.transcribe(audio_file_path, language="auto" if not lang_override else _normalize_lang_code(lang_override))
    stt_time = time.time() - stt_start
    
    if not stt_result or not stt_result.text:
        print("  ❌ [ERROR] STT failed to transcribe any text")
        return False
        
    detected_lang = stt_result.detected_language or "hi"
    detected_conf = stt_result.confidence
    transcript = stt_result.text
    transcript_native = stt_result.native_script_text or transcript
    
    # NEW ALGORITHMIC FIX FOR ALL LANGUAGES: 
    # STT engines often incorrectly flag audio as "Hindi" when spoken in regional languages,
    # despite transcribing the characters in the correct native script.
    # We inspect the native text Unicode blocks to absolutely force the correct language.
    script_ranges = [
        (r'[\u0A00-\u0A7F]', 'pa'), # Gurmukhi (Punjabi)
        (r'[\u0A80-\u0AFF]', 'gu'), # Gujarati
        (r'[\u0B80-\u0BFF]', 'ta'), # Tamil
        (r'[\u0C00-\u0C7F]', 'te'), # Telugu
        (r'[\u0C80-\u0CFF]', 'kn'), # Kannada
        (r'[\u0D00-\u0D7F]', 'ml'), # Malayalam
        (r'[\u0980-\u09FF]', 'bn'), # Bengali/Assamese
        (r'[\u0B00-\u0B7F]', 'or'), # Odia
    ]
    import re
    # Only override if the detected lang from the API appears to be 'hi' or an empty default,
    # OR forcefully apply it to fix STT misclassification for any engine.
    for pattern, lang_code in script_ranges:
        if re.search(pattern, transcript_native):
            detected_lang = lang_code
            break
    stt_provider = getattr(stt_result, "provider", "unknown")

    transcript_phonetic = stt_result.phonetic_text
    if not transcript_phonetic or transcript_phonetic == transcript_native:
        transcript_phonetic = converter.convert_to_phonetic(transcript_native, detected_lang)

    # DISABLED: English rescue pass is too aggressive and misclassifies non-English speech
    # as English (ASCII gibberish looks like English). Trust the primary STT detection instead.
    # When using Groq Whisper, language detection is accurate for Indian languages.
    # if not lang_override and _normalize_lang_code(detected_lang) != "en":
    #     try:
    #         en_result = await stt.transcribe(audio_file_path, language="en")
    #         if en_result and en_result.text and _is_mostly_english(en_result.text):
    #             transcript = en_result.text
    #             detected_lang = en_result.detected_language or "en"
    #             detected_conf = max(detected_conf, en_result.confidence)
    #             stt_provider = f"{stt_provider} -> {en_result.provider}(en-forced)"
    #             print("[LANG] Applied English rescue pass based on forced-English STT result")
    #     except Exception:
    #         # Keep primary STT result if rescue pass fails.
    #         pass

    # Skip using lang_detector completely, trust the API's returned detected_lang
    normalized = _normalize_lang_code(detected_lang)
    if normalized in LANGUAGE_SETTINGS:
        session.lang = normalized
    else:
        detected_str = str(detected_lang or "").lower()
        if detected_str.startswith("en") or "english" in detected_str or _is_mostly_english(transcript):
            session.lang = "en"
        else:
            session.lang = "hi"
    
    # Universal English code-mix detector for every supported Indian language profile
    base_lang = session.lang.split("-en")[0]
    mixed_lang = f"{base_lang}-en"
    if _has_english_code_mix(transcript, base_lang) and mixed_lang in LANGUAGE_SETTINGS:
        session.lang = mixed_lang
        print(f"[LANG] Dynamically engaged mixed-language profile for: {session.lang}")

    lang_meta = LANGUAGE_SETTINGS.get(session.lang, LANGUAGE_SETTINGS["hi"])
    session.lang_name = lang_meta["name"]
    session.gtts_lang = lang_meta["gtts"]
    session.lang_mode = "pure" if "-en" not in session.lang else "mixed"

    print(f"  ✓ Transcription (phonetic): '{transcript_phonetic}'")
    if transcript_native != transcript_phonetic:
        print(f"  ✓ Transcription (native):   '{transcript_native}'")
    
    # ✨ NEW: Display phonetic metadata (accent + English meaning)
    accent_type = getattr(stt_result, 'accent_type', 'standard')
    english_meaning = getattr(stt_result, 'english_meaning', None)
    if accent_type and accent_type != "standard":
        print(f"  🎯 Accent Detected: {accent_type.replace('_', ' ').title()}")
    if english_meaning:
        print(f"  🇬🇧 English Meanings: {english_meaning}")
    
    print(f"  🔧 STT provider used: {stt_provider}")
    print(
        f"  🌐 Language profile: {session.lang_name} ({session.lang}) | "
        f"mode={session.lang_mode} | detected_conf={detected_conf:.2f}"
    )
    print(f"  ⏱️  Time: {stt_time:.2f}s")

    # Step 2: NLP (Language Model)
    print(f"\n[2/3] AI PROCESSING (Groq LLM)")
    nlp_start = time.time()
    
    # Feed native-script transcript to LLM when available for better language fidelity.
    nlp_input = transcript_native if transcript_native else transcript
    response_text = await nlp.generate(nlp_input, session)
    print(f"  🔍 [DEBUG] LLM reply language check:")
    print(f"      Session language: {session.lang}")
    print(f"      LLM output length: {len(response_text)} chars")
    
    # CRITICAL: Verify language alignment before TTS
    if not nlp._is_reply_language_aligned(response_text, session.lang):
        print(f"  ⚠️  [WARN] LLM output NOT aligned with {session.lang}!")
        print(f"      Raw output: {response_text[:100]}...")
    
    # Prefer native-script text for on-screen output + TTS playback when available.
    tts_input_text = await stt.to_native_script_text(response_text, session.lang)
    nlp_time = time.time() - nlp_start
    
    print(f"  ✓ Response (native): '{tts_input_text}'")
    
    print(f"  ⏱️  Time: {nlp_time:.2f}s")

    # Step 3: TTS (Text-to-Speech)
    print(f"\n[3/3] AUDIO SYNTHESIS (ElevenLabs/Groq TTS with Enhancements)")
    print(f"  🔍 [DEBUG] TTS language handoff:")
    print(f"      Session language: {session.lang}")
    print(f"      Input text (first 50 chars): {tts_input_text[:50]}...")
    has_latin = bool(re.search(r"[A-Za-z]{2,}", tts_input_text or ""))
    print(f"      Text is native script: {not has_latin}")
    tts_start = time.time()
    
    temp_wav = f"/tmp/{session.session_id}_reply.wav"
    # CRITICAL: Use native script for TTS to ensure natural-sounding output
    # Native script (e.g., Punjabi/Gurmukhi) sounds much smoother than phonetic romanization
    # Pass language explicitly to prevent default voice selection
    success = enhanced_tts(tts_input_text, session, temp_wav, existing_tts=base_tts)
    tts_time = time.time() - tts_start
    
    if success:
        print(f"  ✓ TTS Language: {session.lang} (voice should match)")
    
    if success:
        # Use persistent output path if provided, otherwise use temp file
        final_wav = output_path if output_path else temp_wav
        if output_path and output_path != temp_wav:
            # Copy temp file to requested output location
            import shutil
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            shutil.copy(temp_wav, output_path)
            print(f"  ✓ Audio Saved (persistent): {output_path}")
        else:
            print(f"  ✓ Audio Generated (temp): {temp_wav}")
        print(f"  ⏱️  Time: {tts_time:.2f}s")
        
        # Auto-play on macOS
        if os.uname().sysname == "Darwin":
            print("  🔊 Playing audio response...")
            os.system(f"afplay {final_wav} 2>/dev/null")
        return True
    else:
        print("  ❌ [ERROR] TTS synthesis failed")
        return False

def record_live_audio(duration_s: int = 10):
    """Record audio from microphone with VAD voice isolation"""
    if not RECORDER_AVAILABLE:
        print("❌ awaaz_recorder module not available. Exiting.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("  MICROPHONE RECORDING & VOICE ISOLATION")
    print("="*60)
    
    print("\n[REC] Calibrating microphone...")
    try:
        calibrator = MicCalibrator()
        calibration = calibrator.calibrate()
        print("  ✓ Microphone calibrated")
    except Exception as e:
        print(f"  ⚠️  Calibration warning: {e}")
        calibration = None
    
    print(f"\n[REC] Recording for up to {duration_s}s...")
    print("  🎤 Speak now. Recording stops on {duration_s}s silence.")
    
    try:
        vad = VADEngine(aggressiveness=2)
        recorder = Recorder(vad=vad, device_index=None)
        
        audio = recorder.record(
            max_duration_s=duration_s,
            silence_ms=1500,
            calibration=calibration
        )
        
        if audio is None or len(audio) == 0:
            print("  ❌ No audio captured")
            return None
        
        duration = len(audio) / TARGET_SR
        print(f"  ✓ Recording complete: {duration:.2f}s captured")
        
        # Save to temp file
        audio_path = tempfile.mktemp(suffix=".wav")
        save_wav(audio, TARGET_SR, audio_path)
        print(f"  ✓ Audio saved: {audio_path}")
        
        return audio_path
        
    except Exception as e:
        print(f"  ❌ Recording error: {e}")
        return None


async def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("  AWAAZ Voice AI System - Live Test")
    print("="*60)

    parser = argparse.ArgumentParser(description="AWAAZ live voice pipeline test")
    parser.add_argument("--mode", choices=["mic", "file"], default="mic", help="Input mode")
    parser.add_argument("--input", type=str, default="", help="Path to input file when --mode file")
    parser.add_argument("--lang", type=str, default="", help="Optional language override (hi, hi-en, mr, mr-en, ta, te, kn, ml, gu, bn, pa, en)")
    parser.add_argument("--duration", type=int, default=10, help="Max microphone capture duration in seconds")
    parser.add_argument("--output", type=str, default="", help="Path to save model response audio (e.g., ./response.wav or ./responses/reply.wav)")
    args = parser.parse_args()

    if args.mode == "file":
        audio_path = args.input
        if not audio_path:
            print("❌ --input is required with --mode file")
            sys.exit(1)
        if not os.path.exists(audio_path):
            print(f"❌ File not found: {audio_path}")
            sys.exit(1)
        print(f"\n✓ Using test file: {audio_path}")
    else:
        # Record from mic (default)
        audio_path = record_live_audio(duration_s=args.duration)
        if not audio_path:
            sys.exit(1)
    
    # Run the full pipeline
    success = await run_pipeline(audio_path, lang_override=(args.lang or None), output_path=(args.output or None))
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("✅ Pipeline completed successfully!")
    else:
        print("❌ Pipeline failed at one or more stages")
    print("="*60 + "\n")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
