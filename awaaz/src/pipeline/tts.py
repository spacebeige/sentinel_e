import os
import re
import logging
import time, requests, base64, struct, wave
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict

# Import phonetic converter for multilingual support
try:
    from awaaz.src.pipeline.phonetic_converter import PhoneticConverter
    PHONETIC_CONVERTER_AVAILABLE = True
except ImportError:
    PHONETIC_CONVERTER_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── CRITICAL: Load environment variables from .env file ──────────────────────
# Search for .env in multiple locations to support both local and module paths
_env_paths = [
    Path(__file__).parent.parent.parent / "awaaz" / ".env",  # awaaz/.env
    Path(__file__).parent.parent.parent / ".env",            # root/.env
    Path.cwd() / ".env",                                      # current working dir
    Path.cwd() / "awaaz" / ".env",                           # cwd/awaaz/.env
]

for env_path in _env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"[INIT] Loaded environment from: {env_path}")
        break
else:
    logger.warning("[INIT] No .env file found in standard locations. Using process environment.")

# Unicode script ranges — covers all AWAAZ supported languages
SCRIPT_RANGES = {
    "devanagari": (r"[\u0900-\u097F]",
                   ["mr", "hi", "ne", "sa", "kok", "bho", "mai", "doi", "brx"]),
    "arabic":     (r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]",
                   ["ar", "ur", "fa", "ps", "sd", "ug"]),
    "tamil":      (r"[\u0B80-\u0BFF]", ["ta"]),
    "telugu":     (r"[\u0C00-\u0C7F]", ["te"]),
    "kannada":    (r"[\u0C80-\u0CFF]", ["kn"]),
    "malayalam":  (r"[\u0D00-\u0D7F]", ["ml"]),
    "bengali":    (r"[\u0980-\u09FF]", ["bn", "as"]),
    "gujarati":   (r"[\u0A80-\u0AFF]", ["gu"]),
    "gurmukhi":   (r"[\u0A00-\u0A7F]", ["pa"]),
    "odia":       (r"[\u0B00-\u0B7F]", ["or"]),
    "sinhala":    (r"[\u0D80-\u0DFF]", ["si"]),
    "thai":       (r"[\u0E00-\u0E7F]", ["th"]),
    "burmese":    (r"[\u1000-\u109F]", ["my"]),
    "khmer":      (r"[\u1780-\u17FF]", ["km"]),
    "georgian":   (r"[\u10A0-\u10FF]", ["ka"]),
    "armenian":   (r"[\u0530-\u058F]", ["hy"]),
    "hebrew":     (r"[\u0590-\u05FF]", ["he", "yi"]),
    "cyrillic":   (r"[\u0400-\u04FF]", ["ru", "uk", "bg", "sr", "mk", "be", "kk"]),
    "cjk":        (r"[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u30FF\uAC00-\uD7AF]",
                   ["zh", "ja", "ko"]),
    "latin":      (r"[a-zA-Z\u00C0-\u024F]",
                   ["en", "fr", "de", "es", "pt", "it", "nl", "pl", "cs",
                    "ro", "sv", "da", "fi", "no", "hu", "id", "ms", "vi",
                    "tr", "az", "uz", "tk"]),
}

# Build reverse map: lang_code -> expected_script_name
_LANG_TO_SCRIPT = {}
for script_name, (pattern, langs) in SCRIPT_RANGES.items():
    for lang in langs:
        _LANG_TO_SCRIPT[lang] = script_name

_COMPILED = {name: re.compile(pat) for name, (pat, _) in SCRIPT_RANGES.items()}

def detect_script(text: str) -> str:
    counts = {}
    for name, rx in _COMPILED.items():
        matches = rx.findall(text)
        if matches:
            counts[name] = len(matches)
    if not counts:
        return "unknown"
    return max(counts, key=counts.get)

def is_native_script(text: str, lang: str) -> bool:
    detected = detect_script(text)
    expected = _LANG_TO_SCRIPT.get(lang, None)
    if expected is None:
        result = True
    else:
        result = (detected == expected)
    logger.debug(
        "[D1-SCRIPT] lang=%r | expected_script=%r | detected_script=%r | "
        "is_native=%s | text_sample=%r",
        lang, expected, detected, result, text[:60]
    )
    return result

INDIC_LANGS = {
    "mr", "hi", "bn", "ta", "te", "kn", "ml", "gu", "pa", "or",
    "as", "ne", "sa", "kok", "bho", "mai", "doi", "brx", "si",
    "awa", "mwr", "bgc", "tcy", "konkani", "ur"  # Added Tulu (tcy) and Urdu
}
ARABIC_SCRIPT_LANGS = {"ar", "ur", "fa", "ps", "sd", "ug"}
CJK_LANGS           = {"zh", "ja", "ko"}
CYRILLIC_LANGS      = {"ru", "uk", "bg", "sr", "mk", "be", "kk"}

def get_provider_order(lang: str) -> list[str]:
    # Extract base language if formatted like "mr-IN"
    lang = lang.split("-")[0]
    # Simple provider chain: Sarvam first (best for Indian languages), ElevenLabs fallback
    return ["sarvam", "elevenlabs"]

ELEVENLABS_MODEL = "eleven_multilingual_v2"
ELEVENLABS_VOICE_MAP = {
    "en":   os.environ.get("ELEVENLABS_VOICE_EN",  "pFZP5JQG7iQjIQuC4Bku"),
    "mr":   os.environ.get("ELEVENLABS_VOICE_MR",  "pFZP5JQG7iQjIQuC4Bku"),
    "hi":   os.environ.get("ELEVENLABS_VOICE_HI",  "pFZP5JQG7iQjIQuC4Bku"),
    "ar":   os.environ.get("ELEVENLABS_VOICE_AR",  "pFZP5JQG7iQjIQuC4Bku"),
    "ta":   os.environ.get("ELEVENLABS_VOICE_TA",  "pFZP5JQG7iQjIQuC4Bku"),
    "te":   os.environ.get("ELEVENLABS_VOICE_TE",  "pFZP5JQG7iQjIQuC4Bku"),
    "kn":   os.environ.get("ELEVENLABS_VOICE_KN",  "pFZP5JQG7iQjIQuC4Bku"),
    "ml":   os.environ.get("ELEVENLABS_VOICE_ML",  "pFZP5JQG7iQjIQuC4Bku"),
    "tcy":  os.environ.get("ELEVENLABS_VOICE_TCY", "pFZP5JQG7iQjIQuC4Bku"),
    "ur":   os.environ.get("ELEVENLABS_VOICE_UR",  "pFZP5JQG7iQjIQuC4Bku"),
}
ELEVENLABS_DEFAULT_VOICE = os.environ.get(
    "ELEVENLABS_VOICE_DEFAULT", "pFZP5JQG7iQjIQuC4Bku"
)

SARVAM_LANG_MAP = {
    "mr": "mr-IN", "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN",
    "te": "te-IN", "kn": "kn-IN", "ml": "ml-IN", "gu": "gu-IN",
    "pa": "pa-IN", "or": "or-IN", "as": "as-IN", "en": "en-IN",
    "si": "si-LK", "kok": "kok-IN", "bho": "hi-IN", "mai": "hi-IN",
    "doi": "hi-IN", "awa": "hi-IN", "mwr": "hi-IN", "bgc": "hi-IN",
    "tcy": "kn-IN",  # Tulu uses Kannada script in Sarvam
    "ur": "ur-IN",   # Urdu support
    # Using closest language for regional variants
}

# ── Sarvam speaker settings - Ritu (female) voice for all languages
# ENHANCED: Added expressiveness (pitch variation), pace, loudness for natural human-like speech
# IMPROVED: Updated Tamil, Telugu, Kannada, Malayalam with better settings to enhance clarity and distinctiveness
SARVAM_SPEAKER_MAP = {
    "hi":   {"speaker": "ritu", "pace": 0.95, "pitch": 0.0, "loudness": 1.5, "emotion": "natural"},  # Neutral tone
    "mr":   {"speaker": "ritu", "pace": 0.90, "pitch": 0.25, "loudness": 1.6, "emotion": "expressive"}, # 0.90 pace but expressive for Marathi
    # IMPROVED: Tamil - enhanced for better vowel clarity and melodic flow
    "ta":   {"speaker": "ritu", "pace": 0.80, "pitch": 0.35, "loudness": 1.6, "emotion": "warm"},   # Slower pace for Tamil distinctiveness
    # IMPROVED: Telugu - enhanced for better consonant clarity
    "te":   {"speaker": "ritu", "pace": 0.85, "pitch": 0.25, "loudness": 1.6, "emotion": "natural"},# Natural flow with improved clarity
    # IMPROVED: Kannada - enhanced for better pronunciation
    "kn":   {"speaker": "ritu", "pace": 0.78, "pitch": 0.15, "loudness": 1.6, "emotion": "calm"},   # Slower, clear - Kannada needs careful pronunciation
    # IMPROVED: Malayalam - enhanced for melodic language characteristics
    "ml":   {"speaker": "ritu", "pace": 0.92, "pitch": 0.20, "loudness": 1.6, "emotion": "warm"},  # Melodic, slightly expressive
    "bn":   {"speaker": "ritu", "pace": 0.92, "pitch": 0.1, "loudness": 1.5, "emotion": "natural"},# Neutral
    "gu":   {"speaker": "ritu", "pace": 1.0, "pitch": 0.35, "loudness": 1.6, "emotion": "expressive"},  # Expressive & lively
    "pa":   {"speaker": "ritu", "pace": 1.0, "pitch": 0.25, "loudness": 1.6, "emotion": "energetic"},   # Energetic tone
    "or":   {"speaker": "ritu", "pace": 0.90, "pitch": 0.2, "loudness": 1.5, "emotion": "friendly"},    # Friendly
    "as":   {"speaker": "ritu", "pace": 0.95, "pitch": 0.15, "loudness": 1.5, "emotion": "natural"},    # Natural
    "en":   {"speaker": "ritu", "pace": 1.0, "pitch": 0.0, "loudness": 1.5, "emotion": "professional"}, # Professional
    "si":   {"speaker": "ritu", "pace": 0.95, "pitch": 0.2, "loudness": 1.5, "emotion": "warm"},        # Warm
    "kok":  {"speaker": "ritu", "pace": 0.90, "pitch": 0.25, "loudness": 1.5, "emotion": "expressive"}, # 0.90 pace but expressive Konkani
    "bho":  {"speaker": "ritu", "pace": 0.88, "pitch": 0.3, "loudness": 1.6, "emotion": "expressive"},  # Expressive
    "mai":  {"speaker": "ritu", "pace": 0.92, "pitch": 0.2, "loudness": 1.5, "emotion": "warm"},        # Warm
    "doi":  {"speaker": "ritu", "pace": 0.95, "pitch": 0.15, "loudness": 1.5, "emotion": "friendly"},   # Friendly
    "awa":  {"speaker": "ritu", "pace": 0.92, "pitch": 0.25, "loudness": 1.5, "emotion": "warm"},       # Warm
    "mwr":  {"speaker": "ritu", "pace": 0.95, "pitch": 0.2, "loudness": 1.5, "emotion": "natural"},     # Natural
    "bgc":  {"speaker": "ritu", "pace": 1.0, "pitch": 0.3, "loudness": 1.6, "emotion": "energetic"},    # Energetic
    # NEW: Tulu - similar to Kannada but with slight adjustments for clarity
    "tcy":  {"speaker": "ritu", "pace": 0.80, "pitch": 0.18, "loudness": 1.6, "emotion": "calm"},  # Tulu - clear, calm delivery similar to Kannada
    # IMPROVED: Urdu - with Arabic script considerations
    "ur":   {"speaker": "ritu", "pace": 0.90, "pitch": 0.20, "loudness": 1.6, "emotion": "warm"},       # Warm, expressive for Urdu
}

SARVAM_DEFAULT_SPEAKER_CONFIG = {
    "speaker": "ritu",  # Female voice - Ritu for all
    "pace": 0.95,       # Slightly slower for naturalness
    "pitch": 0.0,       # Neutral base
    "loudness": 1.5,    # Standard volume
    "emotion": "natural",  # Natural, conversational tone
}

GROQ_VOICE_MAP = {
    "en":   "playai-tts",
    "ar":   "playai-tts-arabic",
    "mr":   "playai-tts",       # Use default for non-natively supported
    "hi":   "playai-tts",
    "ta":   "playai-tts",
    "te":   "playai-tts",
    "kn":   "playai-tts",
    "ml":   "playai-tts",
    "tcy":  "playai-tts",  # Tulu - use default
    "ur":   "playai-tts-arabic",  # Urdu - close to Arabic variant
}
GROQ_DEFAULT_VOICE = "playai-tts"



def _elevenlabs_tts(text: str, lang: str, output_path: str) -> str:
    lang = lang.split("-")[0]
    
    # Convert to phonetic for better pronunciation if available
    if PHONETIC_CONVERTER_AVAILABLE and PhoneticConverter.should_use_phonetic(lang):
        text = PhoneticConverter.convert_to_phonetic(text, lang)
        logger.debug(f"[PHONETIC] ElevenLabs: Converted text to phonetic representation")
    
    api_key  = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key: raise RuntimeError("ELEVENLABS_API_KEY missing")
    voice_id = ELEVENLABS_VOICE_MAP.get(lang, ELEVENLABS_DEFAULT_VOICE)
    url      = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload  = {
        "text": text,
        "model_id": ELEVENLABS_MODEL,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        },
    }
    logger.debug(
        "[D3-REQUEST] provider=elevenlabs | voice=%r | model=%r | "
        "lang=%r | text_len=%d | url=%s",
        voice_id, ELEVENLABS_MODEL, lang, len(text), url
    )
    t0 = time.time()
    resp = requests.post(
        url,
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=20,
    )
    latency_ms = int((time.time() - t0) * 1000)
    logger.debug(
        "[D4-RESPONSE] provider=elevenlabs | status=%d | latency=%dms | "
        "content_type=%s",
        resp.status_code, latency_ms, resp.headers.get("Content-Type", "?")
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"[D5-ERROR] elevenlabs {resp.status_code} | "
            f"lang={lang!r} | voice={voice_id!r} | "
            f"body={resp.text[:300]}"
        )
    with open(output_path, "wb") as f:
        f.write(resp.content)
    logger.info(
        "[D6-SUCCESS] provider=elevenlabs | lang=%r | path=%s | "
        "bytes=%d | latency=%dms",
        lang, output_path, len(resp.content), latency_ms
    )
    return output_path

def _sarvam_tts(text: str, lang: str, output_path: str) -> str:
    lang = lang.split("-")[0]
    
    # Convert to phonetic for better pronunciation if available
    if PHONETIC_CONVERTER_AVAILABLE and PhoneticConverter.should_use_phonetic(lang):
        text = PhoneticConverter.convert_to_phonetic(text, lang)
        logger.debug(f"[PHONETIC] Sarvam: Converted text to phonetic representation")
    
    api_key    = os.environ.get("SARVAM_API_KEY")
    if not api_key: raise RuntimeError("SARVAM_API_KEY missing")
    sarvam_lang = SARVAM_LANG_MAP.get(lang)
    if not sarvam_lang:
        raise ValueError(
            f"[D5-ERROR] sarvam: lang={lang!r} not in SARVAM_LANG_MAP"
        )
    
    # Get language-specific speaker configuration (ritu - female voice)
    speaker_config = SARVAM_SPEAKER_MAP.get(lang, SARVAM_DEFAULT_SPEAKER_CONFIG)
    
    payload = {
        "inputs":               [text],
        "target_language_code": sarvam_lang,
        "speaker":              speaker_config["speaker"],
        "pace":                 speaker_config.get("pace", 1.0),
        "enable_preprocessing": True,
        "model":                "bulbul:v3",
    }
    
    # ── BUGFIX: Sarvam Bulbul V3 does NOT support pitch and loudness yet.
    # We must remove these parameters from the payload while keeping the logic
    # intact for when they upgrade their API or if we switch back to V2
    # if "pitch" in speaker_config:
    #     payload["pitch"] = speaker_config["pitch"]
    # 
    # if "loudness" in speaker_config:
    #     payload["loudness"] = speaker_config["loudness"]
    
    logger.debug(
        "[D3-REQUEST] provider=sarvam | lang=%r | sarvam_lang=%r | "
        "speaker=%r | pace=%r | pitch=%r(skipped_for_v3) | loudness=%r(skipped_for_v3) | emotion=%r | text_len=%d",
        lang, sarvam_lang, speaker_config.get("speaker"), speaker_config.get("pace"), 
        speaker_config.get("pitch"), speaker_config.get("loudness"), 
        speaker_config.get("emotion"), len(text)
    )
    t0 = time.time()
    resp = requests.post(
        "https://api.sarvam.ai/text-to-speech",
        headers={
            "api-subscription-key": api_key,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=15,
    )
    latency_ms = int((time.time() - t0) * 1000)
    logger.debug(
        "[D4-RESPONSE] provider=sarvam | status=%d | latency=%dms",
        resp.status_code, latency_ms
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"[D5-ERROR] sarvam {resp.status_code} | "
            f"lang={lang!r} | body={resp.text[:300]}"
        )
    audio_b64 = resp.json()["audios"][0]
    audio_bytes = base64.b64decode(audio_b64)
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    logger.info(
        "[D6-SUCCESS] provider=sarvam | lang=%r | speaker=%r | pace=%r | pitch=%r | loudness=%r | emotion=%r | "
        "path=%s | bytes=%d | latency=%dms",
        lang, speaker_config.get("speaker"), speaker_config.get("pace"), 
        speaker_config.get("pitch"), speaker_config.get("loudness"),
        speaker_config.get("emotion"), output_path, len(audio_bytes), latency_ms
    )
    return output_path

def _groq_tts(text: str, lang: str, output_path: str) -> str:
    lang = lang.split("-")[0]
    
    # Convert to phonetic for better pronunciation if available
    if PHONETIC_CONVERTER_AVAILABLE and PhoneticConverter.should_use_phonetic(lang):
        text = PhoneticConverter.convert_to_phonetic(text, lang)
        logger.debug(f"[PHONETIC] Groq: Converted text to phonetic representation")
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key: raise RuntimeError("GROQ_API_KEY missing")
    voice   = GROQ_VOICE_MAP.get(lang, GROQ_DEFAULT_VOICE)
    payload = {
        "model": voice,
        "input": text,
        "voice": "Celeste-PlayAI",
    }
    logger.debug(
        "[D3-REQUEST] provider=groq | voice=%r | lang=%r | text_len=%d",
        voice, lang, len(text)
    )
    t0 = time.time()
    resp = requests.post(
        "https://api.groq.com/openai/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=20,
    )
    latency_ms = int((time.time() - t0) * 1000)
    logger.debug(
        "[D4-RESPONSE] provider=groq | status=%d | latency=%dms",
        resp.status_code, latency_ms
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"[D5-ERROR] groq {resp.status_code} | "
            f"lang={lang!r} | body={resp.text[:300]}"
        )
    with open(output_path, "wb") as f:
        f.write(resp.content)
    logger.info(
        "[D6-SUCCESS] provider=groq | lang=%r | path=%s | "
        "bytes=%d | latency=%dms",
        lang, output_path, len(resp.content), latency_ms
    )
    return output_path

def _gtts_tts(text: str, lang: str, output_path: str) -> str:
    try:
        from gtts import gTTS
    except ImportError:
        raise RuntimeError("[D5-ERROR] gtts not installed — pip install gtts")
    gtts_lang = lang.split("-")[0]
    
    # Convert to phonetic for better pronunciation if available
    if PHONETIC_CONVERTER_AVAILABLE and PhoneticConverter.should_use_phonetic(lang):
        text = PhoneticConverter.convert_to_phonetic(text, lang)
        logger.debug(f"[PHONETIC] gTTS: Converted text to phonetic representation")
    
    logger.debug(
        "[D3-REQUEST] provider=gtts | lang=%r | gtts_lang=%r | text_len=%d",
        lang, gtts_lang, len(text)
    )
    t0 = time.time()
    tts = gTTS(text=text, lang=gtts_lang, slow=False)
    tts.save(output_path)
    latency_ms = int((time.time() - t0) * 1000)
    import os as _os
    size = _os.path.getsize(output_path)
    logger.info(
        "[D6-SUCCESS] provider=gtts | lang=%r | path=%s | "
        "bytes=%d | latency=%dms",
        lang, output_path, size, latency_ms
    )
    return output_path

def _google_cloud_tts(text: str, lang: str, output_path: str) -> str:
    """Google Cloud Text-to-Speech (free tier available)."""
    try:
        from google.cloud import texttospeech
    except ImportError:
        raise RuntimeError("[D5-ERROR] google-cloud-texttospeech not installed — pip install google-cloud-texttospeech")
    
    lang_code = lang.split("-")[0]
    
    # Convert to phonetic for better pronunciation if available
    if PHONETIC_CONVERTER_AVAILABLE and PhoneticConverter.should_use_phonetic(lang):
        text = PhoneticConverter.convert_to_phonetic(text, lang)
        logger.debug(f"[PHONETIC] Google Cloud: Converted text to phonetic representation")
    
    client = texttospeech.TextToSpeechClient()
    
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    # Language code mapping for Google Cloud
    gc_lang_map = {
        "hi": "hi-IN", "mr": "mr-IN", "ta": "ta-IN", "te": "te-IN",
        "kn": "kn-IN", "ml": "ml-IN", "bn": "bn-IN", "gu": "gu-IN",
        "pa": "pa-IN", "or": "or-IN", "as": "as-IN", "en": "en-US",
        "si": "si-LK",
    }
    
    gc_lang_code = gc_lang_map.get(lang_code, "en-US")
    
    voice = texttospeech.VoiceSelectionParams(
        language_code=gc_lang_code,
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,  # Female voice selection
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
    )
    
    logger.debug(
        "[D3-REQUEST] provider=google_cloud | lang=%r | voice_lang=%r | text_len=%d",
        lang, gc_lang_code, len(text)
    )
    
    t0 = time.time()
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )
    latency_ms = int((time.time() - t0) * 1000)
    
    with open(output_path, "wb") as f:
        f.write(response.audio_content)
    
    logger.info(
        "[D6-SUCCESS] provider=google_cloud | lang=%r | path=%s | "
        "bytes=%d | latency=%dms",
        lang, output_path, len(response.audio_content), latency_ms
    )
    return output_path

_PROVIDER_FNS = {
    "sarvam":       _sarvam_tts,
    "elevenlabs":   _elevenlabs_tts,
    "groq":         _groq_tts,
    "gtts":         _gtts_tts,
    "google_cloud": _google_cloud_tts,
}

# ── Phonetic Analysis Integration & Audio Polishing ────────────────────────
try:
    from awaaz.src.pipeline.phonetics import PhoneticAnalyzer, AccentAdaptationEngine
    _phonetic_analyzer = PhoneticAnalyzer()
    _accent_engine = AccentAdaptationEngine()
    PHONETICS_ENABLED = True
except ImportError:
    logger.debug("Phonetics module not available—continuing without phonetic analysis")
    _phonetic_analyzer = None
    _accent_engine = None
    PHONETICS_ENABLED = False


async def _get_phonetic_analysis(text: str, lang: str) -> Optional[Dict]:
    """Get phonetic analysis for text (async-compatible)."""
    if not PHONETICS_ENABLED or not _phonetic_analyzer:
        return None
    try:
        analysis = await _phonetic_analyzer.analyze_text(text, lang.split("-")[0])
        return analysis
    except Exception as e:
        logger.debug(f"Phonetic analysis failed: {e}")
        return None

def _get_phonetic_analysis_sync(text: str, lang: str) -> Optional[Dict]:
    """Synchronous wrapper for phonetic analysis."""
    if not PHONETICS_ENABLED or not _phonetic_analyzer:
        return None
    try:
        # Try to get analysis directly without async if available
        if hasattr(_phonetic_analyzer, 'analyze_text_sync'):
            return _phonetic_analyzer.analyze_text_sync(text, lang.split("-")[0])
        else:
            # Fallback: analyze_text is async, skip in sync context
            return None
    except Exception as e:
        logger.debug(f"Phonetic analysis sync failed: {e}")
        return None

def synthesize_speech(
    text: str,
    lang: str,
    output_path: str,
    force_provider: str | None = None,
) -> dict:
    """
    Universal TTS with automatic provider fallback and multilingual support.

    Supports 50+ languages across 4 providers:
    - SARVAM (best for Indic languages: Marathi, Hindi, Tamil, Telugu, etc.)
    - ElevenLabs (multilingual, premium quality)
    - Groq TTS (fast, free tier available)
    - gTTS (offline fallback, no API key needed)

    Args:
        text: Text to synthesize (any language/script)
        lang: BCP-47 language code ('mr', 'hi', 'ta', 'en', 'ar', etc.)
        output_path: Where to write the audio file
        force_provider: Skip routing and use this provider only

    Returns:
        dict with keys:
            provider (str | None): which provider succeeded
            path     (str | None): output file path
            duration_s (float):   wall-clock seconds
            error    (str | None): error message if all failed
    """
    native = is_native_script(text, lang)
    order  = [force_provider] if force_provider else get_provider_order(lang)

    # Detailed language support info
    lang_base = lang.split("-")[0]
    supported_scripts = [s for s, (_, langs) in SCRIPT_RANGES.items() if lang_base in langs]
    
    # ── Phonetic Analysis ──────────────────────────────────────────────────
    phonetic_info = None
    if PHONETICS_ENABLED:
        try:
            # Use synchronous wrapper instead of asyncio.run
            phonetic_info = _get_phonetic_analysis_sync(text, lang)
        except Exception as e:
            logger.debug(f"Phonetic analysis setup failed: {e}")
    
    # Simple logging
    logger.info(
        "[TTS-INIT] lang=%r | provider_chain=%s",
        lang, order
    )

    errors = []
    t_start = time.time()

    for provider_name in order:
        fn = _PROVIDER_FNS.get(provider_name)
        if fn is None:
            logger.warning("[TTS] Unknown provider %r — skipping", provider_name)
            continue
        try:
            path = fn(text, lang, output_path)
            result = {
                "provider":   provider_name,
                "path":       path,
                "duration_s": round(time.time() - t_start, 2),
                "error":      None,
            }
            logger.info(f"[TTS-SUCCESS] lang=%r | provider=%s", lang, provider_name)
            return result
        except Exception as exc:
            msg = str(exc)
            errors.append(f"{provider_name}: {msg}")
            logger.warning(
                "[D5-ERROR] provider=%r failed | lang=%r | error=%s | "
                "trying next provider",
                provider_name, lang, msg
            )

    full_error = " | ".join(errors)
    logger.error(
        "[TTS-ALL-FAILED] lang=%r | text_len=%d | errors=%s",
        lang, len(text), full_error
    )
    return {
        "provider":   None,
        "path":       None,
        "duration_s": round(time.time() - t_start, 2),
        "error":      full_error,
    }

class TTSProcessor:
    def __init__(self, preferred_provider: str = None):
        self.preferred_provider = preferred_provider

    async def load(self):
        pass
        
    async def synthesize(self, text: str, session, output_path: str) -> bool:
        lang = getattr(session, "lang", "en")
        try:
            from awaaz.src.pipeline.tts import synthesize_speech
            res = synthesize_speech(text, lang, output_path)
            return res["path"] is not None
        except Exception as e:
            logger.error(f"TTSProcessor.synthesize error: {e}")
            return False

    def synthesize_to_bytes(self, text: str, session) -> bytes | None:
        lang = getattr(session, "lang", "en")
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                out_path = f.name
            res = synthesize_speech(text, lang, out_path)
            if res["path"]:
                with open(res["path"], "rb") as bf:
                    data = bf.read()
                os.unlink(res["path"])
                return data
        except Exception as e:
            logger.error(f"TTSProcessor.synthesize_to_bytes error: {e}")
        return None

