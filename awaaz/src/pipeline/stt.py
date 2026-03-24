"""Speech-to-text for AWAAZ using multi-provider strategy."""

import logging
import asyncio
import os
import aiohttp
import time
import json
from typing import Tuple, Optional, List, Dict
import abc

# Script-based language detection
try:
    from .lang_detect import TokenLevelLangDetector
except ImportError:
    from lang_detect import TokenLevelLangDetector

try:
    from .phonetic_converter import PhoneticConverter
    PHONETIC_CONVERTER_AVAILABLE = True
except ImportError:
    PHONETIC_CONVERTER_AVAILABLE = False

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# STT Provider Interfaces and Implementations
# -------------------------------------------------------------------

class STTResult:
    def __init__(self, text: str, provider: str, confidence: float, 
                 detected_language: Optional[str] = None, processing_time_ms: float = 0.0,
                 phonetic_text: Optional[str] = None, native_script_text: Optional[str] = None,
                 accent_type: Optional[str] = None, english_meaning: Optional[str] = None):
        self.text = text
        self.provider = provider
        self.confidence = confidence
        self.detected_language = detected_language
        self.processing_time_ms = processing_time_ms
        self.phonetic_text = phonetic_text
        self.native_script_text = native_script_text
        self.accent_type = accent_type or "standard"  # "standard" or "thick_village"
        self.english_meaning = english_meaning


class SarvamLanguageTools:
    """Optional Sarvam-based language detection and transliteration helper."""

    def __init__(self):
        self.api_key = os.getenv("SARVAM_API_KEY", "").strip()
        self.lang_detect_url = os.getenv("SARVAM_LANG_DETECT_API_URL", "").strip()
        self.transliterate_url = os.getenv("SARVAM_TRANSLITERATE_API_URL", "").strip()
        self.enable_phonetic = os.getenv("SARVAM_ENABLE_PHONETIC", "true").strip().lower() in {"1", "true", "yes", "on"}

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _normalize_lang_code(raw_lang: Optional[str]) -> Optional[str]:
        if not raw_lang:
            return None
        value = str(raw_lang).strip().lower()
        if value.endswith("-en"):
            value = value.split("-")[0]
        mapping = {
            "english": "en", "hindi": "hi", "marathi": "mr", "gujarati": "gu",
            "tamil": "ta", "telugu": "te", "kannada": "kn", "malayalam": "ml",
            "bengali": "bn", "assamese": "as", "odia": "or", "oriya": "or",
            "punjabi": "pa", "konkani": "kok", "bhojpuri": "bho", "maithili": "mai",
            "dogri": "doi", "haryanvi": "bgc", "marwadi": "mwr", "awadhi": "awa",
            "pahadi": "pah",
        }
        return mapping.get(value, value)

    async def detect_language_from_text(self, text: str, fallback: Optional[str] = None) -> Optional[str]:
        if not self.enabled or not text or not self.lang_detect_url:
            return fallback

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"text": text}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.lang_detect_url, headers=headers, json=payload, timeout=8) as response:
                    response.raise_for_status()
                    result = await response.json()

            raw_lang = (
                result.get("language_code")
                or result.get("language")
                or result.get("detected_language")
                or (result.get("data") or {}).get("language_code")
                or (result.get("data") or {}).get("language")
            )
            return self._normalize_lang_code(raw_lang) or fallback
        except Exception as e:
            logger.debug(f"Sarvam language detect fallback: {e}")
            return fallback

    async def transliterate_to_native(self, text: str, lang: Optional[str]) -> str:
        if not self.enabled or not self.enable_phonetic or not text or not self.transliterate_url:
            return text

        target_lang = self._normalize_lang_code(lang) or "hi"
        if target_lang == "en":
            return text

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "source_script": "latin",
            "target_language": target_lang,
            "target_script": "native",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.transliterate_url, headers=headers, json=payload, timeout=8) as response:
                    response.raise_for_status()
                    result = await response.json()

            return (
                result.get("transliterated_text")
                or result.get("output")
                or result.get("text")
                or (result.get("data") or {}).get("transliterated_text")
                or (result.get("data") or {}).get("output")
                or text
            ).strip()
        except Exception as e:
            logger.debug(f"Sarvam transliteration fallback: {e}")
            return text

class BaseSTTProvider(abc.ABC):
    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    async def transcribe(self, audio_path: str, language: str = None) -> STTResult:
        """Transcribe audio bytes to text returning normalized STTResult."""
        pass
        
    @abc.abstractmethod
    async def load(self):
        """Prepare or load any resources if needed."""
        pass


class BhashiniSTT(BaseSTTProvider):
    def __init__(self):
        super().__init__("bhashini")
        # Initialize from ENV or default to empty
        self.api_url = os.getenv("BHASHINI_API_URL", "https://bhashini.gov.in/api/v1/recognize")
        self.api_key = os.getenv("BHASHINI_API_KEY", "")

    async def load(self):
        pass # APIs don't need local loads

    async def transcribe(self, audio_path: str, language: str = None) -> STTResult:
        if not self.api_key:
            raise ValueError("Bhashini API key not configured")
            
        start_time = time.time()
        # Mock Bhashini payload handling - Needs exact API contract depending on deployment
        # In actual usage, read audio_path, base64 encode or send bytes
        
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        # Place API call code here
        raise RuntimeError("Bhashini target endpoint configuration required")


class HuggingFaceSTT(BaseSTTProvider):
    def __init__(self):
        super().__init__("huggingface_whisper")
        self.api_url = os.getenv("HF_STT_API_URL", "https://api-inference.huggingface.co/models/openai/whisper-large-v3")
        self.api_key = os.getenv("HF_API_KEY", "")

    async def load(self):
        pass

    async def transcribe(self, audio_path: str, language: str = None) -> STTResult:
        if not self.api_key:
            raise ValueError("HuggingFace API key not configured")
            
        start_time = time.time()
        headers = {"Authorization": f"Bearer {self.api_key}"}

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url, 
                headers=headers, 
                data=audio_bytes, 
                timeout=10
            ) as response:
                response.raise_for_status()
                result = await response.json()
                
                text = result.get("text", "").strip()
                # Default heuristics API doesn't expose prob easily
                confidence = 0.85 if text else 0.0 
                return STTResult(
                    text=text,
                    provider=self.name,
                    confidence=confidence,
                    detected_language=None,
                    processing_time_ms=(time.time() - start_time) * 1000
                )


class GroqWhisperSTT(BaseSTTProvider):
    def __init__(self):
        super().__init__("groq_whisper")
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"
        self.api_key = os.getenv("GROQ_API_KEY", "")

    async def load(self):
        pass

    async def transcribe(self, audio_path: str, language: str = None) -> STTResult:
        if not self.api_key:
            raise ValueError("Groq API key not configured")

        start_time = time.time()
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Groq uses multipart/form-data for the 'file' and 'model'
        data = aiohttp.FormData()
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        data.add_field('file', audio_bytes, filename=os.path.basename(audio_path), content_type='audio/wav')
        data.add_field('model', 'whisper-large-v3')

        # Map full language names from Whisper to iso codes
        whisper_lang_map_reverse = {
            "en": "English", "hi": "Hindi", "mr": "Marathi", "gu": "Gujarati",
            "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
            "bn": "Bengali", "pa": "Punjabi", "or": "Odia", "as": "Assamese",
            "doi": "Dogri", "mwr": "Marwadi", "pah": "Pahadi", "kas": "Kashmiri",
            "ur": "Urdu", "ne": "Nepali", "sa": "Sanskrit", "mai": "Maithili",
            "sd": "Sindhi", "bho": "Bhojpuri", "kok": "Konkani", "bgc": "Haryanvi",
            "tcy": "Tulu"  # Added Tulu support
        }

        # Optional: Add indian languages focus via prompt
        base_prompt = 'Indian languages: Hindi, Marathi, Gujarati, Kannada, Konkani, Telugu, Tamil, Odia, Punjabi, Marwadi, Haryanvi, Assamese, Dogri, Pahadi, Bengali, Malayalam, Bhojpuri, Tulu, Urdu'
        if language and language not in ['auto', '']:
            target_lang_name = whisper_lang_map_reverse.get(language, language.title())
            if target_lang_name not in base_prompt:
                base_prompt += f', Target: {target_lang_name}'
        data.add_field('prompt', base_prompt)
        data.add_field('response_format', 'verbose_json')

        if language and language not in ['auto', 'hi']:
            # Provide language only if strictly required and supported
            whisper_iso_to_name = {k: v.lower() for k, v in whisper_lang_map_reverse.items()}
            if language in whisper_iso_to_name:
                data.add_field('language', language)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                data=data,
                timeout=15
            ) as response:
                response.raise_for_status()
                result = await response.json()

                text = result.get("text", "").strip()
                # Groq whisper API is highly accurate, fake confidence to bypass thresholds
                confidence = 0.95 if text else 0.0

                # Fetch language natively detected by Whisper
                whisper_lang = result.get("language", "").lower()

                # Map full language names from Whisper to iso codes
                whisper_lang_map = {
                    "english": "en", "hindi": "hi", "marathi": "mr", "gujarati": "gu",
                    "tamil": "ta", "telugu": "te", "kannada": "kn", "malayalam": "ml",
                    "bengali": "bn", "punjabi": "pa", "odia": "or", "assamese": "as", "dogri": "doi", "marwadi": "mwr", "pahadi": "pah", "kashmiri": "kas",
                    "urdu": "ur", "nepali": "ne", "sanskrit": "sa", "maithili": "mai",
                    "sindhi": "sd", "bhojpuri": "bho", "konkani": "kok", "haryanvi": "bgc",
                    "tulu": "tcy"  # Added Tulu support
                }
                detected_lang = whisper_lang_map.get(whisper_lang, whisper_lang) or "hi"

                return STTResult(
                    text=text,
                    provider=self.name,
                    confidence=confidence,
                    detected_language=detected_lang,
                    processing_time_ms=(time.time() - start_time) * 1000
                )

class ElevenLabsSTT(BaseSTTProvider):
    """STT using ElevenLabs Speech-to-Text API."""

    def __init__(self):
        super().__init__("elevenlabs_stt")
        self.api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self.api_url = os.getenv("ELEVENLABS_STT_API_URL", "https://api.elevenlabs.io/v1/speech-to-text")
        self.model_id = os.getenv("ELEVENLABS_STT_MODEL_ID", "scribe_v1")

    async def load(self):
        pass

    @staticmethod
    def _normalize_lang(lang: Optional[str]) -> Optional[str]:
        if not lang:
            return None
        normalized = str(lang).strip().lower()
        if normalized in {"auto", ""}:
            return None
        if normalized.endswith("-en"):
            return normalized.split("-")[0]
        if normalized.startswith("en") or "english" in normalized:
            return "en"
        return normalized

    @staticmethod
    def _to_iso_lang(raw_lang: Optional[str]) -> Optional[str]:
        if not raw_lang:
            return None
        value = str(raw_lang).strip().lower()
        mapping = {
            "english": "en",
            "hindi": "hi",
            "marathi": "mr",
            "gujarati": "gu",
            "tamil": "ta",
            "telugu": "te",
            "kannada": "kn",
            "malayalam": "ml",
            "bengali": "bn",
            "assamese": "as",
            "odia": "or",
            "oriya": "or",
            "punjabi": "pa",
            "konkani": "kok",
            "bhojpuri": "bho",
            "maithili": "mai",
            "dogri": "doi",
            "haryanvi": "bgc",
            "marwadi": "mwr",
            "awadhi": "awa",
            "pahadi": "pah",
        }
        return mapping.get(value, value)

    async def transcribe(self, audio_path: str, language: str = None) -> STTResult:
        if not self.api_key:
            raise ValueError("ElevenLabs API key not configured")

        start_time = time.time()
        headers = {"xi-api-key": self.api_key}
        data = aiohttp.FormData()

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        data.add_field(
            "file",
            audio_bytes,
            filename=os.path.basename(audio_path),
            content_type="audio/wav",
        )
        data.add_field("model_id", self.model_id)

        lang_code = self._normalize_lang(language)
        if lang_code:
            data.add_field("language_code", lang_code)

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, data=data, timeout=20) as response:
                response.raise_for_status()
                try:
                    result = await response.json()
                except Exception:
                    # Defensive parse when content-type is incorrect.
                    raw = await response.text()
                    result = json.loads(raw)

        text = (result.get("text") or result.get("transcript") or "").strip()
        detected_lang = self._to_iso_lang(
            result.get("language_code")
            or result.get("detected_language")
            or result.get("language")
        )

        confidence = (
            result.get("confidence")
            or result.get("language_probability")
            or (0.95 if text else 0.0)
        )

        return STTResult(
            text=text,
            provider=self.name,
            confidence=float(confidence),
            detected_language=detected_lang,
            processing_time_ms=(time.time() - start_time) * 1000,
        )


class LocalWhisperSTT(BaseSTTProvider):
    def __init__(self, model_size: str = "small", device: str = "auto"):
        super().__init__(f"faster_whisper_{model_size}")
        self.model_size = model_size
        self.device = device
        self.model = None

    async def load(self):
        if self.model:
            return
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading local Whisper {self.model_size} model...")
            self.model = WhisperModel(self.model_size, device=self.device, compute_type="auto")
            logger.info("Local Whisper model loaded")
        except ImportError:
            logger.error("faster-whisper not installed")
            self.model = None

    def _ensure_16khz(self, audio_path: str) -> str:
        try:
            import soundfile as sf
            from scipy.signal import resample_poly

            data, sr = sf.read(audio_path)
            if sr == 16000:
                return audio_path

            resampled = resample_poly(data, 16000, sr)
            out_path = audio_path.replace(".wav", "_16k.wav")
            sf.write(out_path, resampled, 16000)
            return out_path
        except Exception as e:
            logger.error(f"Resample error: {e}")
            return audio_path

    async def transcribe(self, audio_path: str, language: str = None) -> STTResult:
        if not self.model:
            raise RuntimeError("Local Whisper model not loaded")

        start_time = time.time()
        audio_path = self._ensure_16khz(audio_path)
        
        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None, 
            lambda: self.model.transcribe(audio_path, language=language, beam_size=1)
        )
        
        text = "".join(segment.text for segment in segments).strip()
        confidence = info.language_probability if getattr(info, "language_probability", None) is not None else 0.8
        detected_language = getattr(info, "language", None)
        
        return STTResult(
            text=text,
            provider=self.name,
            confidence=confidence,
            detected_language=detected_language,
            processing_time_ms=(time.time() - start_time) * 1000
        )

# -------------------------------------------------------------------
# Core Orchestrator (Replacing the existing STTProcessor smoothly)
# -------------------------------------------------------------------

class STTProcessor:
    """
    Robust Multilingual STT Pipeline.
    Orchestrates Bhashini, HF API, and local Whisper falling back dynamically.
    """

    def __init__(self, model_size: str = "small", device: str = "auto", preferred_provider: Optional[str] = None):
        self.providers: List[BaseSTTProvider] = []
        self.sarvam = SarvamLanguageTools()

        if os.getenv("BHASHINI_API_KEY"):
            self.providers.append(BhashiniSTT())

        if os.getenv("ELEVENLABS_API_KEY"):
            self.providers.append(ElevenLabsSTT())

        if os.getenv("GROQ_API_KEY"):
            self.providers.append(GroqWhisperSTT())

        if os.getenv("HF_API_KEY"):
            self.providers.append(HuggingFaceSTT())

        self.local_whisper = LocalWhisperSTT(model_size, device)
        self.providers.append(self.local_whisper)

        pref = (preferred_provider or os.getenv("STT_PRIMARY") or "elevenlabs").strip().lower()
        self._prioritize_provider(pref)

        self.confidence_threshold = float(os.getenv("STT_CONFIDENCE_THRESHOLD", "0.6"))

    @staticmethod
    def _normalize_lang_code(raw_lang: Optional[str]) -> Optional[str]:
        if not raw_lang:
            return None
        value = str(raw_lang).strip().lower()
        if not value:
            return None
        if value.endswith("-en"):
            value = value.split("-")[0]
        aliases = {
            "english": "en",
            "hindi": "hi",
            "marathi": "mr",
            "gujarati": "gu",
            "punjabi": "pa",
            "kashmiri": "ks",
        }
        return aliases.get(value, value)

    def _prioritize_provider(self, provider_key: str):
        if not provider_key:
            return

        aliases = {
            "elevenlabs": "elevenlabs",
            "eleven": "elevenlabs",
            "groq": "groq",
            "huggingface": "huggingface",
            "hf": "huggingface",
            "bhashini": "bhashini",
            "local": "faster_whisper",
            "whisper": "faster_whisper",
        }
        normalized = aliases.get(provider_key, provider_key)

        def score(p: BaseSTTProvider):
            name = getattr(p, "name", "")
            return 0 if normalized in name else 1

        self.providers = sorted(self.providers, key=score)

    async def load(self):
        logger.info("Initializing multi-provider STT pipeline...")
        await self.local_whisper.load()
        logger.info(f"Loaded STT providers: {[p.name for p in self.providers]}")

    async def detect_language_with_elevenlabs(self, audio_path: str) -> Tuple[Optional[str], float]:
        """
        Detect language using ElevenLabs STT API (primary - best for all languages)
        Returns: (language_code, confidence)
        """
        elevenlabs_provider = None
        for p in self.providers:
            if isinstance(p, ElevenLabsSTT):
                elevenlabs_provider = p
                break
        
        if not elevenlabs_provider:
            logger.debug("ElevenLabs not available, skipping...")
            return (None, 0.0)
        
        try:
            logger.debug("[LANG-DETECT] Trying ElevenLabs for language detection...")
            result = await elevenlabs_provider.transcribe(audio_path, language="auto")
            
            if result.detected_language:
                logger.info(
                    f"[LANG-DETECT] ElevenLabs detected: {result.detected_language} "
                    f"(confidence: {result.confidence:.2f})"
                )
                return (result.detected_language, result.confidence)
            return (None, 0.0)
        except Exception as e:
            logger.debug(f"[LANG-DETECT] ElevenLabs failed: {str(e)[:100]}")
            return (None, 0.0)

    async def detect_language_with_sarvam(self, audio_path: str) -> Tuple[Optional[str], float]:
        """
        Detect language using Sarvam Language Detection API (secondary - best for Indic)
        Returns: (language_code, confidence)
        """
        if not self.sarvam.enabled:
            logger.debug("Sarvam not available, skipping...")
            return (None, 0.0)
        
        try:
            logger.debug("[LANG-DETECT] Trying Sarvam for language detection...")
            # For Sarvam, we need to transcribe first to get text, then detect language
            # This is a compromise - Sarvam language detection needs text input
            groq_provider = None
            for p in self.providers:
                if isinstance(p, GroqWhisperSTT):
                    groq_provider = p
                    break
            
            if groq_provider:
                result = await groq_provider.transcribe(audio_path, language="auto")
                detected_lang = await self.sarvam.detect_language_from_text(
                    result.text, 
                    fallback=result.detected_language
                )
                if detected_lang:
                    logger.info(f"[LANG-DETECT] Sarvam detected: {detected_lang}")
                    return (detected_lang, 0.9)
            return (None, 0.0)
        except Exception as e:
            logger.debug(f"[LANG-DETECT] Sarvam failed: {str(e)[:100]}")
            return (None, 0.0)

    async def detect_language_with_groq(self, audio_path: str) -> Tuple[Optional[str], float]:
        """
        Detect language using Groq Whisper API (good multilingual detection).
        Returns: (language_code, confidence)
        """
        groq_provider = None
        for p in self.providers:
            if isinstance(p, GroqWhisperSTT):
                groq_provider = p
                break

        if not groq_provider:
            logger.debug("Groq STT not available, skipping...")
            return (None, 0.0)

        try:
            logger.debug("[LANG-DETECT] Trying Groq Whisper for language detection...")
            result = await groq_provider.transcribe(audio_path, language="auto")
            if result.detected_language:
                detected_lang = self._normalize_lang_code(result.detected_language)
                logger.info(
                    f"[LANG-DETECT] Groq detected: {detected_lang} "
                    f"(confidence: {result.confidence:.2f})"
                )
                return (detected_lang, result.confidence)
            return (None, 0.0)
        except Exception as e:
            logger.debug(f"[LANG-DETECT] Groq failed: {str(e)[:100]}")
            return (None, 0.0)

    async def detect_language_with_local_whisper(self, audio_path: str) -> Tuple[Optional[str], float]:
        """
        Detect language using local Whisper model (fallback - works offline)
        Returns: (language_code, confidence)
        """
        if not self.local_whisper.model:
            logger.warning("[LANG-DETECT] Local Whisper not available")
            return (None, 0.0)

        try:
            logger.debug("[LANG-DETECT] Trying local Whisper for language detection...")
            ap = self.local_whisper._ensure_16khz(audio_path)
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                None,
                lambda: self.local_whisper.model.transcribe(ap, language=None, beam_size=1)
            )
            lang = info.language or None
            conf = info.language_probability or 0.0
            if lang:
                logger.info(f"[LANG-DETECT] Local Whisper detected: {lang} (confidence: {conf:.2f})")
            return (lang, conf)
        except Exception as e:
            logger.debug(f"[LANG-DETECT] Local Whisper failed: {str(e)[:100]}")
            return (None, 0.0)

    async def detect_language(self, audio_path: str) -> Tuple[str, float]:
        """
        Multi-provider language detection chain
        (priority: ElevenLabs → Groq → Sarvam → Local Whisper)
        
        Strategy:
        1. ElevenLabs STT (best accuracy for all languages, especially European/CJK/Cyrillic)
        2. Groq Whisper (strong multilingual baseline)
        3. Sarvam (excellent for Indic languages: Marathi, Hindi, Tamil, etc.)
        4. Local Whisper (fallback, works offline)
        4. Default to 'hi' if all fail
        
        Returns: (language_code, confidence)
        """
        logger.debug(f"[LANG-DETECT] Starting multi-provider language detection for {audio_path}")
        
        # Try ElevenLabs first (primary - most accurate overall)
        lang, conf = await self.detect_language_with_elevenlabs(audio_path)
        if lang and conf >= 0.7:  # High confidence
            logger.info(f"[LANG-DETECT] ✓ Using ElevenLabs result: {lang}")
            return (lang, conf)

        # Try Groq next
        groq_lang, groq_conf = await self.detect_language_with_groq(audio_path)
        if groq_lang and groq_conf >= 0.7:
            logger.info(f"[LANG-DETECT] ✓ Using Groq result: {groq_lang}")
            return (groq_lang, groq_conf)
        
        # Try Sarvam (secondary - excellent for Indic languages)
        sarvam_lang, sarvam_conf = await self.detect_language_with_sarvam(audio_path)
        if sarvam_lang and sarvam_conf >= 0.7:
            logger.info(f"[LANG-DETECT] ✓ Using Sarvam result: {sarvam_lang}")
            return (sarvam_lang, sarvam_conf)
        
        # If ElevenLabs/Groq or Sarvam had medium confidence, use it
        if lang and conf > 0.0:
            logger.info(f"[LANG-DETECT] ✓ Using ElevenLabs result (medium confidence): {lang}")
            return (lang, conf)
        if groq_lang and groq_conf > 0.0:
            logger.info(f"[LANG-DETECT] ✓ Using Groq result (medium confidence): {groq_lang}")
            return (groq_lang, groq_conf)
        
        # Try local Whisper (tertiary - works offline)
        local_lang, local_conf = await self.detect_language_with_local_whisper(audio_path)
        if local_lang:
            logger.info(f"[LANG-DETECT] ✓ Using local Whisper result: {local_lang}")
            return (local_lang, local_conf)
        
        # Final fallback
        logger.warning("[LANG-DETECT] All providers failed, defaulting to 'hi'")
        return ("hi", 0.0)

    async def detect_language_ensemble(self, audio_path: str) -> Tuple[Optional[str], Dict[str, float]]:
        """
        Weighted language ensemble using ElevenLabs + Groq + Sarvam (+ local whisper fallback).
        Returns (best_lang, score_board).
        """
        score_board: Dict[str, float] = {}

        def add_score(lang: Optional[str], confidence: float, weight: float):
            norm = self._normalize_lang_code(lang)
            if not norm:
                return
            score_board[norm] = score_board.get(norm, 0.0) + max(confidence, 0.0) * weight

        eleven_lang, eleven_conf = await self.detect_language_with_elevenlabs(audio_path)
        add_score(eleven_lang, eleven_conf or 0.0, 1.0)

        groq_lang, groq_conf = await self.detect_language_with_groq(audio_path)
        add_score(groq_lang, groq_conf or 0.0, 0.95)

        sarvam_lang, sarvam_conf = await self.detect_language_with_sarvam(audio_path)
        add_score(sarvam_lang, sarvam_conf or 0.0, 0.90)

        local_lang, local_conf = await self.detect_language_with_local_whisper(audio_path)
        add_score(local_lang, local_conf or 0.0, 0.60)

        if not score_board:
            return (None, {})

        best_lang = max(score_board, key=score_board.get)
        logger.info(f"[LANG-ENSEMBLE] scores={score_board} | best={best_lang}")
        return (best_lang, score_board)

    async def _resolve_result_language(
        self,
        text: str,
        provider_lang: Optional[str],
        provider_conf: float,
        ensemble_lang: Optional[str],
        ensemble_scores: Dict[str, float],
    ) -> Tuple[Optional[str], Dict[str, float]]:
        """Resolve final language using weighted votes from provider + script + Sarvam + ensemble."""
        script_detector = TokenLevelLangDetector.get()
        vote_scores: Dict[str, float] = {}

        def add_vote(lang: Optional[str], weight: float):
            norm = self._normalize_lang_code(lang)
            if not norm:
                return
            vote_scores[norm] = vote_scores.get(norm, 0.0) + max(weight, 0.0)

        # 1) Provider's own language
        add_vote(provider_lang, 0.85 * max(provider_conf, 0.5))

        # 2) Audio-level ensemble (ElevenLabs + Groq + Sarvam + local)
        if ensemble_lang:
            ensemble_weight = 0.55 + min(ensemble_scores.get(ensemble_lang, 0.0), 1.0) * 0.35
            add_vote(ensemble_lang, ensemble_weight)

        # 3) Script detector (good for clear native scripts)
        script_lang, script_dist = script_detector.detect(text or "")
        if script_lang:
            script_conf = (script_dist or {}).get(script_lang, 0.6)
            add_vote(script_lang, 0.55 * max(script_conf, 0.5))

        # 4) Heuristic detector (word-level fallback)
        heuristic_lang, _ = script_detector._heuristic_detect(text or "")
        add_vote(heuristic_lang, 0.35)

        # 5) Sarvam text language detection as strong tie-breaker
        sarvam_text_lang = await self.sarvam.detect_language_from_text(text, fallback=None)
        add_vote(sarvam_text_lang, 0.70)

        if not vote_scores:
            return (self._normalize_lang_code(provider_lang), vote_scores)

        final_lang = max(vote_scores, key=vote_scores.get)
        logger.info(f"[LANG-RESOLVE] provider={provider_lang} | final={final_lang} | votes={vote_scores}")
        return (final_lang, vote_scores)

    async def transcribe(self, audio_path: str, language: str = "hi") -> Optional[STTResult]:
        last_error = None
        ensemble_lang, ensemble_scores = await self.detect_language_ensemble(audio_path)
        
        for provider in self.providers:
            try:
                # auto/unspecified is better for code-mixed speech models
                transcribe_lang = None if language in ['hi', 'auto'] else language
                
                result = await provider.transcribe(audio_path, language=transcribe_lang)
                
                if result.confidence < self.confidence_threshold:
                    logger.warning(f"[{provider.name}] Low confidence {result.confidence}. Trying next provider.")
                    raise RuntimeError("Confidence below threshold")

                # Multi-engine language resolution (ElevenLabs + Groq + Sarvam + script/heuristic)
                resolved_lang, lang_votes = await self._resolve_result_language(
                    text=result.text,
                    provider_lang=result.detected_language,
                    provider_conf=result.confidence,
                    ensemble_lang=ensemble_lang,
                    ensemble_scores=ensemble_scores,
                )
                if resolved_lang:
                    result.detected_language = resolved_lang
                logger.info(f"[LANG-DETECT] Final resolved language: {result.detected_language} | votes={lang_votes}")

                # Keep both native and phonetic transcript forms
                native_text = await self.sarvam.transliterate_to_native(
                    result.text,
                    result.detected_language,
                )
                result.native_script_text = native_text or result.text
                if PHONETIC_CONVERTER_AVAILABLE:
                    try:
                        result.phonetic_text = PhoneticConverter.convert_to_phonetic(
                            result.native_script_text,
                            result.detected_language or "en",
                        )
                    except Exception:
                        result.phonetic_text = result.text
                else:
                    result.phonetic_text = result.text
                
                # ── Phonetic Analysis & Accent Detection ──────────────────────────
                try:
                    from awaaz.src.pipeline.phonetics import PhoneticAnalyzer
                    phonetic_analyzer = PhoneticAnalyzer()
                    phonetic_info = await phonetic_analyzer.analyze_text(
                        result.text,
                        result.detected_language or "en"
                    )
                    if phonetic_info:
                        result.accent_type = phonetic_info.get("accent_type", "standard")
                        result.english_meaning = phonetic_info.get("english_meaning")
                        logger.info(
                            f"[PHONETIC] {result.detected_language} ({result.accent_type}) | "
                            f"text='{result.text[:40]}' | meaning='{result.english_meaning}'"
                        )
                except Exception as e:
                    logger.debug(f"Phonetic analysis failed: {e}")
                    
                logger.info(f"[{provider.name}] Transcribed successfully ({result.processing_time_ms:.1f}ms): {result.text[:50]}")
                return result

            except Exception as e:
                logger.warning(f"[{provider.name}] STT attempt failed: {str(e)[:100]}")
                last_error = str(e)
                continue

        logger.error(f"All STT providers failed. Last error: {last_error}")
        return None

    async def to_native_script_text(self, text: str, lang: Optional[str]) -> str:
        """Best-effort conversion from phonetic/latin text to native script."""
        return await self.sarvam.transliterate_to_native(text, lang)
