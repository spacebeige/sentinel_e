"""
Phonetics and transliteration utilities for language-aware TTS/STT adaptation.
Handles:
  - Native script transliteration (Roman → Devanagari, etc.)
  - Phonetic IPA conversion
  - Regional accent/dialect normalization
  - Language meaning extraction
  - Debug output for pipeline transparency
"""

import logging
import os
from typing import Optional, Dict, List, Tuple
import aiohttp
import json

logger = logging.getLogger(__name__)


# ── Language-specific IPA & Phonetic Mappings ──────────────────────────────

# Indic language phonetic markers for accent detection
REGIONAL_ACCENT_PATTERNS = {
    "hi": {
        "thick_village": ["mujhe", "tumhe", "kuch", "kya", "hai"],  # Common rural patterns
        "standard": ["mujhe", "tumhe", "kuch", "kya", "hain"],
    },
    "mr": {
        "thick_village": ["mala", "tula", "kay", "aahe", "nay"],  # Marathi rural
        "standard": ["mala", "tula", "kai", "aahe", "nahi"],
    },
    "gu": {
        "thick_village": ["mane", "tane", "shu", "chhe", "naathi"],  # Gujarati rural
        "standard": ["mane", "tane", "shu", "chhe", "nathi"],
    },
    "ta": {
        "thick_village": ["enakku", "unkku", "enna", "irukku", "illa"],  # Tamil rural
        "standard": ["enakku", "unakku", "enna", "irukku", "illai"],
    },
    "te": {
        "thick_village": ["nannu", "ninnu", "enta", "undhi", "nundi"],  # Telugu rural
        "standard": ["nannu", "ninnu", "yenta", "undhi", "nundi"],
    },
    "bn": {
        "thick_village": ["amake", "tumake", "ki", "ache", "nai"],  # Bengali rural
        "standard": ["amar", "tumar", "kya", "ache", "nai"],
    },
}

# Native script mappings for transliteration
NATIVE_SCRIPT_MAPPING = {
    "hi": "Devanagari",
    "mr": "Devanagari",
    "gu": "Gujarati",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "pa": "Gurmukhi",
    "or": "Odia",
    "as": "Bengali",
    "ne": "Devanagari",
    "si": "Sinhala",
}

# Common Indic word meanings (for debugging/understanding)
INDIC_WORD_MEANINGS = {
    "hi": {
        "namaste": "hello/greeting",
        "shukriya": "thank you",
        "haan": "yes",
        "nahi": "no",
        "kya": "what",
        "kahan": "where",
        "kaun": "who",
        "kab": "when",
        "kyun": "why",
    },
    "mr": {
        "namaskar": "hello/greeting",
        "dhanyavaad": "thank you",
        "hoy": "is",
        "nahi": "no",
        "kay": "what",
        "kathe": "where",
        "kon": "who",
        "kahi": "when",
    },
    "gu": {
        "namaste": "hello/greeting",
        "dhanyavaad": "thank you",
        "haan": "yes",
        "nahi": "no",
        "shu": "what",
        "kya": "where",
        "su": "who",
        "kada": "when",
    },
    "ta": {
        "vanakkam": "hello/greeting",
        "nandri": "thank you",
        "aama": "yes",
        "illai": "no",
        "enna": "what",
        "epdi": "how",
        "yaaru": "who",
        "eppo": "when",
    },
    "te": {
        "namaste": "hello/greeting",
        "dhanyavadamulu": "thank you",
        "aa": "yes",
        "ledu": "no",
        "yenta": "what",
        "endi": "where",
        "yaaru": "who",
        "eppudu": "when",
    },
}


class PhoneticAnalyzer:
    """Handles phonetic analysis, transliteration, and accent detection."""

    def __init__(self):
        self.sarvam_api_key = os.getenv("SARVAM_API_KEY", "").strip()
        self.sarvam_transliterate_url = os.getenv("SARVAM_TRANSLITERATE_API_URL", "").strip()
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.groq_translate_url = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1").strip()
        self.enable_phonetic_debug = os.getenv("ENABLE_PHONETIC_DEBUG", "true").lower() in {"true", "1", "yes"}

    async def analyze_text(
        self,
        text: str,
        detected_language: str,
    ) -> Dict:
        """
        Comprehensive phonetic analysis of text.
        
        Returns:
        {
            "original_text": str,
            "language": str,
            "native_script": str,
            "phonetic_ipa": str,
            "english_meaning": str,
            "accent_type": str,  # "standard" or "thick_village"
            "debug_output": str,
        }
        """
        result = {
            "original_text": text,
            "language": detected_language,
            "native_script": NATIVE_SCRIPT_MAPPING.get(detected_language, "Unknown"),
            "phonetic_ipa": "",
            "english_meaning": "",
            "accent_type": "standard",
            "debug_output": "",
        }

        try:
            # Detect accent type
            accent_type = self._detect_accent(text, detected_language)
            result["accent_type"] = accent_type

            # Get English meaning if available
            meaning = self._get_word_meaning(text, detected_language)
            if meaning:
                result["english_meaning"] = meaning

            # Transliterate to native script using Sarvam
            if self.sarvam_api_key and detected_language != "en":
                native_script = await self._transliterate_to_native_script(text, detected_language)
                if native_script:
                    result["native_script"] = native_script

            # Generate IPA phonetic representation
            phonetic_ipa = await self._generate_phonetic_ipa(text, detected_language)
            if phonetic_ipa:
                result["phonetic_ipa"] = phonetic_ipa

            # Build debug output
            if self.enable_phonetic_debug:
                result["debug_output"] = self._format_debug_output(result)

        except Exception as e:
            logger.warning(f"Phonetic analysis error for '{text}' in {detected_language}: {e}")
            result["debug_output"] = f"⚠️ Phonetic analysis partial: {str(e)}"

        return result

    def _detect_accent(self, text: str, lang: str) -> str:
        """Detect if text uses thick village accent or standard pronunciation."""
        if lang not in REGIONAL_ACCENT_PATTERNS:
            return "standard"

        text_lower = text.lower()
        patterns = REGIONAL_ACCENT_PATTERNS[lang]

        # Check thick village patterns
        village_matches = sum(1 for word in patterns.get("thick_village", []) if word in text_lower)
        standard_matches = sum(1 for word in patterns.get("standard", []) if word in text_lower)

        if village_matches > standard_matches:
            return "thick_village"
        return "standard"

    def _get_word_meaning(self, text: str, lang: str) -> Optional[str]:
        """Get English meaning of common words."""
        if lang not in INDIC_WORD_MEANINGS:
            return None

        words = text.lower().split()
        meanings_found = []

        for word in words:
            if word in INDIC_WORD_MEANINGS[lang]:
                meanings_found.append(
                    f"{word}={INDIC_WORD_MEANINGS[lang][word]}"
                )

        return " | ".join(meanings_found) if meanings_found else None

    async def _transliterate_to_native_script(self, text: str, lang: str) -> Optional[str]:
        """Transliterate Roman text to native script using Sarvam API."""
        if not self.sarvam_api_key or not self.sarvam_transliterate_url or lang == "en":
            return None

        try:
            headers = {
                "Authorization": f"Bearer {self.sarvam_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "input": text,
                "source_script": "roman",
                "target_script": NATIVE_SCRIPT_MAPPING.get(lang, "devanagari"),
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.sarvam_transliterate_url,
                    headers=headers,
                    json=payload,
                    timeout=5,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("transliterated_text") or data.get("output", text)
        except Exception as e:
            logger.debug(f"Transliteration error: {e}")

        return None

    async def _generate_phonetic_ipa(self, text: str, lang: str) -> Optional[str]:
        """Generate IPA phonetic representation using language patterns."""
        # Basic IPA patterns for Indian languages
        ipa_patterns = {
            "hi": self._hindi_to_ipa,
            "mr": self._marathi_to_ipa,
            "gu": self._gujarati_to_ipa,
            "ta": self._tamil_to_ipa,
            "te": self._telugu_to_ipa,
            "bn": self._bengali_to_ipa,
        }

        if lang in ipa_patterns:
            return ipa_patterns[lang](text)

        return None

    @staticmethod
    def _hindi_to_ipa(text: str) -> str:
        """Convert Hindi text to approximate IPA."""
        # Simplified IPA mapping for common Hindi phonemes
        mapping = {
            "a": "ə", "aa": "aː", "i": "ɪ", "ii": "iː",
            "u": "ʊ", "uu": "uː", "e": "eː", "o": "oː",
            "ka": "kə", "kha": "kʰə", "ga": "ɡə", "gha": "ɡʰə",
            "cha": "tʃə", "chha": "tʃʰə", "ja": "dʒə",
            "tha": "ʈə", "tha": "ʈʰə", "da": "ɖə",
            "pa": "pə", "pha": "pʰə", "ba": "bə", "bha": "bʰə",
            "ya": "jə", "ra": "ɾə", "la": "lə", "wa": "ʋə",
        }
        result = text.lower()
        for roman, ipa in mapping.items():
            result = result.replace(roman, ipa)
        return result

    @staticmethod
    def _marathi_to_ipa(text: str) -> str:
        """Convert Marathi text to approximate IPA."""
        # Similar to Hindi with some variations
        mapping = {
            "a": "ə", "aa": "aː", "i": "ɪ", "ie": "iː",
            "u": "ʊ", "oo": "uː", "e": "eː", "o": "oː",
            "hy": "ʰ", "ny": "ŋ",
        }
        result = text.lower()
        for roman, ipa in mapping.items():
            result = result.replace(roman, ipa)
        return result

    @staticmethod
    def _gujarati_to_ipa(text: str) -> str:
        """Convert Gujarati text to approximate IPA."""
        mapping = {
            "aa": "aː", "ee": "iː", "oo": "uː",
            "kh": "kʰ", "gh": "ɡʰ", "ph": "pʰ", "bh": "bʰ",
        }
        result = text.lower()
        for roman, ipa in mapping.items():
            result = result.replace(roman, ipa)
        return result

    @staticmethod
    def _tamil_to_ipa(text: str) -> str:
        """Convert Tamil text to approximate IPA."""
        mapping = {
            "aa": "aː", "ee": "iː", "oo": "uː",
            "kk": "kː", "pp": "pː", "tt": "tː",
            "ng": "ŋ", "ny": "ɲ",
        }
        result = text.lower()
        for roman, ipa in mapping.items():
            result = result.replace(roman, ipa)
        return result

    @staticmethod
    def _telugu_to_ipa(text: str) -> str:
        """Convert Telugu text to approximate IPA."""
        mapping = {
            "aa": "aː", "ii": "iː", "uu": "uː",
            "rru": "ɾuː", "llu": "luː",
        }
        result = text.lower()
        for roman, ipa in mapping.items():
            result = result.replace(roman, ipa)
        return result

    @staticmethod
    def _bengali_to_ipa(text: str) -> str:
        """Convert Bengali text to approximate IPA."""
        mapping = {
            "aa": "aː", "ee": "iː", "oo": "uː",
            "rri": "ɾiː", "llt": "lːt",
        }
        result = text.lower()
        for roman, ipa in mapping.items():
            result = result.replace(roman, ipa)
        return result

    def _format_debug_output(self, analysis: Dict) -> str:
        """Format phonetic analysis as readable debug output."""
        lines = [
            f"📝 Original: {analysis['original_text']}",
            f"🌐 Language: {analysis['language'].upper()}",
            f"🔤 Script: {analysis['native_script']}",
            f"🎯 Accent: {analysis['accent_type'].replace('_', ' ').title()}",
        ]

        if analysis["native_script"]:
            lines.append(f"📄 Native: {analysis['native_script']}")

        if analysis["phonetic_ipa"]:
            lines.append(f"🔊 Phonetic IPA: {analysis['phonetic_ipa']}")

        if analysis["english_meaning"]:
            lines.append(f"🇬🇧 English: {analysis['english_meaning']}")

        return " | ".join(lines)


class AccentAdaptationEngine:
    """Adapts STT and TTS behavior based on detected accent/dialect."""

    ACCENT_ADAPTATIONS = {
        "thick_village": {
            "stt_confidence_threshold": 0.65,  # Lower threshold for dialects
            "tts_pace": 0.85,  # Slower for clarity
            "tts_pitch": -5,   # Slightly lower pitch for rural tones
            "extra_silence": 0.2,  # Brief pauses between words
        },
        "standard": {
            "stt_confidence_threshold": 0.80,
            "tts_pace": 0.95,
            "tts_pitch": 0,
            "extra_silence": 0.0,
        },
    }

    def get_stt_parameters(self, accent_type: str) -> Dict:
        """Get STT parameters adapted for accent type."""
        return self.ACCENT_ADAPTATIONS.get(accent_type, self.ACCENT_ADAPTATIONS["standard"])

    def get_tts_parameters(self, accent_type: str, lang: str) -> Dict:
        """Get TTS parameters adapted for accent and language."""
        base = self.ACCENT_ADAPTATIONS.get(accent_type, self.ACCENT_ADAPTATIONS["standard"])
        return {
            "pace": base["tts_pace"],
            "pitch": base["tts_pitch"],
            "extra_silence": base["extra_silence"],
        }
