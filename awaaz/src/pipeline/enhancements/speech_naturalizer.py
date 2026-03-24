"""Speech Naturalizer Module.

Adds phonetic-friendly transforms and pause markers so TTS sounds less robotic.
"""
import re

PAUSE_SHORT = " [P:200] "
PAUSE_MEDIUM = " [P:400] "
PAUSE_LONG = " [P:700] "
_PHONETIC_MAP = {
    r"\bok\b": "okay",
    r"\bpls\b|\bplz\b": "please",
    r"\bmsg\b": "message",
    r"\bgovt\b": "government",
    r"\bdept\b": "department",
    r"\binfo\b": "information",
    r"\basap\b": "as soon as possible",
    r"\bmins\b": "minutes",
    r"\bsec\b": "seconds",
    r"\bETA\b": "E T A",
    r"\bAI\b": "A I",
    r"\bIVR\b": "I V R",
    r"\bOTP\b": "O T P",
    r"\bUPI\b": "U P I",
}

_LANG_PHONETIC_MAP = {
    "hi": {
        r"\bkripya\b": "kripaya",
        r"\bjaldi\b": "jaldi se",
    },
    "mr": {
        r"\bkrupaya\b": "krupayaa",
        r"\blavkar\b": "lavkarat lavkar",
    },
    "ta": {
        r"\bromba\b": "romba",
        r"\bseri\b": "sari",
    },
    "te": {
        r"\bdayachesi\b": "daya chesi",
        r"\bventane\b": "ventane",
    },
    "kn": {
        r"\bdayavittu\b": "daya vittu",
        r"\bbega\b": "beega",
    },
    "ml": {
        r"\bdayavaayi\b": "dayavaayi",
        r"\bvegam\b": "vegamayi",
    },
    "gu": {
        r"\bkrupaa\b": "krupa kari",
        r"\bjaldi\b": "jaldi thi",
    },
    "bn": {
        r"\bonugroho\b": "onugroho kore",
        r"\btaratari\b": "tora tari",
    },
    "pa": {
        r"\bkirpa\b": "kirpa karke",
        r"\bjaldi\b": "jaldi naal",
    },
    "or": {
        r"\bdayakari\b": "daya kari",
    },
    "as": {
        r"\banugraha\b": "anugraha kori",
    },
    "kok": {
        r"\bporim\b": "porim kore",
    },
    "mai": {
        r"\bkripya\b": "kripya kari",
    },
    "bho": {
        r"\bkripya\b": "kripya karke",
    },
    "awa": {
        r"\bkripya\b": "kripya kari",
    },
    "bgc": {
        r"\bjaldi\b": "jaldi tai",
    },
    "doi": {
        r"\bkripya\b": "kirpa kari",
    },
    "mwr": {
        r"\bghano\b": "ghano sa",
    },
    "pah": {
        r"\bthoda\b": "thoda ji",
    },
    "en": {
        r"\bgonna\b": "going to",
        r"\bwanna\b": "want to",
        r"\bgotta\b": "got to",
    },
}

class SpeechNaturalizer:
    def __init__(self, groq_client=None):
        self.groq_client = groq_client

    def _structural_naturalize(self, text: str) -> str:
        """Fallback naturalization using regex if LLM is unavailable or skipped."""
        for pattern, repl in _PHONETIC_MAP.items():
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

        # Make decimal/currency pronunciations smoother for TTS
        text = re.sub(r"\bRs\.?\s*(\d+)\b", r"rupees \1", text, flags=re.IGNORECASE)
        text = re.sub(r"\$(\d+)", r"\1 dollars", text)
        text = re.sub(r"(\d+)%(?!\d)", r"\1 percent", text)
        text = re.sub(r"(\d+)\.(\d+)", r"\1 point \2", text)

        # Replace clause terminators with medium pauses
        text = re.sub(r'([।\.!?])', r'\1' + PAUSE_MEDIUM, text)
        # Replace clause pauses with short pauses (avoid ':' to protect [P:N] markers)
        text = re.sub(r'([,;])', r'\1' + PAUSE_SHORT, text)
        return re.sub(r'\s+', ' ', text).strip()

    def _apply_lang_phonetics(self, text: str, lang: str) -> str:
        base_lang = (lang or "hi").split("-")[0].lower()
        lang_map = _LANG_PHONETIC_MAP.get(base_lang, {})
        for pattern, repl in lang_map.items():
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        return text

    def naturalize(self, text: str, lang: str, lang_name: str, lang_mode: str, 
                   formality_label: str, accent_region: str) -> str:
        """Naturalizes text for human-like spoken delivery."""
        if len(text.split()) <= 3:
            return text
        text = self._apply_lang_phonetics(text, lang)
        return self._structural_naturalize(text)

_global_naturalizer = None

def get_naturalizer(groq_client=None) -> SpeechNaturalizer:
    global _global_naturalizer
    if not _global_naturalizer:
        _global_naturalizer = SpeechNaturalizer(groq_client=groq_client)
    return _global_naturalizer

def naturalize_text(text: str, lang: str = "hi", lang_name: str = "Hindi", 
                   lang_mode: str = "pure", formality_label: str = "SIMPLE", 
                   accent_region: str = "hi-UP-rural", groq_client=None) -> str:
    n = get_naturalizer(groq_client)
    return n.naturalize(text, lang, lang_name, lang_mode, formality_label, accent_region)


def apply_naturalization(text: str, lang: str = "hi", **kwargs) -> str:
    """Compatibility alias used by older enhancement pipeline imports."""
    return naturalize_text(text, lang=lang, **kwargs)
