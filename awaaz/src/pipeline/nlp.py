"""NLP module for AWAAZ - grievance classification and LLM."""

import logging
import json
import re
import os
from typing import Dict, Optional

import aiohttp

from src.pipeline.lang_detect import TokenLevelLangDetector

logger = logging.getLogger(__name__)

# Grievance category mappings
GRIEVANCE_CATEGORIES = {
    "GR-01": "Water Supply",
    "GR-02": "Sanitation & Sewerage",
    "GR-03": "Road Infrastructure",
    "GR-04": "Public Health",
    "GR-05": "Electricity",
    "GR-06": "Education",
    "GR-07": "Documents & Permits",
    "GR-08": "Other",
}

LANGUAGE_CONFIG = {
    "hi": {"name": "Hindi", "gtts": "hi", "script": "Devanagari"},
    "ta": {"name": "Tamil", "gtts": "ta", "script": "Tamil"},
    "te": {"name": "Telugu", "gtts": "te", "script": "Telugu"},
    "kn": {"name": "Kannada", "gtts": "kn", "script": "Kannada"},
    "ml": {"name": "Malayalam", "gtts": "ml", "script": "Malayalam"},
    "mr": {"name": "Marathi", "gtts": "mr", "script": "Devanagari"},
    "gu": {"name": "Gujarati", "gtts": "gu", "script": "Gujarati"},
    "bn": {"name": "Bengali", "gtts": "bn", "script": "Bengali"},
    "pa": {"name": "Punjabi", "gtts": "pa", "script": "Gurmukhi"},
    "or": {"name": "Odia", "gtts": "or", "script": "Odia"},
    "as": {"name": "Assamese", "gtts": "bn", "script": "Bengali"},
    "kok": {"name": "Konkani", "gtts": "mr", "script": "Devanagari"},
    "mai": {"name": "Maithili", "gtts": "hi", "script": "Devanagari"},
    "bho": {"name": "Bhojpuri", "gtts": "hi", "script": "Devanagari"},
    "awa": {"name": "Awadhi", "gtts": "hi", "script": "Devanagari"},
    "bgc": {"name": "Haryanvi", "gtts": "hi", "script": "Devanagari"},
    "doi": {"name": "Dogri", "gtts": "hi", "script": "Devanagari"},
    "mwr": {"name": "Marwadi", "gtts": "hi", "script": "Devanagari"},
    "pah": {"name": "Pahadi", "gtts": "hi", "script": "Devanagari"},
    "ne": {"name": "Nepali", "gtts": "hi", "script": "Devanagari"},
    "sa": {"name": "Sanskrit", "gtts": "hi", "script": "Devanagari"},
    "ks": {"name": "Kashmiri", "gtts": "ur", "script": "Nastaliq"},
    "sd": {"name": "Sindhi", "gtts": "ur", "script": "Arabic"},
    "ur": {"name": "Urdu", "gtts": "ur", "script": "Nastaliq"},
    "en": {"name": "English", "gtts": "en", "script": "Latin"},
    "sat": {"name": "Santali", "gtts": "bn", "script": "Ol Chiki"},
    "brx": {"name": "Bodo", "gtts": "bn", "script": "Devanagari"},
    "mni": {"name": "Manipuri", "gtts": "bn", "script": "Meitei"},
    "tcy": {"name": "Tulu", "gtts": "kn", "script": "Kannada"},
    "hne": {"name": "Chhattisgarhi", "gtts": "hi", "script": "Devanagari"},
    "raj": {"name": "Rajasthani", "gtts": "hi", "script": "Devanagari"},
    "kfy": {"name": "Kumaoni", "gtts": "hi", "script": "Devanagari"},
    "gbm": {"name": "Garhwali", "gtts": "hi", "script": "Devanagari"},
    "kru": {"name": "Kurukh", "gtts": "hi", "script": "Devanagari"},
    "mah": {"name": "Magahi", "gtts": "hi", "script": "Devanagari"},
    "lmn": {"name": "Lambadi", "gtts": "te", "script": "Devanagari"},
    "dcc": {"name": "Dakhini Urdu", "gtts": "ur", "script": "Nastaliq"},
    "saz": {"name": "Saurashtra", "gtts": "gu", "script": "Latin"},
}


def _with_mixed_variants(base_config: dict) -> dict:
    """Generate `<lang>-en` mixed profiles for every non-English language."""
    expanded = dict(base_config)
    for code, cfg in list(base_config.items()):
        if code == "en" or code.endswith("-en"):
            continue
        mixed_code = f"{code}-en"
        if mixed_code in expanded:
            continue
        expanded[mixed_code] = {
            "name": f"{cfg['name']}-English",
            "gtts": cfg["gtts"],
            "script": f"Mixed ({cfg['script']} + Latin)",
        }
    return expanded


LANGUAGE_CONFIG = _with_mixed_variants(LANGUAGE_CONFIG)


class ModelProcessor:
    """LLM-based NLP processor for AWAAZ."""

    def __init__(self, provider: str = "groq", model: str = "llama-3.1-8b-instant"):
        self.provider = provider
        self.model = model
        self.client = None
        self.max_tokens = 150
        self.temperature = 0.2  # LOWERED: Critical for language-only constraint enforcement
        self.lang_detector = TokenLevelLangDetector.get()
        self.sarvam_api_key = os.getenv("SARVAM_API_KEY", "").strip()
        self.sarvam_translate_url = os.getenv("SARVAM_TRANSLATE_API_URL", "").strip()

    async def load(self):
        """Initialize LLM client."""
        try:
            if self.provider == "groq":
                from groq import Groq

                api_key = __import__("os").environ.get("GROQ_API_KEY")
                if not api_key:
                    logger.error("GROQ_API_KEY not set")
                    return
                self.client = Groq(api_key=api_key)
                logger.info(f"Groq client initialized with model {self.model}")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")

    @staticmethod
    def _contains_devanagari(text: str) -> bool:
        return bool(re.search(r"[\u0900-\u097F]", text or ""))

    @staticmethod
    def _contains_script(text: str, script: str) -> bool:
        script_ranges = {
            "Devanagari": r"[\u0900-\u097F]",
            "Tamil": r"[\u0B80-\u0BFF]",
            "Telugu": r"[\u0C00-\u0C7F]",
            "Kannada": r"[\u0C80-\u0CFF]",
            "Malayalam": r"[\u0D00-\u0D7F]",
            "Gujarati": r"[\u0A80-\u0AFF]",
            "Gurmukhi": r"[\u0A00-\u0A7F]",
            "Odia": r"[\u0B00-\u0B7F]",
            "Bengali": r"[\u0980-\u09FF]",
            "Latin": r"[A-Za-z]",
        }
        rx = script_ranges.get(script)
        return bool(rx and re.search(rx, text or ""))

    @staticmethod
    def _base_lang(lang: str) -> str:
        code = (lang or "hi").lower()
        return code.split("-en")[0] if code.endswith("-en") else code

    def _is_reply_language_aligned(self, reply: str, session_lang: str) -> bool:
        """Validate that LLM reply is in the correct language/script.
        
        Checks:
        1. Language detected matches session language
        2. SCRIPT DOMINANCE: 80%+ of text is in correct script (prevents mixed-script corruption)
        3. For English: no Devanagari
        4. For mixed modes: accepts either base lang or English
        """
        lang = (session_lang or "hi").lower()
        target_base = self._base_lang(lang)

        detected_reply_lang, _ = self.lang_detector.detect(reply or "")
        detected_base = self._base_lang(detected_reply_lang)

        if target_base == "en":
            return detected_base == "en" and not self._contains_devanagari(reply)

        # For mixed languages like mr-en, accept reply if it contains EITHER base lang OR English
        if lang.endswith("-en"):
            if detected_base in {target_base, "en"}:
                return True
            lang_cfg = LANGUAGE_CONFIG.get(target_base, {})
            script = (lang_cfg.get("script") or "").split(" ")[0]
            if script and script != "Mixed":
                has_native_script = self._contains_script(reply, script)
                has_english = bool(re.search(r"[A-Za-z]", reply or ""))
                if has_native_script and has_english:
                    return True
            return False
        
        elif detected_base == target_base:
            # IMPROVED: Script dominance check
            # When text has multiple scripts, ensure 80%+ is in correct script
            lang_cfg = LANGUAGE_CONFIG.get(target_base, {})
            script = (lang_cfg.get("script") or "").split(" ")[0]
            if script and script not in {"Mixed", "", "Latin"}:
                script_dominance = self._get_script_dominance(reply, script)
                if script_dominance < 0.80:  # Less than 80% in correct script
                    logger.warning(
                        f"[SCRIPT-ALIGN] Low dominance ({script_dominance:.1%} {script}) "
                        f"in reply: {reply[:50]}... Expected ≥80%"
                    )
                    return False  # Reject low-dominance responses
            return True

        # Script-level backup validation (useful when language detector is uncertain)
        lang_cfg = LANGUAGE_CONFIG.get(target_base, {})
        script = (lang_cfg.get("script") or "").split(" ")[0]
        if script in {"Mixed", ""}:
            return True
        
        # For strong enforcement: check script dominance even in fallback
        script_dominance = self._get_script_dominance(reply, script)
        if script_dominance < 0.60:  # Less than 60% in correct script as fallback
            logger.warning(
                f"[SCRIPT-ALIGN] Fallback low dominance ({script_dominance:.1%} {script})"
            )
            return False
        
        return self._contains_script(reply, script)

    def _get_script_dominance(self, text: str, target_script: str) -> float:
        """Calculate what percentage of non-whitespace chars are in target script.
        
        Returns dominance ratio 0.0-1.0
        Example: If text has 70 chars in Kannada + 30 in Latin, returns 0.7
        """
        if not text:
            return 0.0
        
        script_ranges = {
            "Devanagari": (0x0900, 0x097F),
            "Tamil": (0x0B80, 0x0BFF),
            "Telugu": (0x0C00, 0x0C7F),
            "Kannada": (0x0C80, 0x0CFF),
            "Malayalam": (0x0D00, 0x0D7F),
            "Gujarati": (0x0A80, 0x0AFF),
            "Gurmukhi": (0x0A00, 0x0A7F),
            "Odia": (0x0B00, 0x0B7F),
            "Bengali": (0x0980, 0x09FF),
            "Latin": (0x0041, 0x005A),  # A-Z only for counting purposes
            "Nastaliq": (0x0600, 0x06FF),
        }
        
        if target_script not in script_ranges:
            return 1.0  # Unknown script - allow
        
        start, end = script_ranges[target_script]
        target_count = sum(1 for c in text if start <= ord(c) <= end)
        total_graphemes = len([c for c in text if ord(c) > 32])  # Exclude whitespace
        
        if total_graphemes == 0:
            return 1.0
        
        return target_count / total_graphemes

    def _build_system_prompt(self, session, strict_level: int = 0) -> str:
        """Build system prompt for call state.
        
        strict_level: 0 = normal, 1 = strict language-only, 2 = extreme language-only lock
        """
        # Handle dynamic mixed language keys (e.g. "kok-en", "mai-en") if exact key doesn't exist
        lookup_lang = session.lang
        if lookup_lang not in LANGUAGE_CONFIG and lookup_lang.endswith("-en"):
            base_lang = lookup_lang.split("-en")[0]
            if base_lang in LANGUAGE_CONFIG:
                base_config = LANGUAGE_CONFIG[base_lang]
                lang_name = f"{base_config['name']}-English"
                script = f"Mixed ({base_config['script']} for {base_config['name']}, Latin for English)"
            else:
                lang_name = lookup_lang
                script = "Unknown"
        else:
            lang_config = LANGUAGE_CONFIG.get(lookup_lang, {})
            lang_name = lang_config.get("name", lookup_lang)
            script = lang_config.get("script", "Unknown")
            
        lang_mode = getattr(session, 'lang_mode', 'pure')
        lang_distribution = getattr(session, 'lang_distribution', {})

        base_prompt = f"""You are AWAAZ, a multilingual grievance assistant for Indian citizens.

CALL STATE:
- Language:       {session.lang} ({lang_name})
- Script:         {script}
- Language mode:  {lang_mode} (pure / mixed)
- Distribution:   {lang_distribution}
- Turn:           {session.turn_number}
- State:          {session.state}
- Category:       {session.grievance_category or 'Not yet determined'}

⚠️  CRITICAL LANGUAGE CONSTRAINT (YOU MUST FOLLOW THIS ABSOLUTELY):

YOU WILL RESPOND ONLY IN: {lang_name} ({session.lang})
DO NOT use English.
DO NOT use any other language.
DO NOT explain why you're using {lang_name}.
JUST RESPOND IN {lang_name} ({session.lang}) FOR EVERY WORD.

👩 FEMININE LANGUAGE CONSTRAINT (YOU MUST FOLLOW THIS):
Always respond using FEMININE LANGUAGE FORMS regardless of the caller's gender.
This assistant (AWAAZ) is female presenting and uses feminine grammatical forms.

Language-specific feminine forms:
- Hindi/Marathi/Devanagari: Use -ूँ, -और, -हू, -में (feminine verb endings)
- Tamil/Telugu/Kannada/Malayalam: Use feminine grammatical markers
- Gujarati: Use -ી, -ો endings (feminine forms)
- Punjabi: Use feminine honorifics and verb forms (ਮੈ, ਮੈਨੂ)
- Bengali: Use feminine forms (-ি, -ু, feminine pronouns)
- English: Use "I" and neutral/professional tone
- All languages: Maintain professional feminine politeness

LANGUAGE-SPECIFIC RULES:

1. ABSOLUTE RULE: Reply ONLY in {session.lang} ({lang_name})
   - Bhojpuri input → Bhojpuri reply ONLY
   - Chhattisgarhi input → Chhattisgarhi reply ONLY
   - Hinglish input → Hinglish reply at same code-mix ratio ONLY
   - Roman Hindi → Roman Hindi ONLY
   - Devanagari → Devanagari ONLY
   - Hindi → Hindi (Devanagari) ONLY
   - English → English ONLY
   - Tamil → Tamil ONLY
   - Telugu → Telugu ONLY
   - Kannada → Kannada ONLY
   - Malayalam → Malayalam ONLY
   - Marathi → Marathi ONLY
   - Bengali → Bengali ONLY
   - Punjabi → Punjabi (Gurmukhi) ONLY
   - Gujarati → Gujarati ONLY
   - All others → Reply ONLY in the detected language/script

2. Preserve the caller's tone:
   - If informal: stay informal
   - If formal: stay formal

3. Use FEMININE LANGUAGE FORMS:
   - All responses must be in feminine grammatical forms for your language
   - This creates consistency with the female voice and feminine presentation

4. Format output for TTS:
   - Expand: "9876543210" → use digit words in {lang_name}
   - Expand: "TKT-2047" → spell it out in {lang_name}
   - No markdown, JSON, or special formatting

5. JSON MODE (for GATHERING/CONFIRMING only):
   Return: {{"reply": "YOUR_RESPONSE_IN_{lang_name}_FEMININE", "meta": {{...}}}}

6. PLAIN TEXT MODE (for other states):
   Return: Just the response in {lang_name} (feminine forms), nothing else

"""

        if strict_level >= 1:
            base_prompt += f"\n🔒 STRICT MODE: Every single word of your response MUST be in {lang_name} ({session.lang}). No exceptions."
        
        if strict_level >= 2:
            base_prompt += f"\n🔐 EXTREME LOCK: RESPOND ONLY IN {lang_name.upper()} ({session.lang.upper()}). THIS IS NON-NEGOTIABLE."

        state_instructions = {
            "GREETING": f"Greet the caller warmly in {lang_name}. Ask how you can help. Use {lang_name} ONLY.",
            "GATHERING": f"Gather details about their grievance. Ask follow-up questions ONLY in {lang_name}.",
            "CONFIRMING": f"Confirm the details the caller provided. Summarize ONLY in {lang_name}.",
            "FILING": f"Create a ticket for this grievance. Thank the caller ONLY in {lang_name}.",
        }
        
        state_instr = state_instructions.get(session.state, "")
        if state_instr:
            return base_prompt + f"\n\n{state_instr}"
        return base_prompt

    async def _sarvam_translate_to_target(self, text: str, session_lang: str) -> Optional[str]:
        """Optional hard fallback: translate reply to session language via Sarvam."""
        if not text:
            return None

        target_lang = self._base_lang(session_lang or "hi")
        if target_lang == "en":
            return text

        if not self.sarvam_api_key or not self.sarvam_translate_url:
            return None

        headers = {
            "Authorization": f"Bearer {self.sarvam_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "source_language": "auto",
            "target_language": target_lang,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.sarvam_translate_url,
                    headers=headers,
                    json=payload,
                    timeout=10,
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

            return (
                result.get("translated_text")
                or result.get("output")
                or result.get("text")
                or (result.get("data") or {}).get("translated_text")
                or (result.get("data") or {}).get("output")
                or None
            )
        except Exception as e:
            logger.debug(f"Sarvam translation fallback failed: {e}")
            return None

    async def generate(self, user_input: str, session) -> str:
        """Generate LLM response with aggressive language enforcement."""
        if not self.client:
            logger.error("LLM client not initialized")
            return "Please try again."

        try:
            session_lang = getattr(session, "lang", "hi")
            lang_cfg = LANGUAGE_CONFIG.get(session_lang.split("-")[0], {})
            lang_name = lang_cfg.get("name", session_lang)
            
            # Try with normal prompt first (strict_level=0)
            system_prompt = self._build_system_prompt(session, strict_level=0)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                max_tokens=self.max_tokens,
                temperature=0.1,  # VERY LOW for language constraint
            )
            reply = response.choices[0].message.content.strip()
            logger.debug(f"[Gen 1/3] Generated reply ({session_lang}): {reply[:60]}...")

            # Check language alignment
            if not self._is_reply_language_aligned(reply, session_lang):
                logger.warning(f"[Gen 1/3] Language mismatch ({session_lang}). Retrying with strict mode...")
                
                # Retry with strict prompt (strict_level=1)
                strict_system_prompt = self._build_system_prompt(session, strict_level=1)
                retry = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": strict_system_prompt},
                        {"role": "user", "content": user_input},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.05,  # EVEN LOWER
                )
                retry_reply = retry.choices[0].message.content.strip()
                logger.debug(f"[Gen 2/3] Retry reply ({session_lang}): {retry_reply[:60]}...")
                
                if self._is_reply_language_aligned(retry_reply, session_lang):
                    reply = retry_reply
                    logger.info(f"[Gen 2/3] ✓ Language aligned after retry")
                else:
                    # Final attempt: EXTREME mode
                    logger.warning(f"[Gen 2/3] Still misaligned ({session_lang}). Final attempt with extreme lock...")
                    extreme_system_prompt = self._build_system_prompt(session, strict_level=2)
                    final = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": extreme_system_prompt},
                            {"role": "user", "content": user_input},
                        ],
                        max_tokens=self.max_tokens,
                        temperature=0.0,  # ABSOLUTE ZERO
                    )
                    final_reply = final.choices[0].message.content.strip()
                    logger.debug(f"[Gen 3/3] Final reply ({session_lang}): {final_reply[:60]}...")
                    
                    if self._is_reply_language_aligned(final_reply, session_lang):
                        reply = final_reply
                        logger.info(f"[Gen 3/3] ✓ Language aligned on final attempt")
                    else:
                        # Hard fallback: translate to target language (if Sarvam translate endpoint configured)
                        translated_reply = await self._sarvam_translate_to_target(final_reply, session_lang)
                        if translated_reply and self._is_reply_language_aligned(translated_reply, session_lang):
                            reply = translated_reply
                            logger.info(f"[Gen 3/3] ✓ Language aligned via Sarvam translation fallback")
                        else:
                            # Last fallback: Use final reply anyway (model tried its best)
                            logger.error(f"[Gen 3/3] ✗ Could not enforce {lang_name} ({session_lang}) after 3 attempts. Using final reply.")
                            reply = final_reply

            logger.debug(f"LLM reply: {reply[:100]}")
            return reply
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "I'm having trouble understanding. Can you please repeat?"


def check_emergency(text: str, lang: str, model_processor: ModelProcessor) -> bool:
    """Check if text indicates an emergency using LLM + language-specific keywords.
    
    Supports all 32+ Indian languages with native emergency keywords.
    """
    if not text or not text.strip():
        return False

    # Comprehensive emergency keywords in all supported Indian languages
    EMERGENCY_KEYWORDS = {
        "hi": ["मदद", "आपातकाल", "बचाओ", "जल्दी", "डाक्टर", "पुलिस", "आग", "दुर्घटना", "चोट", "खतरा"],
        "mr": ["मदद", "आपत्काल", "वाचवा", "लवकर", "डॉक्टर", "पोलिस", "आग", "अपघात", "दुखापल", "संकट"],
        "ta": ["உதவி", "அவசரம்", "காப்பாற்று", "வேகம்", "மருத்துவர்", "போலீஸ்", "தீ", "விபத்து", "காயம்", "அபாயம்"],
        "te": ["సహాయం", "అత్యవసర", "రక్షణ", "వేగం", "డాక్టర్", "పోలీసు", "불", "ప్రమాదం", "గాయం", "ప్రమాదం"],
        "kn": ["ಸಹಾಯ", "ತುರ್ತು", "ರಕ್ಷಿಸಿ", "ವೇಗ", "ವೈದ್ಯ", "ಪೋಲೀಸು", "ಬೆಂೆ", "ಅಪಘಾತ", "ಗಾಯ", "ಅಪಾಯ"],
        "ml": ["സഹായം", "അടിയന്തര", "രക്ഷിക്കുക", "വേഗം", "ഡോക്ടർ", "പോലീസ്", "തീ", "ബുദ്ധിമുട്ട്", "പരിക്ക്", "അപായം"],
        "gu": ["મદદ", "આપાત", "બચાવો", "ઝડપ", "ડૉક્ટર", "પોલીસ", "આગ", "અકસ્માત", "ઇજા", "જોખમ"],
        "bn": ["সাহায্য", "জরুরি", "বাঁচান", "দ্রুত", "ডাক্তার", "পুলিশ", "আগুন", "দুর্ঘটনা", "আঘাত", "বিপদ"],
        "pa": ["ਸਹਾਇਤਾ", "ਜਰੂਰੀ", "ਬਚਾਓ", "ਤੇਜ", "ਡਾਕਟਰ", "ਪੁਲਿਸ", "ਅੱਗ", "ਹਾਦਸਾ", "ਜ਼ਖਮ", "ਖ਼ਤਰਾ"],
        "or": ["ସାହାଯ୍ୟ", "ଜରୁରୀ", "ବଞ୍ଚାନ", "ଦ୍ରୁତ", "ଡାକ୍ତର", "ପୋଲିସ", "ଆଗୁଣ", "ଦୁର୍ଘଟନା", "ଅଶୁଭ", "ବିପଦ"],
        "as": ["সহায়তা", "জরুরি", "বাঁচান", "দ্রুত", "ডাক্তার", "পুলিশ", "আগুন", "দুর্ঘটনা", "আঘাত", "বিপদ"],
        
        # Regional Devanagari languages
        "bho": ["मदद", "आपातकाल", "बचावो", "लव्कर", "डाक्टर", "पलिस", "आग", "दुर्गटना", "चोट"],
        "awa": ["मदद", "आपातकाल", "बचाओ", "लव्कर", "डाक्टर", "पलिस", "आग", "दुर्घटना"],
        "mai": ["मदद", "आपातकाल", "बचाओ", "छैट", "डाक्टर", "पलिस", "आग"],
        "bgc": ["मदद", "आपातकाल", "बचाओ", "लव्का", "डाक्टर", "पलिस"],
        "doi": ["मदद", "आपातकालीन", "बचाना", "जल्दी", "डाक्टर"],
        "mwr": ["मदद", "आपातकाल", "बचाओ", "जल्दी", "डाक्टर"],
        "pah": ["मदद", "आपातकाल", "बचाओ", "जल्दी"],
        "ne": ["सहायता", "आपातकाल", "बचाउ", "छिट्टै", "डाक्टर"],
        "kok": ["मदद", "आपातकाल", "बचावो", "जल्दी"],
        "hne": ["मदद", "आपातकाल", "बचाओ", "लव्का"],
        "raj": ["मदद", "आपातकाल", "बचाओ", "जल्दी"],
        "kfy": ["मदद", "आपातकाल", "बचाओ"],
        "gbm": ["मदद", "आपातकाल", "बचाओ"],
        
        # Santali, Bodo, Manipuri
        "sat": ["मदद", "बचाओ", "आपातकाल"],
        "brx": ["मदद", "बचाओ", "आपातकाल"],
        "mni": ["मदद", "बचाओ", "आपातकाल"],
        
        # Kashmiri, Urdu, Sindhi
        "ks": ["مدد", "بچاؤ", "ڈاکٹر"],
        "ur": ["مدد", "ایمرجنسی", "بچاؤ", "پولیس"],
        "sd": ["مدد", "بچاؤ", "ڈاکٹر"],
        
        # English
        "en": ["help", "emergency", "ambulance", "police", "fire", "accident", "injury", "danger"],
    }

    # Check language-specific keywords (fast local check)
    keywords = EMERGENCY_KEYWORDS.get(lang, [])
    text_lower = text.lower()
    for keyword in keywords:
        if keyword.lower() in text_lower:
            logger.warning(f"Emergency keyword detected ({lang}): {keyword} in '{text}'")
            return True

    # If no local keywords matched but model_processor available, use LLM as backup
    if model_processor and model_processor.client:
        prompt = (
            "Does this message describe an active emergency requiring "
            "immediate government or medical help?\n"
            "Consider ALL languages including regional Indian ones.\n"
            "Answer only YES or NO.\n"
            f"Language: {lang}\n"
            f"Message: {text}"
        )
        try:
            response = model_processor.client.chat.completions.create(
                model=model_processor.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5,
            )
            answer = response.choices[0].message.content.strip().upper()
            if "YES" in answer:
                logger.warning(f"Emergency detected via LLM ({lang}): {text}")
                return True
        except Exception as e:
            logger.debug(f"LLM emergency check failed: {e}, using keyword-based detection")
    
    return False


def update_session_from_nlp(result: dict, session):
    """Write NLP results back to session."""
    if result.get("grievance_category"):
        session.grievance_category = result["grievance_category"]
    if result.get("dept_assigned"):
        session.dept_assigned = result["dept_assigned"]
    if result.get("priority"):
        session.priority = result["priority"]
    if result.get("complaint_summary"):
        session.complaint_summary = result["complaint_summary"]


def parse_llm_output(raw_reply: str) -> tuple:
    """Parse LLM output - may be JSON or plain text."""
    try:
        data = json.loads(raw_reply)
        reply = data.get("reply", raw_reply)
        meta = data.get("meta", {})
        return (reply, meta)
    except (json.JSONDecodeError, ValueError):
        # Treat as plain text
        return (raw_reply, {})
