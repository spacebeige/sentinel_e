"""Language detection for AWAAZ using fastText with script-based routing."""

import logging
import os
import re
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class TokenLevelLangDetector:
    """Detects language at token level using fastText with script-based routing - singleton pattern."""

    _instance = None
    _model = None

    # Language markers for heuristic detection and disambiguation
    LANGUAGE_MARKERS = {
        "hi": {"native": ["है", "को", "में", "दिल्ली", "भारत", "मदद", "नहीं", "यह", "और", "कि", "कर", "से", "लिए", "था", "थी", "थे", "मेरा", "मेरी", "क्या", "कहाँ", "कौन", "कैसे", "मैं", "तुम", "आप", "हम", "हैं", "हूँ"], "latin": ["hai", "ko", "mein", "delhi", "bharat", "madad", "nahi", "yah"]},
        "mr": {"native": ["आहे", "ला", "मुंबई", "महाराष्ट्र", "साहय", "नाही", "हा", "आणि", "पण", "मी", "तू", "काय", "कुठे", "कसा", "कसे", "खूप", "आपण", "माझा", "माझी", "माझे", "सांग", "करा", "झालं", "होता", "होती", "होते", "हे", "ते", "का", "तुमचे", "आमचे", "करून", "द्या", "येतो", "येते", "विचार", "मध्ये", "चा", "ची", "चे"], "latin": ["ahe", "la", "mumbai", "maharashtra", "sahy", "nahi", "ha", "aani", "ani", "pan", "mi", "tu", "kay", "kuthe", "kasa", "kase", "khup", "aapan", "mazha", "mazi", "maze", "sang", "kara", "zala", "hota", "hoti", "hote", "he", "te", "ka", "tumche", "aamche", "karun", "dya", "yeto", "yete", "vichar", "madhye", "cha", "chi", "che", "aahe", "nahin", "proverb"]},
        "gu": {"native": ["છે", "ને", "અમદાવાદ", "ગુજરાત", "મદદ", "નહીં", "આ", "અને", "પણ", "હું", "તમે", "શું", "ક્યાં", "કેવી", "કેમ", "મારું", "મારો", "મારી", "કરવું", "થયું", "હતો", "હતી", "હતું"], "latin": ["che", "ne", "ahmedabad", "gujaraat", "madad", "nahi", "aa", "kem", "cho", "chho", "majama", "su", "kya", "pan", "kem cho"]},
        "pa": {"native": ["ਹੈ", "ਨੂੰ", "ਅੰਮ੍ਰਿਤਸਰ", "ਪੰਜਾਬ", "ਮਦਦ", "ਨਹੀਂ", "ਇਹ", "ਅਤੇ", "ਪਰ", "ਮੈਂ", "ਤੁਸੀਂ", "ਕੀ", "ਕਿੱਥੇ", "ਕਿਵੇਂ", "ਮੇਰਾ", "ਮੇਰੀ", "ਕਰਨਾ", "ਹੋਇਆ", "ਸੀ", "ਸਨ", "ਹਾਂ", "ਹਨ", "ਕਿਉਂ"], "latin": ["hai", "noon", "amritsar", "punjab", "madad", "nahi", "eh"]},
        # IMPROVED: Enhanced Tamil detection with more unique markers
        "ta": {"native": ["உள்ளது", "க்கு", "சென்னை", "தமிழ்நாடு", "உதவி", "இல்லை", "இது", "மற்றும்", "ஆனால்", "நான்", "நீங்கள்", "என்ன", "எங்கே", "எப்படி", "ஏன்", "என்", "எனது", "செய்", "ஆனது", "இருந்தது", "ஆம்", "ற்", "ய்", "ள்", "ணை", "ணी", "ணும்", "ணிலை"], "latin": ["ullathu", "kku", "chennai", "tamilnadu", "udhavi", "illai", "ithu"]},
        # IMPROVED: Enhanced Telugu detection with more unique markers
        "te": {"native": ["ఉంది", "కు", "హైదరాబాద్", "తెలంగాణ", "సహాయ", "లేదు", "ఇది", "మరియు", "కానీ", "నేను", "మీరు", "ఏమిటి", "ఎక్కడ", "ఎలా", "ఎందుకు", "నా", "నాది", "చేయి", "అయింది", "ఉండేది", "అవును", "ే్", "ై", "ూ", "ృ", "ాలు"], "latin": ["undi", "ku", "hyderabad", "telangana", "sahay", "ledu", "idi"]},
        # IMPROVED: Enhanced Kannada detection with more unique markers
        "kn": {"native": ["ಇದೆ", "ಗೆ", "ಬೆಂಗಳೂರು", "ಕರ್ನಾಟಕ", "ಸಹಾಯ", "ಇಲ್ಲ", "ಇದು", "ಮತ್ತು", "ಆದರೆ", "ನಾನು", "ನೀವು", "ಏನು", "ಎಲ್ಲಿ", "ಹೇಗೆ", "ಯಾಕೆ", "ನನ್ನ", "ಮಾಡು", "ಆಯಿತು", "ಇತ್ತು", "ಹೌದು", "್ಯ", "ೆ", "ಣ", "ೃ", "ೀ"], "latin": ["ide", "ge", "bangalore", "karnataka", "sahay", "illa", "idu"]},
        # IMPROVED: Enhanced Malayalam detection with distinctive markers
        "ml": {"native": ["ഉണ്ടാകുന്നു", "ക്കു", "കോച്ചി", "കേരളം", "സഹായം", "ഇല്ല", "ഇത്", "കൂടാതെ", "പക്ഷേ", "ഞാൻ", "നിങ്ങൾ", "എന്ത്", "എവിടെ", "എങ്ങനെ", "എന്തുകൊണ്ട്", "എന്റെ", "ചെയ്യുക", "ആയി", "ഉണ്ടായിരുന്നു", "അതെ", "അല്ല", "്ര", "്റ", "െ", "ോ", "ൌ"], "latin": ["undakunnu", "kku", "kochi", "kerala", "sahayam", "illa", "ith"]},
        # NEW: Tulu (tcy) detection markers
        "tcy": {"native": ["ಇದೆ", "ಗೆ", "ಅಪೋ", "ಮಂಗಳೂರು", "ತುಳು", "ಎಡೂ", "ಈಡೀ", "ಬಾರು", "ಚಿಕ್ಕ", "ಕೆಲಸ", "ಕೊಡು", "ತಾರಿ", "ನೋಕು", "ಶರಿ"], "latin": ["ide", "ge", "appo", "mangalore", "tulu", "edu", "baru"]},
        # IMPROVED: Enhanced Urdu detection with Arabic script markers
        "ur": {"native": ["ہے", "کو", "لاہور", "پاکستان", "مدد", "نہیں", "یہ", "اور", "لیکن", "میں", "تم", "آپ", "کیا", "کہاں", "کیسے", "کیوں", "میرا", "کرو", "ہوا", "تھا", "ہاں", "ھ", "ی", "ٹ", "ں", "گ"], "latin": ["hai", "ko", "lahore", "pakistan", "madad", "nahi", "ye"]},
        "bn": {"native": ["আছে", "কে", "ঢাকা", "বাংলাদেশ", "সাহায্য", "নেই", "এটি", "এবং", "কিন্তু", "আমি", "তুমি", "আপনি", "কী", "কোথায়", "কেমন", "কেন", "আমার", "করুন", "হয়েছে", "ছিল", "হ্যাঁ", "না"], "latin": ["ache", "ke", "dhaka", "bangladesh", "sahajjya", "nei", "eti"]},
        "or": {"native": ["ଅଛି", "କୁ", "ଭୁବନେଶ୍ୱର", "ଓଡିଶା", "ସାହାଯ୍ୟ", "ନାହିଁ", "ଏହା", "ଏବଂ", "କିନ୍ତୁ", "ମୁଁ", "ତୁମେ", "ଆପଣ", "କଣ", "କେଉଁଠାରେ", "କିପରି", "କାହିଁକି", "ମୋର", "କରନ୍ତୁ", "ହେଲା", "ଥିଲା", "ହଁ", "ନା"], "latin": ["achi", "ku", "bhubaneswar", "odisha", "sahajjya", "nahi", "eha"]},
        "sa": {"native": ["अस्ति", "कस्य", "नमस्ते", "संस्कृतम्", "सहायता", "नास्ति", "एतत्", "च", "किन्तु", "अहम्", "त्वम्", "भवान्", "किम्", "कुत्र", "कथम्", "किमर्थम्", "मम", "करोतु", "आसीत्", "आम", "न"], "latin": ["asti", "kasya", "namaste", "sanskrit", "sahayata", "nasti", "etat"]},
        "mai": {"native": ["छै", "कें", "दरभंगा", "मिथिला", "मदद", "नइखे", "एह", "अछि", "हम", "अहाँ", "कियैक", "आ", "मुदा", "की", "कतय", "केना", "हमर", "करू", "भेल", "छल", "हँ", "नै"], "latin": ["chai", "ken", "darbhanga", "mithila", "madad", "naikhe", "eh"]},
        "bho": {"native": ["है", "को", "वाराणसी", "भोजपुर", "मदद", "नहीं", "यह", "बा", "रउवा", "तोहरा", "हमनी", "काहे", "होखे", "ई", "उ", "हौ", "आउर", "बाकिर", "हम", "का", "कहाँ", "कइसे", "हमार", "करं", "भइल", "रहल", "हाँ", "ना"], "latin": ["hai", "ko", "varanasi", "bhojpur", "madad", "nahi", "yah"]},
        "kok": {"native": ["आसा", "ला", "गोवा", "कोंकणी", "साहय", "नाही", "हो", "आणि", "पूण", "हाव", "तू", "कितें", "खंय", "कसो", "कशी", "खुब्ब", "अमी", "म्हाजो", "म्हजी", "सांग", "कर", "जाले", "आशिल्लो", "आशिल्ली", "हें", "तें", "कित्याक", "तुमचें", "आमचें", "ना", "आसात", "दिय", "येता"], "latin": ["asa", "la", "goa", "konkani", "sahy", "nahi", "ho"]},
        "ks": {"native": ["آ", "کے", "شرینگر", "کشمیر", "مدد", "نیست", "ہے", "تہ", "مگر", "بہ", "ژہ", "کیاہ", "کتی", "کیتھ", "کیٛازِ", "میون", "کر", "گوو", "اوس", "نہ"], "latin": ["aa", "ke", "srinagar", "kashmir", "madad", "nist", "hai"]},
        "ur": {"native": ["ہے", "کو", "لاہور", "پاکستان", "مدد", "نہیں", "یہ", "اور", "لیکن", "میں", "تم", "آپ", "کیا", "کہاں", "کیسے", "کیوں", "میرا", "کرو", "ہوا", "تھا", "ہاں"], "latin": ["hai", "ko", "lahore", "pakistan", "madad", "nahi", "ye"]},
        "en": {"native": [], "latin": ["is", "the", "problem", "electricity", "help", "and", "but", "i", "you", "what", "where", "how", "why", "my", "do", "did", "was", "yes", "no", "are", "am", "he", "she", "it", "they", "we", "hello", "hi", "thanks", "thanks", "ok", "okay"]},
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get(cls):
        """Get singleton instance."""
        return cls()

    def load(self, model_path: str = None):
        """Load fastText model (call once at startup)."""
        if self._model:
            return

        if not model_path:
            model_path = os.environ.get("FASTTEXT_MODEL_PATH", "/tmp/lid.176.bin")

        try:
            import fasttext
            logger.info(f"Loading fastText model from {model_path}")
            self._model = fasttext.load_model(model_path)
            logger.info("fastText model loaded")
        except Exception as e:
            logger.error(f"Failed to load fastText: {e}")

    def detect(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Detect language distribution in text.
        Priority: 1) Script-based detection  2) FastText  3) Heuristic fallback

        Returns: (primary_lang, distribution_dict)
        Example: ('hi', {'hi': 0.8, 'en': 0.2})
        """
        if not text:
            return ("hi", {"hi": 1.0})

        # Step 1: Try script-based detection first (PRIMARY METHOD)
        script_lang = self._detect_by_script(text)
        if script_lang:
            return (script_lang, {script_lang: 0.95})

        # Step 2: Try fastText if available
        if self._model:
            try:
                predictions = self._model.predict(text.replace("\n", " "), k=5)
                labels = predictions[0]
                scores = predictions[1]
                dist = {label.replace("__label__", ""): score for label, score in zip(labels, scores)}
                primary_lang = list(dist.keys())[0] if dist else "hi"
                logger.debug(f"FastText distribution: {dist}")
                return (primary_lang, dist)
            except Exception as e:
                logger.warning(f"FastText detection failed: {e}, falling back to heuristic")

        # Step 3: Heuristic fallback
        return self._heuristic_detect(text)

    def _detect_by_script(self, text: str) -> Optional[str]:
        """Detect language primarily by Unicode script ranges (PRIMARY method).
        For MIXED scripts, returns the DOMINANT script (most char count).
        IMPROVED: Better differentiation for Tamil, Telugu, Kannada, Malayalam, Tulu."""
        if not text:
            return None

        # IMPROVED: More granular script detection for South Indian languages
        # Each script has a unique range and distinctive combination of characters
        script_scores = {
            "ta": (0x0B80, 0x0BFF, "Tamil"),              # Tamil script (distinct vowels: ு, ி, ா)
            "te": (0x0C00, 0x0C7F, "Telugu"),             # Telugu script (distinct: ె, ీ, ూ, ీ)
            "kn": (0x0C80, 0x0CFF, "Kannada"),            # Kannada script (distinct: ೆ, ೇ, ೀ)
            "ml": (0x0D00, 0x0D7F, "Malayalam"),          # Malayalam script (distinct: െ, േ, ോ, ൌ)
            "tcy": (0x0C80, 0x0CFF, "Telugu/Kannada*"),   # Tulu uses Kannada script but needs special handling
            "pa": (0x0A00, 0x0A7F, "Gurmukhi (Punjabi)"),
            "gu": (0x0A80, 0x0AFF, "Gujarati"),
            "or": (0x0B00, 0x0B7F, "Oriya/Odia"),
            "bn": (0x0980, 0x09FF, "Bengali"),
            "sat": (0x1950, 0x197F, "Ol Chiki (Santali)"),
            "ur_ks": (0x0600, 0x06FF, "Arabic/Urdu/Kashmiri"),
            "hi_devanagari": (0x0900, 0x097F, "Devanagari"),
        }

        # Count occurrences of each script
        script_counts = {}
        for lang_key, (start, end, name) in script_scores.items():
            count = self._count_script_chars(text, start, end)
            if count > 0:
                script_counts[lang_key] = count

        if not script_counts:
            return None

        # Find dominant script (most character count)
        dominant_lang = max(script_counts, key=script_counts.get)
        
        # Special handling for specific scripts
        if dominant_lang == "ur_ks":
            # Disambiguate between Urdu, Kashmiri, and Shahmukhi (Punjabi)
            # Check for Punjabi (Shahmukhi) markers
            if any(word in text for word in ["سوہنے", "کیتا", "نوں", "وچ", "دا", "دی", "دے", "ساتھ", "حال", "سریع"]):
                logger.debug(f"[SCRIPT-DETECT] Shahmukhi markers found, classifying as Punjabi (pa)")
                return "pa"
            # Disambiguate between Urdu and Kashmiri using specific markers
            urdu_markers = ["ہے", "کو", "لاہور", "پاکستان", "ھ", "ی", "ٹ", "ں", "گ"]
            kashmiri_markers = ["چھُ", "چھِ", "چُھو", "گژھ", "یہِ", "کینٛہہ", "أكۍ"]
            urdu_count = sum(1 for marker in urdu_markers if marker in text)
            kashmiri_count = sum(1 for marker in kashmiri_markers if marker in text)
            if kashmiri_count > urdu_count:
                return "ks"
            return "ur"
        elif dominant_lang == "tcy":
            # IMPROVED: Distinguish Tulu from Kannada using specific markers
            tulu_markers = ["ಅಪೋ", "ಮಂಗಳೂರು", "ತುಳು", "ಎಡೂ", "ನೋಕು", "ತಾರಿ", "ಕೊಡು", "ಬಾರು", "ಶರಿ", "ಕೆಲಸ"]
            kannada_markers = ["ಬೆಂಗಳೂರು", "ಕರ್ನಾಟಕ", "ಹೌದು", "ನಾವು", "ಇಂದು", "ಈ"]
            tulu_count = sum(1 for marker in tulu_markers if marker in text)
            kannada_count = sum(1 for marker in kannada_markers if marker in text)
            if tulu_count > kannada_count:
                logger.debug(f"[SCRIPT-DETECT] Tulu markers detected, classifying as Tulu (tcy)")
                return "tcy"
            return "kn"  # Default to Kannada if no clear distinction
        elif dominant_lang == "ta":
            # Tamil has distinctive vowel markers - double check
            tamil_markers = ["ற்", "ய்", "ள்", "ணை", "ணी", "ணum"]
            if any(marker in text for marker in tamil_markers):
                logger.debug(f"[SCRIPT-DETECT] Tamil-specific markers found")
                return "ta"
        elif dominant_lang == "te":
            # Telugu has distinctive vowel markers - double check
            telugu_markers = ["ే్", "ై", "ూ", "ృ", "ాలు"]
            if any(marker in text for marker in telugu_markers):
                logger.debug(f"[SCRIPT-DETECT] Telugu-specific markers found")
                return "te"
        elif dominant_lang == "ml":
            # Malayalam has distinctive vowel markers - double check  
            malayalam_markers = ["്ര", "്റ", "െ", "ോ", "ൌ"]
            if any(marker in text for marker in malayalam_markers):
                logger.debug(f"[SCRIPT-DETECT] Malayalam-specific markers found")
                return "ml"
        elif dominant_lang == "kn":
            # Kannada distinctive markers
            kannada_markers = ["್ಯ", "ೆ", "ಣ", "ೃ", "ೀ"]
            if any(marker in text for marker in kannada_markers):
                logger.debug(f"[SCRIPT-DETECT] Kannada-specific markers found")
                return "kn"
        elif dominant_lang == "hi_devanagari":
            # Disambiguate Devanagari variants (Hindi, Marathi, Sanskrit, etc.)
            return self._detect_devanagari_variant(text)
        else:
            # Map key to language code
            lang_code = dominant_lang
            logger.debug(f"[SCRIPT-DETECT] Dominant: {script_counts} → {lang_code}")
            return lang_code

        return None

    def _detect_devanagari_variant(self, text: str) -> str:
        """Disambiguate which Devanagari language (Hindi, Marathi, Konkani, Sanskrit, etc.)"""
        markers = self.LANGUAGE_MARKERS
        best_match = None
        max_score = 0
        text_lower = (text or "").lower()
        # Tokenize by whitespace/punctuation to avoid substring false positives.
        native_tokens = set(
            token for token in re.split(r"[\s,.;:!?।،؛]+", text_lower) if token
        )

        # Check marker frequencies for each Devanagari language
        devanagari_langs = ["sa", "kok", "mai", "bho", "mr", "hi"]
        for lang in devanagari_langs:
            if lang not in markers:
                continue
            native_marks = markers[lang]["native"]
            matches = 0
            for mark in native_marks:
                mark_norm = (mark or "").strip().lower()
                if not mark_norm:
                    continue
                if " " in mark_norm:
                    if mark_norm in text_lower:
                        matches += 2
                else:
                    # Ignore 1-char markers to avoid cross-language collisions like "च"
                    if len(mark_norm) < 2:
                        continue
                    if mark_norm in native_tokens:
                        matches += 1
            if matches > max_score:
                max_score = matches
                best_match = lang

        # If we found strong matches, return best
        if max_score > 0:
            return best_match

        # Use key Sanskrit indicators
        if any(word in text_lower for word in ["नमस्ते", "अस्ति", "स्वागतम्", "कथम्"]):
            return "sa"

        return None  # Let STT language payload or Sarvam fallback catch it instead of blindly dropping to Hindi

    def _count_script_chars(self, text: str, start: int, end: int) -> int:
        """Count how many characters from text fall into given Unicode range.
        Used for DOMINANT script detection when text has mixed scripts."""
        return sum(1 for c in text if start <= ord(c) <= end)

    def _contains_script(self, text: str, start: int, end: int) -> bool:
        """Check if text contains characters from given Unicode range."""
        return any(start <= ord(c) <= end for c in text)

    def _heuristic_detect(self, text: str) -> Tuple[str, Dict[str, float]]:
        """Fallback heuristic detection when script/fastText not available."""
        import re
        raw = (text or "").strip()
        if not raw:
            return ("hi", {"hi": 1.0})

        markers = self.LANGUAGE_MARKERS
        scores = {}
        raw_lower = raw.lower()
        has_non_latin_alpha = any((ord(c) > 127 and c.isalpha()) for c in raw)
        
        # Tokenize preserving alphanumeric characters for word boundaries
        latin_words = set(re.findall(r'[a-zA-Z]+', raw_lower))
        native_words = set(
            token for token in re.split(r"[\s,.;:!?।،؛]+", raw_lower) if token
        )

        for lang, marker_set in markers.items():
            # Native match using token boundaries to avoid substring poisoning.
            native_matches = 0
            for mark in marker_set["native"]:
                mark_lower = (mark or "").strip().lower()
                if not mark_lower:
                    continue
                if " " in mark_lower:
                    if mark_lower in raw_lower:
                        native_matches += 1
                else:
                    if len(mark_lower) < 2:
                        continue
                    if mark_lower in native_words:
                        native_matches += 1
            
            # Latin match (exact word match to prevent "he" matching "hello", or phrase match)
            latin_matches = 0
            for mark in marker_set["latin"]:
                mark_lower = mark.lower()
                if " " in mark_lower:
                    # phrase match
                    if mark_lower in raw_lower:
                        latin_matches += 1
                else:
                    # exact word match
                    if mark_lower in latin_words:
                        latin_matches += 1
                        
            scores[lang] = (native_matches * 2 + latin_matches) / 10.0  # Weight native higher
            if lang == "en" and has_non_latin_alpha:
                # Penalize English when text clearly contains non-Latin language content.
                scores[lang] *= 0.2

        if scores:
            best_lang = max(scores, key=scores.get)
            best_score = scores[best_lang]
            if best_lang == "en" and has_non_latin_alpha and len(latin_words) < 3:
                return (None, {})
            if best_score > 0:
                return (best_lang, {best_lang: 0.9})

        return (None, {})
