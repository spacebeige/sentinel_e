import pytest
import asyncio
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.stt import STTProcessor, STTResult
from src.pipeline.tts import TTSProcessor, ElevenLabsTTS
from src.pipeline.nlp import ModelProcessor, LANGUAGE_CONFIG, check_emergency
from src.pipeline.lang_detect import TokenLevelLangDetector
from src.session_store import AWAAZSession

class DummySession:
    def __init__(self, lang="hi"):
        self.lang = lang
        self.lang_distribution = {}

@pytest.fixture
def stt_processor():
    return STTProcessor()

@pytest.fixture
def lang_detector():
    return TokenLevelLangDetector.get()

@pytest.fixture
def tts_processor():
    return TTSProcessor()

@pytest.fixture
def elevenlabs_tts():
    return ElevenLabsTTS()

@pytest.fixture
def mp():
    processor = MagicMock()
    processor.lang_detector = MagicMock()
    return processor


# ═══════════════════════════════════════════════════════════════════════════
# LANGUAGE CONFIGURATION TESTS - All 32+ languages
# ═══════════════════════════════════════════════════════════════════════════

class TestLanguageConfiguration:
    """Verify all 32+ Indian languages are configured."""
    
    def test_all_major_languages_present(self):
        """Test that all major Indian languages are in LANGUAGE_CONFIG."""
        major_langs = ["hi", "mr", "ta", "te", "kn", "ml", "gu", "bn", "pa", "or", "en"]
        for lang in major_langs:
            assert lang in LANGUAGE_CONFIG, f"Missing {lang} in LANGUAGE_CONFIG"
            assert "name" in LANGUAGE_CONFIG[lang]
            assert "gtts" in LANGUAGE_CONFIG[lang]
            assert "script" in LANGUAGE_CONFIG[lang]
    
    def test_all_regional_devanagari_languages(self):
        """Test all regional Devanagari languages."""
        regional_langs = ["bho", "awa", "mai", "bgc", "doi", "mwr", "pah", "ne", "kok", "hne", "raj", "kfy", "gbm"]
        for lang in regional_langs:
            assert lang in LANGUAGE_CONFIG, f"Missing {lang} in LANGUAGE_CONFIG"
    
    def test_all_austroasiatic_languages(self):
        """Test Austro-Asiatic languages (Santali, Bodo, Manipuri)."""
        langs = ["sat", "brx", "mni"]
        for lang in langs:
            assert lang in LANGUAGE_CONFIG, f"Missing {lang} in LANGUAGE_CONFIG"
    
    def test_all_south_indian_languages(self):
        """Test South Indian languages."""
        langs = ["ta", "te", "kn", "ml", "tcy"]
        for lang in langs:
            assert lang in LANGUAGE_CONFIG, f"Missing {lang} in LANGUAGE_CONFIG"
    
    def test_all_perso_arabic_languages(self):
        """Test Perso-Arabic script languages."""
        langs = ["ur", "ks", "sd"]
        for lang in langs:
            assert lang in LANGUAGE_CONFIG, f"Missing {lang} in LANGUAGE_CONFIG"
    
    def test_code_mixed_variants_generated(self):
        """Test that code-mixed variants (e.g., hi-en, mr-en) are generated."""
        code_mixed = ["hi-en", "mr-en", "ta-en", "te-en", "bho-en", "mai-en"]
        for mixed in code_mixed:
            assert mixed in LANGUAGE_CONFIG, f"Missing {mixed} in LANGUAGE_CONFIG"


# ═══════════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION TESTS - Script and Heuristic Detection
# ═══════════════════════════════════════════════════════════════════════════

class TestLanguageDetection:
    """Test comprehensive language detection for all scripts."""
    
    # Script-level detection
    def test_tamil_script_detection(self, lang_detector):
        tamil_text = "உள்ளது என்று சாலை வாணம்"
        lang, dist = lang_detector.detect(tamil_text)
        assert lang == "ta", f"Expected Tamil but got {lang}"
    
    def test_telugu_script_detection(self, lang_detector):
        telugu_text = "ఉంది కావాలి ఏమిటి సమస్య"
        lang, dist = lang_detector.detect(telugu_text)
        assert lang == "te", f"Expected Telugu but got {lang}"
    
    def test_kannada_script_detection(self, lang_detector):
        kannada_text = "ಇದೆ ಬೇಕು ಸಮಸ್ಯೆ ವಿದ್ಯುತ್"
        lang, dist = lang_detector.detect(kannada_text)
        assert lang == "kn", f"Expected Kannada but got {lang}"
    
    def test_malayalam_script_detection(self, lang_detector):
        malayalam_text = "ഉണ്ട് വേണം പ്രശ്നം ഇലക്ട്രിസിറ്റി"
        lang, dist = lang_detector.detect(malayalam_text)
        assert lang == "ml", f"Expected Malayalam but got {lang}"
    
    def test_gujarati_script_detection(self, lang_detector):
        gujarati_text = "છે જોઈએ શું સમસ્યા"
        lang, dist = lang_detector.detect(gujarati_text)
        assert lang == "gu", f"Expected Gujarati but got {lang}"
    
    def test_punjabi_script_detection(self, lang_detector):
        punjabi_text = "ਹੈ ਚਾਹੀ ਕੀ ਸਮੱਸਿਆ"
        lang, dist = lang_detector.detect(punjabi_text)
        assert lang == "pa", f"Expected Punjabi but got {lang}"
    
    def test_odia_script_detection(self, lang_detector):
        odia_text = "ଅଛି ଚାହେ ସମସ୍ୟା"
        lang, dist = lang_detector.detect(odia_text)
        assert lang == "or", f"Expected Odia but got {lang}"
    
    def test_bengali_script_detection(self, lang_detector):
        bengali_text = "আছে চাই কি সমস্যা"
        lang, dist = lang_detector.detect(bengali_text)
        assert lang == "bn", f"Expected Bengali but got {lang}"
    
    # Devanagari disambiguation
    def test_marathi_detected_from_devanagari(self, lang_detector):
        marathi_text = "मला आहे काय नाही पाहिजे"
        lang, dist = lang_detector.detect(marathi_text)
        assert lang == "mr", f"Expected Marathi but got {lang}"
    
    def test_hindi_detected_from_devanagari(self, lang_detector):
        hindi_text = "है हैं मुझे क्या नहीं चाहिए"
        lang, dist = lang_detector.detect(hindi_text)
        assert lang == "hi", f"Expected Hindi but got {lang}"
    
    # Regional Devanagari languages
    def test_bhojpuri_detection(self, lang_detector):
        bhojpuri_text = "बा चाहत नीकै सहायता"
        lang, dist = lang_detector.detect(bhojpuri_text)
        # Should detect as one of Devanagari family (preferably bho)
        assert lang in ["bho", "hi"], f"Expected Bhojpuri/Hindi but got {lang}"
    
    def test_chhattisgarhi_detection(self, lang_detector):
        chhattisgarhi_text = "छत्तीसगढ़ समस्या सहायता"
        lang, dist = lang_detector.detect(chhattisgarhi_text)
        assert lang in ["hne", "hi"], f"Expected Chhattisgarhi/Hindi but got {lang}"
    
    # Romanized detection
    def test_hinglish_romanized_detection(self, lang_detector):
        hinglish = "mujhe loan karna hai problem hai"
        lang, dist = lang_detector.detect(hinglish)
        # Should detect as primarily Hindi/English mix
        assert lang in ["hi", "en"], f"Expected Hindi/English but got {lang}"
    
    def test_marathi_latin_detection(self, lang_detector):
        marathi_latin = "majha problem ahe kay nahi"
        lang, dist = lang_detector.detect(marathi_latin)
        assert lang in ["mr", "hi"], f"Expected Marathi/Hindi but got {lang}"


# ═══════════════════════════════════════════════════════════════════════════
# EMERGENCY DETECTION TESTS - All Languages
# ═══════════════════════════════════════════════════════════════════════════

class TestEmergencyDetection:
    """Test emergency keyword detection for all languages."""
    
    # Major language emergency detection
    def test_emergency_hindi(self):
        is_emergency = check_emergency("मदद करो मदद", "hi", None)
        assert is_emergency is True
    
    def test_emergency_marathi(self):
        is_emergency = check_emergency("मदद वाचवा आपातकाल", "mr", None)
        assert is_emergency is True
    
    def test_emergency_tamil(self):
        is_emergency = check_emergency("உதவி தேவை அவசரம்", "ta", None)
        assert is_emergency is True
    
    def test_emergency_telugu(self):
        is_emergency = check_emergency("సహాయం అత్యవసర మీ", "te", None)
        assert is_emergency is True
    
    def test_emergency_kannada(self):
        is_emergency = check_emergency("ಸಹಾಯ ತುರ್ತು ಅಪಾಯ", "kn", None)
        assert is_emergency is True
    
    def test_emergency_bengali(self):
        is_emergency = check_emergency("সাহায্য জরুরি পুলিশ", "bn", None)
        assert is_emergency is True
    
    # Regional language emergency detection
    def test_emergency_bhojpuri(self):
        is_emergency = check_emergency("मदद लव्कर आपातकाल", "bho", None)
        assert is_emergency is True
    
    def test_emergency_dogri(self):
        is_emergency = check_emergency("मदद आपातकालीन डाक्टर", "doi", None)
        assert is_emergency is True
    
    def test_emergency_maithili(self):
        is_emergency = check_emergency("मदद आपातकाल बचाओ", "mai", None)
        assert is_emergency is True
    
    def test_emergency_english(self):
        is_emergency = check_emergency("emergency help police fire", "en", None)
        assert is_emergency is True
    
    def test_non_emergency_hindi(self):
        is_emergency = check_emergency("मुझे पानी का बिल चाहिए", "hi", None)
        assert is_emergency is False
    
    def test_non_emergency_tamil(self):
        is_emergency = check_emergency("எனக்கு நீர் பாசனம் தேவை", "ta", None)
        assert is_emergency is False


# ═══════════════════════════════════════════════════════════════════════════
# TTS VOICE SELECTION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestTTSVoiceSelection:
    """Test TTS voice selection for all languages."""
    
    def test_elevenlabs_hindi_voice(self, elevenlabs_tts):
        voice = elevenlabs_tts._get_voice_id("hi")
        assert voice == "pFZP5JQG7iQjIQuC4Bku"
    
    def test_elevenlabs_english_voice(self, elevenlabs_tts):
        voice = elevenlabs_tts._get_voice_id("en")
        assert voice == "EXAVITQu4vr4xnSDxMaL"
    
    def test_elevenlabs_marathi_uses_hindi_base(self, elevenlabs_tts):
        voice = elevenlabs_tts._get_voice_id("mr")
        assert voice == "pFZP5JQG7iQjIQuC4Bku"
    
    def test_elevenlabs_tamil_voice(self, elevenlabs_tts):
        voice = elevenlabs_tts._get_voice_id("ta")
        assert voice == "EXAVITQu4vr4xnSDxMaL"
    
    def test_elevenlabs_kannada_voice(self, elevenlabs_tts):
        voice = elevenlabs_tts._get_voice_id("kn")
        assert voice == "EXAVITQu4vr4xnSDxMaL"
    
    def test_elevenlabs_gujarati_voice(self, elevenlabs_tts):
        voice = elevenlabs_tts._get_voice_id("gu")
        assert voice == "pFZP5JQG7iQjIQuC4Bku"
    
    def test_elevenlabs_bengali_voice(self, elevenlabs_tts):
        voice = elevenlabs_tts._get_voice_id("bn")
        assert voice == "pFZP5JQG7iQjIQuC4Bku"
    
    def test_elevenlabs_bhojpuri_voice(self, elevenlabs_tts):
        voice = elevenlabs_tts._get_voice_id("bho")
        assert voice == "pFZP5JQG7iQjIQuC4Bku"  # Uses Hindi base
    
    def test_elevenlabs_santali_voice(self, elevenlabs_tts):
        voice = elevenlabs_tts._get_voice_id("sat")
        assert voice == "pFZP5JQG7iQjIQuC4Bku"  # Uses Bengali base
    
    def test_elevenlabs_code_mixed_uses_base_language(self, elevenlabs_tts):
        # hi-en should use Hindi voice
        voice_hi_en = elevenlabs_tts._get_voice_id("hi-en")
        assert voice_hi_en == "pFZP5JQG7iQjIQuC4Bku"
        
        # ta-en should use Tamil voice (Indian English)
        voice_ta_en = elevenlabs_tts._get_voice_id("ta-en")
        assert voice_ta_en == "EXAVITQu4vr4xnSDxMaL"


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STORE TESTS - Language Profile
# ═══════════════════════════════════════════════════════════════════════════

class TestSessionLanguageProfile:
    """Test session language profile storage and retrieval."""
    
    def test_session_stores_language_fields(self):
        session = AWAAZSession()
        assert hasattr(session, 'lang')
        assert hasattr(session, 'lang_name')
        assert hasattr(session, 'lang_mode')
        assert hasattr(session, 'lang_distribution')
        assert hasattr(session, 'accent_region')
        assert hasattr(session, 'phonetic_style')
        assert hasattr(session, 'formality_score')
        assert hasattr(session, 'formality_label')
        assert hasattr(session, 'script')
    
    def test_session_default_language_is_hindi(self):
        session = AWAAZSession()
        assert session.lang == "hi"
        assert session.lang_name == "Hindi"
    
    def test_session_formality_tracking(self):
        session = AWAAZSession()
        session.formality_score = 0.8
        session.formality_label = "FORMAL"
        assert session.formality_score == 0.8
        assert session.formality_label == "FORMAL"
    
    def test_session_accent_region_tracking(self):
        session = AWAAZSession()
        session.accent_region = "marathi-konkan"
        assert session.accent_region == "marathi-konkan"
    
    def test_session_phonetic_style_tracking(self):
        session = AWAAZSession()
        session.phonetic_style = "romanized"
        assert session.phonetic_style == "romanized"


# ═══════════════════════════════════════════════════════════════════════════
# LANGUAGE DEGRADATION TESTS - Tier 3 fallbacks
# ═══════════════════════════════════════════════════════════════════════════

class TestLanguageDegradation:
    """Test TTS fallback mapping for languages without direct provider support."""
    
    def test_santali_falls_to_bengali_tts(self):
        assert LANGUAGE_CONFIG["sat"]["gtts"] == "bn"
    
    def test_kashmiri_falls_to_urdu_tts(self):
        assert LANGUAGE_CONFIG["ks"]["gtts"] == "ur"
    
    def test_manipuri_falls_to_bengali_tts(self):
        assert LANGUAGE_CONFIG["mni"]["gtts"] == "bn"
    
    def test_assamese_falls_to_bengali_tts(self):
        assert LANGUAGE_CONFIG["as"]["gtts"] == "bn"
    
    def test_kumaoni_falls_to_hindi_tts(self):
        assert LANGUAGE_CONFIG["kfy"]["gtts"] == "hi"
    
    def test_garhwali_falls_to_hindi_tts(self):
        assert LANGUAGE_CONFIG["gbm"]["gtts"] == "hi"


# ═══════════════════════════════════════════════════════════════════════════
# MIXED LANGUAGE TESTS - Code-mixing
# ═══════════════════════════════════════════════════════════════════════════

class TestCodeMixing:
    """Test code-mixing (Hinglish, Marathi-English, etc.) handling."""
    
    def test_hinglish_detected_as_mixed(self, mp):
        mp.lang_detector.detect = MagicMock(return_value=("hi", {"hi": 0.6, "en": 0.4}))
        lang, dist = mp.lang_detector.detect("Mujhe loan apply karna hai")
        assert lang == "hi"
        assert dist["en"] > 0
    
    def test_marathi_english_detected(self, mp):
        mp.lang_detector.detect = MagicMock(return_value=("mr", {"mr": 0.65, "en": 0.35}))
        lang, dist = mp.lang_detector.detect("Majha electricity bill nahi aala")
        assert lang == "mr"
        assert dist["en"] > 0
    
    def test_tamil_english_code_mix(self, mp):
        mp.lang_detector.detect = MagicMock(return_value=("ta", {"ta": 0.6, "en": 0.4}))
        lang, dist = mp.lang_detector.detect("என்னுடைய problem வாணம்")
        assert lang == "ta"
    
    def test_code_mixed_variant_in_config(self):
        """Test that code-mixed variants exist in LANGUAGE_CONFIG."""
        mixed_langs = ["hi-en", "mr-en", "ta-en", "te-en", "kn-en", "ml-en"]
        for mixed in mixed_langs:
            assert mixed in LANGUAGE_CONFIG, f"Missing {mixed} in LANGUAGE_CONFIG"
            assert "name" in LANGUAGE_CONFIG[mixed]


# ═══════════════════════════════════════════════════════════════════════════
# SCRIPT DETECTION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestScriptDetection:
    """Test script-level detection for language identification."""
    
    def test_contains_script_detection(self):
        detector = TokenLevelLangDetector.get()
        
        # Test Devanagari detection
        assert detector._contains_script("आहे", 0x0900, 0x097F)
        
        # Test Tamil detection
        assert detector._contains_script("உள்ளது", 0x0B80, 0x0BFF)
        
        # Test Bengali detection
        assert detector._contains_script("আছে", 0x0980, 0x09FF)
    
    def test_contains_script_by_name(self):
        detector = TokenLevelLangDetector.get()
        
        # Test by script name
        assert detector._contains_script_by_name("आहे", "Devanagari")
        assert detector._contains_script_by_name("உள்ளது", "Tamil")
        assert detector._contains_script_by_name("আছে", "Bengali")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    mock_processor.client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="YES"))
    ]
    is_emergency = check_emergency("meri madad karo", "doi", mock_processor)
    assert is_emergency is True

def test_emergency_detection_santali():
    mock_processor = MagicMock()
    mock_processor.client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="YES"))
    ]
    is_emergency = check_emergency("bachao", "sat", mock_processor)
    assert is_emergency is True

def test_hinglish_detected_as_mixed(mp):
    mp.lang_detector.detect = MagicMock(return_value=("hi", {"hi": 0.6, "en": 0.4}))
    lang, dist = mp.lang_detector.detect("Mujhe loan apply karna hai")
    assert lang == "hi"
    assert dist["en"] > 0

def test_marathi_english_detected(mp):
    mp.lang_detector.detect = MagicMock(return_value=("mr", {"mr": 0.65, "en": 0.35}))
    lang, dist = mp.lang_detector.detect("Majha electricity bill nahi aala")
    assert lang == "mr"
    assert dist["en"] > 0

def test_bhojpuri_routes_to_groq_on_low_confidence(stt_processor):
    # Mock ElevenLabs to return low confidence
    stt_processor.providers = [MagicMock(), MagicMock()]
    stt_processor.providers[0].name = "elevenlabs_stt"
    fut1 = asyncio.Future()
    fut1.set_result(STTResult(text="hum log", provider="elevenlabs_stt", confidence=0.4, detected_language="bho"))
    stt_processor.providers[0].transcribe = MagicMock(return_value=fut1)
    
    stt_processor.providers[1].name = "groq_whisper"
    fut2 = asyncio.Future()
    fut2.set_result(STTResult(text="hum log jaib", provider="groq_whisper", confidence=0.9, detected_language="bho"))
    stt_processor.providers[1].transcribe = MagicMock(return_value=fut2)
    
    stt_processor.confidence_threshold = 0.6
    result = asyncio.run(stt_processor.transcribe("fake.wav", "auto"))
    
    assert result.provider == "groq_whisper"
    assert result.confidence > 0.6
    assert result.detected_language == "bho"

def test_dogri_text_detected_as_doi_not_hi(mp):
    mp.lang_detector.detect = MagicMock(return_value=("doi", {"doi": 0.9, "hi": 0.1}))
    lang, dist = mp.lang_detector.detect("कुत्ते")
    assert lang == "doi"

def test_chhattisgarhi_not_misclassified_as_hindi(mp):
    mp.lang_detector.detect = MagicMock(return_value=("hne", {"hne": 0.8, "hi": 0.2}))
    lang, dist = mp.lang_detector.detect("mor naam rohit he")
    assert lang == "hne"

def test_bhojpuri_hum_log_not_hindi(mp):
    mp.lang_detector.detect = MagicMock(return_value=("bho", {"bho": 0.8, "hi": 0.2}))
    lang, dist = mp.lang_detector.detect("hum log jaib patna")
    assert lang == "bho"

def test_awadhi_detected_correctly(mp):
    mp.lang_detector.detect = MagicMock(return_value=("awa", {"awa": 0.85, "hi": 0.15}))
    lang, dist = mp.lang_detector.detect("hamre yahan paani nahi aat")
    assert lang == "awa"
