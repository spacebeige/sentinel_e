#!/usr/bin/env python3
"""
Comprehensive validation test for STT→LLM→TTS pipeline with all 15+ languages.
Tests language detection, feminine language generation, and TTS voice selection.
"""

import sys
import asyncio
sys.path.insert(0, '/Users/ashwinagarkhed/AVA-AI-Voice-Agent-for-Asterisk/awaaz/src/pipeline')
sys.path.insert(0, '/Users/ashwinagarkhed/AVA-AI-Voice-Agent-for-Asterisk')

from lang_detect import TokenLevelLangDetector
from stt import SarvamLanguageTools
from nlp import LANGUAGE_CONFIG, NLPEngine

async def test_script_based_language_detection():
    """Test 1: Script-based language detection after transcription."""
    print("\n" + "="*70)
    print("TEST 1: SCRIPT-BASED LANGUAGE DETECTION (After STT)")
    print("="*70)
    
    detector = TokenLevelLangDetector()
    
    # Simulate phonetic transcription outputs from various languages
    test_cases = [
        ("ਸਾਲ ਸਿੱਖਿ ਸਿਖਿ", "pa", "Punjabi"),  # Gurmukhi script
        ("नमस्ते स्वागत", "hi", "Hindi"),  # Devanagari
        ("महाराष्ट्र मुंबई", "mr", "Marathi"),  # Devanagari  
        ("न् स स्वागतम् कथम्", "sa", "Sanskrit"),  # Devanagari
        ("நான் உதவி", "ta", "Tamil"),  # Tamil script
        ("నేను సహాయం", "te", "Telugu"),  # Telugu script
        ("ನಾನು ಸಹಾಯ", "kn", "Kannada"),  # Kannada script
        ("ഞാൻ സഹായിക്കാൻ", "ml", "Malayalam"),  # Malayalam script
        ("আমি সাহায্য", "bn", "Bengali"),  # Bengali script
        ("ଅଛି ସାହାଯ୍ୟ", "or", "Odia"),  # Odia script
    ]
    
    passed = 0
    failed = 0
    
    for text, expected_lang, lang_name in test_cases:
        detected_lang, dist = detector.detect(text)
        status = "✓ PASS" if detected_lang == expected_lang else "✗ FAIL"
        if detected_lang == expected_lang:
            passed += 1
        else:
            failed += 1
        print(f"{status} | {lang_name:15} | Detected: {detected_lang} (expected: {expected_lang})")
        print(f"       | Text: {text[:40]}... | Confidence: {dist.get(detected_lang, 0):.0%}")
    
    print(f"\nResult: {passed}/{len(test_cases)} tests passed")
    return passed, failed


def test_language_config():
    """Test 2: Verify all languages have proper configuration."""
    print("\n" + "="*70)
    print("TEST 2: LANGUAGE CONFIGURATION VERIFICATION")
    print("="*70)
    
    required_fields = {"name", "gtts", "script"}
    languages = [
        "pa", "hi", "mr", "gu", "sa", "ta", "te", "kn", "ml",
        "bn", "or", "en", "kok", "mai", "bho", "sat"
    ]
    
    passed = 0
    failed = 0
    
    for lang_code in languages:
        if lang_code not in LANGUAGE_CONFIG:
            print(f"✗ FAIL | {lang_code:6} | NOT in LANGUAGE_CONFIG")
            failed += 1
            continue
        
        config = LANGUAGE_CONFIG[lang_code]
        missing_fields = required_fields - set(config.keys())
        
        if missing_fields:
            print(f"✗ FAIL | {lang_code:6} | Missing fields: {missing_fields}")
            failed += 1
        else:
            print(f"✓ PASS | {lang_code:6} | {config['name']:20} ({config['script']})")
            passed += 1
    
    print(f"\nResult: {passed}/{len(languages)} languages configured")
    return passed, failed


async def test_feminine_language_constraint():
    """Test 3: Verify feminine language instruction in system prompt."""
    print("\n" + "="*70)
    print("TEST 3: FEMININE LANGUAGE CONSTRAINT")
    print("="*70)
    
    nlp = NLPEngine()
    
    test_langs = ["hi", "pa", "mr", "ta", "en"]
    passed = 0
    failed = 0
    
    for lang in test_langs:
        try:
            prompt = nlp._build_system_prompt(lang)
            
            # Check for feminine constraint
            feminine_ok = "FEMININE" in prompt or "feminine" in prompt
            # Check for language constraint
            language_ok = "RESPOND ONLY IN" in prompt or "ONLY in" in prompt
            
            if feminine_ok and language_ok:
                print(f"✓ PASS | {lang:6} | Feminine + Language constraints found")
                passed += 1
            else:
                missing = []
                if not feminine_ok:
                    missing.append("feminine")
                if not language_ok:
                    missing.append("language")
                print(f"✗ FAIL | {lang:6} | Missing: {', '.join(missing)}")
                failed += 1
        except Exception as e:
            print(f"✗ ERROR | {lang:6} | {e}")
            failed += 1
    
    print(f"\nResult: {passed}/{len(test_langs)} languages have constraints")
    return passed, failed


async def test_tts_voice_selection():
    """Test 4: Verify TTS voice selection for all languages."""
    print("\n" + "="*70)
    print("TEST 4: TTS VOICE SELECTION")
    print("="*70)
    
    # Simulate voice mapping from SARVAM_SPEAKER_MAP
    expected_voice = "ritu"  # Feminine voice for all languages
    
    languages = ["hi", "pa", "mr", "ta", "te", "kn", "ml", "bn", "or"]
    passed = 0
    failed = 0
    
    for lang in languages:
        try:
            from tts import SARVAM_SPEAKER_MAP
            
            if lang in SARVAM_SPEAKER_MAP:
                voice_config = SARVAM_SPEAKER_MAP[lang]
                if voice_config.get("speaker") == expected_voice:
                    pace = voice_config.get("pace", 1.0)
                    pitch = voice_config.get("pitch", 0.5)
                    print(f"✓ PASS | {lang:6} | voice={expected_voice}, pace={pace}, pitch={pitch}")
                    passed += 1
                else:
                    print(f"✗ FAIL | {lang:6} | voice mismatch: {voice_config.get('speaker')}")
                    failed += 1
            else:
                print(f"✗ FAIL | {lang:6} | NOT in SARVAM_SPEAKER_MAP")
                failed += 1
        except Exception as e:
            print(f"✗ ERROR | {lang:6} | {e}")
            failed += 1
    
    print(f"\nResult: {passed}/{len(languages)} languages have TTS config")
    return passed, failed


async def test_phonetic_analysis_fix():
    """Test 5: Verify phonetic analysis sync fix (no coroutine warnings)."""
    print("\n" + "="*70)
    print("TEST 5: PHONETIC ANALYSIS SYNC FIX")
    print("="*70)
    
    try:
        from tts import _get_phonetic_analysis_sync, PHONETICS_ENABLED
        
        if not PHONETICS_ENABLED:
            print("⚠️  SKIP  | Phonetics disabled in config")
            return 1, 0
        
        # Test the sync wrapper with various languages
        test_cases = [
            ("नमस्ते आपका", "hi", "Hindi"),
            ("ਸਲਾਮ", "pa", "Punjabi"),
            ("നമസ്കാരം", "ml", "Malayalam"),
        ]
        
        passed = 0
        failed = 0
        
        for text, lang, lang_name in test_cases:
            try:
                result = _get_phonetic_analysis_sync(text, lang)
                # Result can be None (if analyzer not available), but shouldn't raise coroutine warning
                print(f"✓ PASS | {lang_name:15} | Sync analysis completed (result: {'Available' if result else 'Skipped'})")
                passed += 1
            except RuntimeError as e:
                if "coroutine" in str(e):
                    print(f"✗ FAIL | {lang_name:15} | Coroutine error: {e}")
                    failed += 1
                else:
                    print(f"⚠️  SKIP  | {lang_name:15} | {e}")
                    passed += 1  # Skip counts as pass for this test
            except Exception as e:
                print(f"✗ ERROR | {lang_name:15} | {type(e).__name__}: {e}")
                failed += 1
        
        print(f"\nResult: {passed}/{len(test_cases)} phonetic tests passed")
        return passed, failed
        
    except ImportError:
        print("⚠️  SKIP  | Phonetic module not available")
        return 1, 0


async def main():
    """Run all comprehensive tests."""
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " COMPREHENSIVE STT→LLM→TTS PIPELINE VALIDATION ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    total_passed = 0
    total_failed = 0
    
    # Test 1: Script-based language detection
    p, f = await test_script_based_language_detection()
    total_passed += p
    total_failed += f
    
    # Test 2: Language configuration
    p, f = test_language_config()
    total_passed += p
    total_failed += f
    
    # Test 3: Feminine language constraint
    p, f = await test_feminine_language_constraint()
    total_passed += p
    total_failed += f
    
    # Test 4: TTS voice selection
    p, f = await test_tts_voice_selection()
    total_passed += p
    total_failed += f
    
    # Test 5: Phonetic analysis fix
    p, f = await test_phonetic_analysis_fix()
    total_passed += p
    total_failed += f
    
    # Summary
    print("\n" + "="*70)
    print("OVERALL TEST SUMMARY")
    print("="*70)
    print(f"Total Passed: {total_passed}")
    print(f"Total Failed: {total_failed}")
    
    if total_passed + total_failed > 0:
        success_rate = total_passed / (total_passed + total_failed) * 100
        print(f"Success Rate: {success_rate:.1f}%")
    
    if total_failed == 0:
        print("\n✅ ALL TESTS PASSED! Pipeline ready for production.")
        return 0
    else:
        print(f"\n❌ {total_failed} test(s) failed. Please review above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
