#!/usr/bin/env python3
"""
Comprehensive test suite for enhanced language detection with script-based routing.
Tests all 15+ supported languages and their script variants.
"""

import sys
sys.path.insert(0, '/Users/ashwinagarkhed/AVA-AI-Voice-Agent-for-Asterisk/awaaz/src/pipeline')

from lang_detect import TokenLevelLangDetector


def test_script_based_detection():
    """Test script-based detection (PRIMARY METHOD)"""
    detector = TokenLevelLangDetector()
    
    test_cases = [
        # (text, expected_lang, script_name)
        ("ਮੈਂ ਤੁਹਾਡੀ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ", "pa", "Gurmukhi/Punjabi"),
        ("ମୁଁ ତୁମ୍ଭକର ସାହାଯ୍ୟ କରିପାରିବା", "or", "Odia"),
        ("நான் உனக்கு உதவி செய்ய முடியும்", "ta", "Tamil"),
        ("నేను మీకు సహాయం చేయగలను", "te", "Telugu"),
        ("ನಾನು ನಿನ್ನನ್ನು ಸಹಾಯ ಮಾಡಬಲ್ಲೆ", "kn", "Kannada"),
        ("ഞാൻ നിന്നെ സഹായിക്കാൻ കഴിയും", "ml", "Malayalam"),
        ("আমি আপনাকে সাহায্য করতে পারি", "bn", "Bengali"),
        ("मैं तुम्हें मदद कर सकता हूं", "hi", "Hindi/Devanagari"),
        ("मी तुम्हाला मदत करू शकतो", "mr", "Marathi/Devanagari"),
        ("नमस्ते स्वागतम् कथम् भवान्", "sa", "Sanskrit/Devanagari"),
        ("ہے کو لاہور پاکستان", "ur", "Urdu/Perso-Arabic"),
    ]
    
    print("=" * 60)
    print("SCRIPT-BASED DETECTION TESTS (PRIMARY METHOD)")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for text, expected, script_name in test_cases:
        detected, dist = detector.detect(text)
        status = "✓ PASS" if detected == expected else "✗ FAIL"
        if detected == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} | {script_name:30} | Detected: {detected:6} (Expected: {expected:6})")
        print(f"       | Text: {text[:40]}...")
        print(f"       | Confidence: {dist}")
    
    print(f"\nScript Detection Results: {passed} passed, {failed} failed out of {len(test_cases)}")
    return passed, failed


def test_devanagari_disambiguation():
    """Test Devanagari variant disambiguation"""
    detector = TokenLevelLangDetector()
    
    test_cases = [
        ("नमस्ते, स्वागतम्, कथम् आस्मि", "sa", "Sanskrit"),
        ("महाराष्ट्र मुंबई साहय आहे", "mr", "Marathi"),
        ("दिल्ली भारत मदद है", "hi", "Hindi"),
        ("दरभंगा मिथिला मदद छै", "mai", "Maithili"),
        ("गोवा कोंकणी साहय", "kok", "Konkani"),
    ]
    
    print("\n" + "=" * 60)
    print("DEVANAGARI DISAMBIGUATION TESTS")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for text, expected, lang_name in test_cases:
        detected, dist = detector.detect(text)
        status = "✓ PASS" if detected == expected else "✗ FAIL"
        if detected == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} | {lang_name:15} | Detected: {detected:6} (Expected: {expected:6})")
        print(f"       | Text: {text[:50]}...")
    
    print(f"\nDevanagari Disambiguation Results: {passed} passed, {failed} failed out of {len(test_cases)}")
    return passed, failed


def test_mixed_language():
    """Test mixed language detection (should prioritize script)"""
    detector = TokenLevelLangDetector()
    
    test_cases = [
        ("नमस्ते, hello, how are you?", "hi", "Devanagari with English"),
        ("ਸਤਿ ਨਾਮ hello", "pa", "Punjabi/Gurmukhi with English"),
        ("வணக்கம் hello world", "ta", "Tamil with English"),
    ]
    
    print("\n" + "=" * 60)
    print("MIXED LANGUAGE TESTS (SCRIPT PRIORITY)")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for text, expected, scenario in test_cases:
        detected, dist = detector.detect(text)
        status = "✓ PASS" if detected == expected else "✗ FAIL"
        if detected == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} | {scenario:30} | Detected: {detected:6}")
        print(f"       | Text: {text[:50]}...")
        print(f"       | Mode: pure" if dist.get(detected, 0) > 0.8 else "       | Mode: mixed")
    
    print(f"\nMixed Language Results: {passed} passed, {failed} failed out of {len(test_cases)}")
    return passed, failed


def test_feminine_language_config():
    """Verify that feminine language instruction is still in place"""
    sys.path.insert(0, '/Users/ashwinagarkhed/AVA-AI-Voice-Agent-for-Asterisk/awaaz/src/pipeline')
    
    try:
        from nlp import NLPEngine
        
        nlp = NLPEngine()
        # Check if the system prompt contains feminine language constraint
        sample_prompt = nlp._build_system_prompt("hi")
        
        print("\n" + "=" * 60)
        print("FEMININE LANGUAGE CONSTRAINT VERIFICATION")
        print("=" * 60)
        
        if "FEMININE" in sample_prompt and "feminine" in sample_prompt.lower():
            print("✓ PASS | Feminine language constraint found in system prompt")
            print(f"       | Prompt includes feminine language instruction")
            return 1, 0
        else:
            print("✗ FAIL | Feminine language constraint NOT found")
            return 0, 1
    except Exception as e:
        print(f"✗ ERROR | Could not verify feminine language: {e}")
        return 0, 1


def test_tts_settings():
    """Verify TTS voice and pace settings"""
    sys.path.insert(0, '/Users/ashwinagarkhed/AVA-AI-Voice-Agent-for-Asterisk/awaaz/src/pipeline')
    
    try:
        from tts import TTSEngine
        import yaml
        
        print("\n" + "=" * 60)
        print("TTS SETTINGS VERIFICATION")
        print("=" * 60)
        
        # Check if SARVAM_SPEAKER_MAP has pace settings
        if hasattr(TTSEngine, 'SARVAM_SPEAKER_MAP'):
            speaker_map = TTSEngine.SARVAM_SPEAKER_MAP
            
            tests_passed = 0
            tests_failed = 0
            
            # Test Kannada (should be 0.8)
            if "kn" in speaker_map and speaker_map["kn"].get("pace") == 0.8:
                print("✓ PASS | Kannada pace = 0.8 (20% slower)")
                tests_passed += 1
            else:
                print("✗ FAIL | Kannada pace not set correctly")
                tests_failed += 1
            
            # Test Gujarati (should have expressiveness)
            if "gu" in speaker_map and speaker_map["gu"].get("pitch"):
                print(f"✓ PASS | Gujarati has expressiveness settings (pitch={speaker_map['gu'].get('pitch')})")
                tests_passed += 1
            else:
                print("✗ FAIL | Gujarati expressiveness not configured")
                tests_failed += 1
            
            return tests_passed, tests_failed
        else:
            print("⚠ SKIP  | TTS Speaker map not available")
            return 0, 0
            
    except Exception as e:
        print(f"✗ ERROR | Could not verify TTS settings: {e}")
        return 0, 1


def main():
    """Run all comprehensive tests"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " COMPREHENSIVE LANGUAGE DETECTION TEST SUITE ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    total_passed = 0
    total_failed = 0
    
    # Test 1: Script-based detection
    p, f = test_script_based_detection()
    total_passed += p
    total_failed += f
    
    # Test 2: Devanagari disambiguation
    p, f = test_devanagari_disambiguation()
    total_passed += p
    total_failed += f
    
    # Test 3: Mixed language
    p, f = test_mixed_language()
    total_passed += p
    total_failed += f
    
    # Test 4: Feminine language constraint
    p, f = test_feminine_language_config()
    total_passed += p
    total_failed += f
    
    # Test 5: TTS Settings
    p, f = test_tts_settings()
    total_passed += p
    total_failed += f
    
    # Summary
    print("\n" + "=" * 60)
    print("OVERALL TEST SUMMARY")
    print("=" * 60)
    print(f"Total Passed: {total_passed}")
    print(f"Total Failed: {total_failed}")
    print(f"Success Rate: {total_passed / (total_passed + total_failed) * 100:.1f}%" if (total_passed + total_failed) > 0 else "N/A")
    
    if total_failed == 0:
        print("\n✓ ALL TESTS PASSED! System is ready for deployment.")
    else:
        print(f"\n✗ {total_failed} test(s) failed. Please review the output above.")
    
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
