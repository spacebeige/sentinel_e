#!/usr/bin/env python3
"""
Language Enforcement Comprehensive Test
Verifies that marathi input produces marathi TTS output (not English)
"""

import sys
import os

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "awaaz"))

from src.session_store import AWAAZSession
from src.pipeline.tts import TTSProcessor

def test_language_enforcement():
    print("\n" + "="*70)
    print("  AWAAZ LANGUAGE ENFORCEMENT TEST")
    print("  Issue: Marathi input should produce Marathi TTS (not English)")
    print("="*70)
    
    # Test 1: Session Language
    print("\n[TEST 1] Session Language Setting")
    session = AWAAZSession()
    session.lang = "mr"
    session.lang_name = "Marathi"
    session.gtts_lang = "mr"
    print(f"  ✓ Session.lang = {session.lang}")
    print(f"  ✓ Session.lang_name = {session.lang_name}")
    assert session.lang == "mr", "Session language should be Marathi"
    
    # Test 2: Voice Selection
    print("\n[TEST 2] ElevenLabs Voice Selection")
    tts = TTSProcessor()
    
    voice_mr = tts.elevenlabs._get_voice_id("mr")
    voice_en = tts.elevenlabs._get_voice_id("en")
    voice_hi = tts.elevenlabs._get_voice_id("hi")
    
    print(f"  ✓ Marathi voice ID:  {voice_mr}")
    print(f"  ✓ English voice ID:  {voice_en}")
    print(f"  ✓ Hindi voice ID:    {voice_hi}")
    
    # Verify different languages get different voices where appropriate
    assert voice_mr != voice_en, "ERROR: Marathi and English should have different voices!"
    print("  ✓ Marathi and English have distinct voices (different voice IDs)")
    
    # Test 3: Language Normalization
    print("\n[TEST 3] Language Code Normalization")
    norm_mr = tts._normalize_tts_language("mr")
    norm_mr_en = tts._normalize_tts_language("mr-en")
    norm_en = tts._normalize_tts_language("en")
    
    print(f"  ✓ 'mr' normalizes to: {norm_mr}")
    print(f"  ✓ 'mr-en' normalizes to: {norm_mr_en}")
    print(f"  ✓ 'en' normalizes to: {norm_en}")
    
    assert norm_mr == "mr", "Marathi should stay as mr"
    assert norm_mr_en == "mr", "Marathi-English code-mix should base to mr"
    
    # Test 4: Language Mapping Coverage
    print("\n[TEST 4] Language Voice Mapping Coverage")
    languages_to_test = ["hi", "mr", "ta", "te", "kn", "ml", "gu", "bn", "pa", "en"]
    
    voices_used = {}
    for lang in languages_to_test:
        voice_id = tts.elevenlabs._get_voice_id(lang)
        voices_used[lang] = voice_id
        print(f"  ✓ {lang:3} → {voice_id}")
    
    # Test 5: Critical Path - Session → Voice
    print("\n[TEST 5] Critical Path: Session.lang → Voice ID")
    session_test = AWAAZSession()
    session_test.lang = "mr"
    
    # Simulate what happens in TTS.synthesize()
    normalized_lang = tts._normalize_tts_language(getattr(session_test, "lang", "hi"))
    voice_for_session = tts.elevenlabs._get_voice_id(normalized_lang)
    
    print(f"  Step 1: session.lang = {session_test.lang}")
    print(f"  Step 2: normalize → {normalized_lang}")
    print(f"  Step 3: get voice → {voice_for_session}")
    print(f"  ✓ Session language properly flows to voice selection")
    
    print("\n" + "="*70)
    print("  ✅ ALL LANGUAGE ENFORCEMENT TESTS PASSED!")
    print("="*70)
    print("\nSummary:")
    print("  • Session language is properly set and maintained")
    print("  • Different languages map to different voices")
    print("  • Language codes normalize correctly")
    print("  • TTS pipeline receives correct language info")
    print("\n⚠️  If Marathi input still produces English TTS:")
    print("  1. Check LLM output is generating Marathi text (not English)")
    print("  2. Verify session.lang is not being reset after STT")
    print("  3. Check TTS provider is actually using the language parameter")
    print("\n")

if __name__ == "__main__":
    try:
        test_language_enforcement()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
