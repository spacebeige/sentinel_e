#!/usr/bin/env python3
"""
Test ritu voice - clean & simple with Sarvam + ElevenLabs fallback.
All 20+ languages supported without enhancements.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline.tts import synthesize_speech
import time

# All supported languages
test_languages = {
    "hi": "नमस्ते! आप कैसे हैं? आज बहुत सुंदर दिवस है।",
    "mr": "नमस्कार! तुम्ही कसे आहात? आज छान दिवस आहे.",
    "ta": "வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?",
    "te": "నమస్కారం! మీరు ఎలా ఉన్నారు?",
    "kn": "ನಮಸ್ಕಾರ! ನೀವು ಹೇಗಿರುವಿರಿ?",
    "ml": "നമസ്കാരം! നിങ്ങൾ എങ്ങനെയിരിക്കുന്നു?",
    "bn": "নমস্কার! আপনি কেমন আছেন?",
    "gu": "નમસ્તે! તમે કેવા છો?",
    "pa": "ਨਮਸ੍ਤੇ! ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?",
    "or": "ନମସ୍କାର! ଆପଣ କେମିତି ଅଛନ୍ତି?",
    "as": "নমস্কাৰ! আপুনি কেনে আছেন?",
    "en": "Hello! How are you? Today is a beautiful day.",
    "si": "ඔබට සුබෝ! ඔබ කොහොමද?",
    "kok": "नमस्कार! तुम कसे आहात?",  # Konkani
    "bho": "नमस्कार! आप कइसे हैं?",   # Bhojpuri
    "mai": "नमस्कार! आप कइसे छथि?",   # Maithili
    "doi": "नमश्कार! तूँ कइसा ए?",    # Dogri
    "awa": "नमस्कार! अरे कइसे हौ?",  # Awadhi
    "mwr": "नमस्कार! आप कइसे हो?",   # Marwadi
    "bgc": "नमस्कार! तू कइसा ए?",    # Haryanvi
}

print("=" * 70)
print("  RITU VOICE - ALL LANGUAGES TEST")
print("  Simple approach: Sarvam (ritu) + ElevenLabs fallback")
print("=" * 70)
print()

success_count = 0
failed_count = 0

for lang, text in test_languages.items():
    output_file = f"./multilang_test_{lang}.wav"
    
    try:
        t0 = time.time()
        result = synthesize_speech(text, lang, output_file)
        duration = time.time() - t0
        
        if result["error"]:
            print(f"✗ [{lang.upper()}] {result['error']}")
            failed_count += 1
        else:
            print(f"✓ [{lang.upper()}] {result['provider']:12s} | {duration:.2f}s")
            success_count += 1
    except Exception as e:
        print(f"✗ [{lang.upper()}] Exception: {e}")
        failed_count += 1

print()
print("=" * 70)
print(f"Results: {success_count} ✓ | {failed_count} ✗")
print("=" * 70)
print()
print("All 20+ languages tested with:")
print("  • Voice: ritu (single consistent female voice)")
print("  • Provider: Sarvam primary, ElevenLabs fallback")
print("  • Approach: Simple, clean, no enhancements")
print()
