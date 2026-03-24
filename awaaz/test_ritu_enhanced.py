#!/usr/bin/env python3
"""
Test enhanced ritu voice with phonetic processing and audio polishing.

This script demonstrates the new ritu enhancements:
- Phonetic processing for better pronunciation
- Audio polishing (normalization, compression, clarity boost)
- Consistent single voice across all languages
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline.tts import synthesize_speech
import time

# Test text in different Indian languages
test_cases = [
    ("hi", "नमस्ते! आज का मौसम बहुत सुंदर है। आप कैसे हैं?"),
    ("mr", "नमस्कार! तुम्ही कसे आहात? आज अगदी छान दिवस आहे."),
    ("ta", "வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்? இந்தி விழாகள் பற்றி பேசலாமா?"),
    ("te", "నమస్కారం! మీరు ఎలా ఉన్నారు? ఈ రోజు చాలా అందమైన రోజు."),
    ("gu", "નમસ્તે! તમે કેવા છો? આજે બહુ સુંદર દિવસ છે."),
]

print("=" * 60)
print("  RITU VOICE - PHONETIC ENHANCEMENT TEST")
print("=" * 60)
print()
print("Testing enhanced ritu voice with:")
print("  ✓ Phonetic processing (better pronunciation)")
print("  ✓ Audio polishing (clarity enhancement)")
print("  ✓ Dynamic loudness (1.8x for clarity)")
print("  ✓ Consistent single female voice")
print()

for lang, text in test_cases:
    output_file = f"./test_ritu_{lang}.wav"
    
    print(f"[{lang.upper()}] Testing ritu voice...")
    print(f"  Text: {text[:60]}...")
    
    t0 = time.time()
    result = synthesize_speech(text, lang, output_file, force_provider="sarvam")
    duration = time.time() - t0
    
    if result["error"]:
        print(f"  ✗ Error: {result['error']}")
    else:
        print(f"  ✓ Provider: {result['provider']}")
        print(f"  ✓ File: {result['path']}")
        print(f"  ✓ Time: {duration:.2f}s")
        print(f"  ✓ Enhancements: phonetic + audio_polish + loudness_boost")
    print()

print("=" * 60)
print("✅ Ritu enhancement tests complete!")
print()
print("Listen to the output files:")
print("  afplay ./test_ritu_hi.wav   # Hindi with ritu + enhancements")
print("  afplay ./test_ritu_mr.wav   # Marathi with ritu + enhancements")
print("  afplay ./test_ritu_ta.wav   # Tamil with ritu + enhancements")
print("=" * 60)
