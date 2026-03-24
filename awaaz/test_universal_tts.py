import logging, sys, os
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout
)

# Adjust import path to match your project structure
sys.path.insert(0, "./awaaz")
from src.pipeline.tts import (
    detect_script, is_native_script, get_provider_order, synthesize_speech
)

TESTS = [
    # (lang, sample_text,               expected_script,  expect_native)
    ("mr", "नमस्कार! तुम्ही AWAAZ मध्ये आहात.", "devanagari", True),
    ("hi", "नमस्ते, आप कैसे हैं?",              "devanagari", True),
    ("ta", "வணக்கம், எப்படி இருக்கீங்க?",       "tamil",      True),
    ("te", "నమస్కారం, మీరు ఎలా ఉన్నారు?",        "telugu",     True),
    ("kn", "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?",           "kannada",    True),
    ("ml", "നമസ്കാരം, സുഖമാണോ?",                "malayalam",  True),
    ("bn", "নমস্কার, আপনি কেমন আছেন?",            "bengali",    True),
    ("gu", "નમસ્તે, તમે કેમ છો?",                 "gujarati",   True),
    ("pa", "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?",       "gurmukhi",   True),
    ("ar", "مرحبا، كيف حالك؟",                    "arabic",     True),
    ("ur", "آپ کیسے ہیں؟",                         "arabic",     True),
    ("en", "Hello, how are you?",                  "latin",      True),
    ("fr", "Bonjour, comment allez-vous?",         "latin",      True),
    ("ru", "Здравствуйте, как дела?",              "cyrillic",   True),
    ("zh", "你好，你怎么样？",                       "cjk",        True),
    ("ja", "こんにちは、お元気ですか？",              "cjk",        True),
    # Mismatch test — Latin text passed as Marathi
    ("mr", "Hello world",                          "devanagari", False),
]

print("\n=== SCRIPT DETECTION TESTS ===")
all_pass = True
for lang, text, expected_script, expected_native in TESTS:
    detected = detect_script(text)
    native   = is_native_script(text, lang)
    order    = get_provider_order(lang)
    ok_det   = detected == expected_script
    ok_nat   = native   == expected_native
    status   = "PASS" if (ok_det and ok_nat) else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(
        f"  [{status}] lang={lang:4s} | "
        f"script={detected:12s} ({'ok' if ok_det else 'WRONG, expected '+expected_script}) | "
        f"native={native} ({'ok' if ok_nat else 'WRONG'}) | "
        f"providers={order}"
    )

print(f"\n{'ALL DETECTION TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}\n")

print("=== TTS SYNTHESIS TEST (Marathi) ===")
result = synthesize_speech(
    "नमस्कार! तुम्ही AWAAZ मध्ये आहात.",
    "mr",
    "/tmp/test_mr.wav"
)
print(f"  Provider used : {result['provider']}")
print(f"  Output path   : {result['path']}")
print(f"  Time          : {result['duration_s']}s")
if result['error']:
    print(f"  Error         : {result['error']}")

print("\n=== TTS SYNTHESIS TEST (Tamil) ===")
result = synthesize_speech("வணக்கம்!", "ta", "/tmp/test_ta.wav")
print(f"  Provider: {result['provider']} | Time: {result['duration_s']}s")

print("\n=== TTS SYNTHESIS TEST (Arabic) ===")
result = synthesize_speech("مرحبا!", "ar", "/tmp/test_ar.wav")
print(f"  Provider: {result['provider']} | Time: {result['duration_s']}s")

print("\n=== TTS SYNTHESIS TEST (English) ===")
result = synthesize_speech("Hello, world!", "en", "/tmp/test_en.wav")
print(f"  Provider: {result['provider']} | Time: {result['duration_s']}s")

print("\nDone.")
