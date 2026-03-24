#!/usr/bin/env python3
"""
🎤 AWAAZ Live Interactive Voice Testing
Real-time language detection + STT + TTS demonstration

Features:
- Live mic recording (if available)
- Language auto-detection (ElevenLabs → Sarvam → Local)
- Speech-to-text transcription
- Text-to-speech synthesis with native speakers
- Multi-language support (50+ languages)
"""

import os
import sys
import asyncio
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

from src.pipeline.stt import STTProcessor
from src.pipeline.tts import synthesize_speech

# Try to import recorder
try:
    from awaaz_recorder import Recorder, save_wav, MicCalibrator, TARGET_SR
    RECORDER_AVAILABLE = True
    print("✅ Mic recorder available")
except ImportError:
    RECORDER_AVAILABLE = False
    print("⚠️  Mic recorder not available (use --file option)")
    TARGET_SR = 16000


LANGUAGE_OPTIONS = {
    "1": ("mr", "Marathi (मराठी)"),
    "2": ("hi", "Hindi (हिंदी)"),
    "3": ("ta", "Tamil (தமிழ்)"),
    "4": ("te", "Telugu (తెలుగు)"),
    "5": ("kn", "Kannada (ಕನ್ನಡ)"),
    "6": ("gu", "Gujarati (ગુજરાતી)"),
    "7": ("pa", "Punjabi (ਪੰਜਾਬੀ)"),
    "8": ("bn", "Bengali (বাংলা)"),
    "9": ("as", "Assamese (অসমীয়া)"),
    "10": ("en", "English"),
    "11": ("ar", "Arabic (العربية)"),
    "0": ("auto", "Auto-detect (Recommended)"),
}


async def test_language_detection():
    """Test language detection chain."""
    print("\n" + "="*70)
    print("🧪 LANGUAGE DETECTION TEST")
    print("="*70)
    
    print("\n📋 Available languages for testing:")
    for key, (code, name) in LANGUAGE_OPTIONS.items():
        print(f"  {key}. {name} [{code}]")
    
    choice = input("\nSelect language (or press Enter for auto-detect): ").strip()
    if not choice or choice not in LANGUAGE_OPTIONS:
        choice = "0"
    
    selected_lang, selected_name = LANGUAGE_OPTIONS[choice]
    print(f"\n✓ Selected: {selected_name}")
    
    # Get audio input
    audio_path = await get_audio_input()
    if not audio_path:
        print("❌ Failed to get audio input")
        return
    
    # Initialize STT processor
    print("\n🔄 Initializing STT processor...")
    stt = STTProcessor()
    await stt.load()
    print("✓ STT processor loaded")
    
    # Test language detection
    print("\n🎤 Detecting language...")
    start = time.time()
    detected_lang, confidence = await stt.detect_language(audio_path)
    detect_time = time.time() - start
    
    print(f"⏱️  Detection time: {detect_time*1000:.1f}ms")
    print(f"📊 Detected language: {detected_lang}")
    print(f"📈 Confidence: {confidence:.2%}")
    
    # Test transcription
    print("\n🎙️  Transcribing...")
    start = time.time()
    result = await stt.transcribe(audio_path, language=detected_lang)
    transcribe_time = time.time() - start
    
    if result:
        print(f"⏱️  Transcription time: {transcribe_time*1000:.1f}ms")
        print(f"📝 Text: {result.text}")
        print(f"🔊 Provider: {result.provider}")
        print(f"✓ Confidence: {result.confidence:.2%}")
        
        # Show native script if available
        if result.native_script_text and result.native_script_text != result.text:
            print(f"📖 Native Script: {result.native_script_text}")
    else:
        print("❌ Transcription failed")
        return
    
    # Test TTS with detected language
    print("\n🎵 Synthesizing speech...")
    response_text = f"आपका वाक्य: {result.text[:50]}" if detected_lang in ["mr", "hi"] else f"You said: {result.text[:50]}"
    
    start = time.time()
    tts_result = synthesize_speech(
        text=response_text,
        lang=detected_lang,
        output_path=tempfile.mktemp(suffix=".wav")
    )
    tts_time = time.time() - start
    
    print(f"⏱️  Synthesis time: {tts_time*1000:.1f}ms")
    print(f"🔊 Provider: {tts_result.get('provider')}")
    print(f"📊 Audio size: {tts_result.get('bytes')} bytes")
    if tts_result.get('path') and os.path.exists(tts_result['path']):
        print(f"✓ Audio saved: {tts_result['path']}")
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE")
    print("="*70)


async def test_full_pipeline():
    """Full STT → TTS pipeline test."""
    print("\n" + "="*70)
    print("🚀 FULL PIPELINE TEST (STT → TTS)")
    print("="*70)
    
    # Get audio
    audio_path = await get_audio_input()
    if not audio_path:
        return
    
    # Initialize
    print("\n🔄 Initializing pipeline...")
    stt = STTProcessor()
    await stt.load()
    print("✓ STT processor loaded")
    
    # Language detection + transcription
    print("\n🎤 Processing audio...")
    detected_lang, conf = await stt.detect_language(audio_path)
    result = await stt.transcribe(audio_path, language=detected_lang)
    
    if not result or not result.text:
        print("❌ Transcription failed")
        return
    
    print(f"📝 Transcribed ({detected_lang}): {result.text}")
    
    # Generate TTS response
    print("\n🔄 Generating response...")
    if detected_lang in ["mr", "hi"]:
        response = f"आपने कहा: {result.text[:50]}"
    elif detected_lang in ["ta", "te", "kn", "ml"]:
        response = f"தங்கள் கூற்று: {result.text[:50]}"
    else:
        response = f"You said: {result.text[:50]}"
    
    tts_result = synthesize_speech(
        text=response,
        lang=detected_lang,
        output_path=tempfile.mktemp(suffix=".wav")
    )
    
    print(f"🔊 Response ({tts_result['provider']}): {response}")
    if tts_result.get('path') and os.path.exists(tts_result['path']):
        print(f"✓ Audio generated: {tts_result['path']}")
    
    # Try to play audio if available
    try:
        import subprocess
        if sys.platform == "darwin":  # macOS
            subprocess.run(["afplay", tts_result['path']], check=True, timeout=10)
        elif sys.platform == "linux":  # Linux
            subprocess.run(["aplay", tts_result['path']], check=True, timeout=10)
        print("✓ Audio played")
    except:
        print("⚠️  Could not auto-play audio")
    
    print("\n✅ Pipeline test complete")


async def get_audio_input():
    """Get audio from mic or file."""
    print("\n" + "-"*70)
    print("🎙️  AUDIO INPUT")
    print("-"*70)
    
    if RECORDER_AVAILABLE:
        choice = input("Record from mic (m) or use test file (f)? [m/f]: ").strip().lower()
    else:
        choice = "f"
        print("Using file mode (mic not available)")
    
    if choice == "m" and RECORDER_AVAILABLE:
        return await record_from_mic()
    else:
        return await select_test_file()


async def record_from_mic(duration=5):
    """Record audio from microphone."""
    print(f"\n🎙️  Recording for {duration} seconds...")
    print("Start speaking now...")
    
    try:
        recorder = Recorder(sample_rate=TARGET_SR)
        
        # Calibrate if available
        try:
            calibrator = MicCalibrator()
            rec_level = calibrator.calibrate(duration=2)
            print(f"🎚️  Recording level: {rec_level:.1f}%")
        except:
            pass
        
        # Record
        start = time.time()
        samples = recorder.record(duration=duration)
        elapsed = time.time() - start
        
        if samples and len(samples) > 0:
            output_path = tempfile.mktemp(suffix=".wav")
            save_wav(samples, output_path, sample_rate=TARGET_SR)
            print(f"✓ Recorded {elapsed:.1f}s: {output_path}")
            return output_path
        else:
            print("❌ No audio recorded")
            return None
            
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        return None


async def select_test_file():
    """Select a test audio file."""
    test_dir = BASE_DIR / "tests" / "audio"
    
    if test_dir.exists():
        files = list(test_dir.glob("*.wav")) + list(test_dir.glob("*.mp3"))
        if files:
            print(f"\n📁 Found {len(files)} test files:")
            for i, f in enumerate(files[:10], 1):
                print(f"  {i}. {f.name}")
            
            try:
                choice = int(input("Select file (1-10): ")) - 1
                if 0 <= choice < len(files):
                    print(f"✓ Selected: {files[choice].name}")
                    return str(files[choice])
            except:
                pass
    
    # Fallback
    print("\n📝 Enter audio file path (or press Enter for demo):")
    path = input().strip()
    
    if path and os.path.exists(path):
        print(f"✓ Using: {path}")
        return path
    
    print("⚠️  No file selected")
    return None


async def main():
    """Interactive menu."""
    print("\n" + "="*70)
    print("🎤 AWAAZ LIVE VOICE TESTING SYSTEM")
    print("="*70)
    print("\nSupported Languages: 50+ (Hindi, Marathi, Tamil, Telugu, English, etc.)")
    print("STT Providers: ElevenLabs (primary) → Sarvam → Local Whisper")
    print("TTS Providers: Sarvam (Indic) → ElevenLabs → Groq → gTTS")
    
    while True:
        print("\n" + "-"*70)
        print("📋 TESTS")
        print("-"*70)
        print("1. Language Detection Test")
        print("2. Full Pipeline Test (STT → TTS)")
        print("3. Exit")
        
        choice = input("\nSelect test (1-3): ").strip()
        
        if choice == "1":
            await test_language_detection()
        elif choice == "2":
            await test_full_pipeline()
        elif choice == "3":
            print("\n✓ Goodbye!")
            break
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⊘ Interrupted by user")
        sys.exit(0)
