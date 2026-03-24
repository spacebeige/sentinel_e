#!/usr/bin/env python3
"""
AWAAZ Voice Pipeline Integration Test
Tests complete flow: STT → LLM → TTS with sample audio
"""

import sys
import os
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import tempfile

sys.path.insert(0, str(Path(__file__).parent / "awaaz"))

from src.pipeline.stt import STTProcessor
from src.pipeline.nlp import ModelProcessor
from src.pipeline.tts import TTSProcessor, synthesize_speech
from src.session_store import AWAAZSession, SessionStore

# Load environment
load_dotenv(dotenv_path=Path(__file__).parent / "awaaz" / ".env")

async def run_e2e_pipeline(audio_file: str) -> bool:
    """Execute complete voice pipeline"""
    print("\n" + "="*70)
    print("  AWAAZ Voice AI Pipeline - Full Integration Test")
    print("="*70)
    
    # Create session
    session = AWAAZSession()
    session.channel_id = f"test_channel_{int(time.time())}"
    session.lang = "hi"
    session.state = "active"
    
    print("\n[INIT] Loading pipeline components...")
    
    # Initialize STT
    stt = STTProcessor()
    await stt.load()
    print("  ✓ STT Pipeline loaded (Groq Whisper Large V3 + fallbacks)")
    
    # Initialize LLM
    nlp = ModelProcessor(provider="groq", model="llama-3.1-8b-instant")
    print("  ✓ LLM Pipeline loaded (Groq llama-3.1-8b)")
    
    # Initialize TTS
    tts = TTSProcessor()
    await tts.load()
    print("  ✓ TTS Pipeline loaded (Groq Audio Synthesis + fallbacks)")
    
    # Step 1: STT
    print(f"\n[1/3] SPEECH-TO-TEXT")
    print(f"      Input file: {audio_file}")
    stt_start = time.time()
    
    if not os.path.exists(audio_file):
        print(f"      ✗ File not found!")
        return False
    
    result = await stt.transcribe(audio_file, language="auto")
    stt_time = time.time() - stt_start
    
    if isinstance(result, str):
        transcript = result
        print(f"      ✓ Transcribed: '{transcript}'")
    elif hasattr(result, 'text'):
        transcript = result.text
        provider = getattr(result, 'provider', 'unknown')
        confidence = getattr(result, 'confidence', 0.0)
        lang = getattr(result, 'detected_language', 'auto')
        print(f"      ✓ Transcribed: '{transcript}'")
        print(f"        Provider: {provider} | Confidence: {confidence:.2%} | Language: {lang}")
    else:
        transcript = str(result)
        print(f"      ✓ Transcribed: '{transcript}'")
    
    print(f"      ⏱️  Time: {stt_time:.2f}s")
    
    if not transcript or len(transcript.strip()) == 0:
        print("      ✗ No text transcribed!")
        return False
    
    # Step 2: NLP/LLM
    print(f"\n[2/3] LANGUAGE MODEL PROCESSING")
    nlp_start = time.time()
    
    try:
        response = await nlp.generate(transcript, session)
        nlp_time = time.time() - nlp_start
        print(f"      ✓ LLM Response: '{response}'")
        print(f"      ⏱️  Time: {nlp_time:.2f}s")
    except Exception as e:
        print(f"      ✗ NLP error: {e}")
        return False
    
    # Step 3: TTS
    print(f"\n[3/3] TEXT-TO-SPEECH")
    tts_start = time.time()
    
    out_wav = tempfile.mktemp(suffix=".wav", prefix="awaaz_")
    try:
        result = synthesize_speech(response if 'response' in locals() else test_text, getattr(session, 'lang', 'hi'), getattr(session, 'output_path', out_wav if 'out_wav' in locals() else 'out.wav'))
        success = result['path'] is not None
        tts_time = time.time() - tts_start
        
        print(f"      TTS Result: success={success}, file_exists={os.path.exists(out_wav)}")
        
        if success and os.path.exists(out_wav):
            size_kb = os.path.getsize(out_wav) / 1024
            print(f"      ✓ Audio synthesized: {out_wav}")
            print(f"        Size: {size_kb:.1f} KB")
            print(f"      ⏱️  Time: {tts_time:.2f}s")
            
            # Auto-play on macOS
            if os.uname().sysname == "Darwin":
                print("\n      🔊 Playing audio response...")
                os.system(f"afplay {out_wav} 2>/dev/null")
            
            return True
        else:
            print(f"      ✗ TTS synthesis failed")
            return False
    except Exception as e:
        print(f"      ✗ TTS error: {e}")
        return False

async def main():
    """Main entry point"""
    
    # Check CLI argument first
    if len(sys.argv) > 1:
        test_audio = sys.argv[1]
    else:
        # Try to find a sample audio file
        test_audio = None
        for candidate in [
            "test_audio.wav",
            "test_audio_speech.mp3",
            "sample.wav",
            "data/sample_audio.wav",
            "assets/sample.wav"
        ]:
            if os.path.exists(candidate):
                test_audio = candidate
                break
    
    if not test_audio:
        print("\n❌ No test audio file provided")
        print("\nUsage: python3 test_e2e_pipeline.py [audio_file.wav]")
        print("\nOr place a test audio file as:")
        print("   • test_audio.wav")
        print("   • sample.wav")
        print("   • data/sample_audio.wav")
        sys.exit(1)
    
    success = await run_e2e_pipeline(test_audio)
    
    print("\n" + "="*70)
    if success:
        print("✅ Pipeline test SUCCESSFUL!")
    else:
        print("❌ Pipeline test FAILED")
    print("="*70 + "\n")
    
    return 0 if success else 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
