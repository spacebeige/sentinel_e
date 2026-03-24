#!/usr/bin/env python3
"""
Test TTS providers individually to debug which ones work
"""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "awaaz"))

async def test_tts():
    print("Testing TTS Providers\n")
    
    test_text = "Namaste, this is a test."
    
    # Test 1: gTTS (should work)
    print("[1] Testing gTTS...")
    try:
        from gtts import gTTS
        output = tempfile.mktemp(suffix=".mp3")
        tts = gTTS(text=test_text, lang='hi', slow=False)
        tts.save(output)
        size_kb = os.path.getsize(output) / 1024
        print(f"    ✓ gTTS works! Generated {size_kb:.1f} KB")
        os.unlink(output)
    except Exception as e:
        print(f"    ✗ gTTS failed: {e}")
    
    # Test 2: ElevenLabs (needs API key)
    print("\n[2] Testing ElevenLabs...")
    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    if api_key:
        try:
            import aiohttp
            from src.pipeline.tts import ElevenLabsTTS
            el = ElevenLabsTTS()
            output = tempfile.mktemp(suffix=".mp3")
            success = await el.synthesize(test_text, output)
            if success:
                size_kb = os.path.getsize(output) / 1024
                print(f"    ✓ ElevenLabs works! Generated {size_kb:.1f} KB")
                os.unlink(output)
            else:
                print(f"    ✗ ElevenLabs returned False")
        except Exception as e:
            print(f"    ✗ ElevenLabs failed: {e}")
    else:
        print("    ⊘ ElevenLabs API key not configured")
    
    # Test 3: Coqui TTS (needs model)
    print("\n[3] Testing Coqui TTS...")
    try:
        from TTS.api import TTS
        print("    Coqui TTS found, but skipping full model load (too slow)")
    except ImportError:
        print("    ⊘ Coqui TTS not installed (pip install tts)")
    
    # Test 4: Full TTSProcessor
    print("\n[4] Testing TTSProcessor with fallback chain...")
    try:
        from src.pipeline.tts import TTSProcessor
        from src.session_store import AWAAZSession
        import asyncio
        
        tts_proc = TTSProcessor()
        await tts_proc.load()
        
        session = AWAAZSession()
        session.lang = "hi"
        session.gtts_lang = "hi"
        
        output = tempfile.mktemp(suffix=".wav")
        success = await tts_proc.synthesize(test_text, session, output)
        
        if success and os.path.exists(output):
            size_kb = os.path.getsize(output) / 1024
            print(f"    ✓ TTSProcessor works! Generated {size_kb:.1f} KB")
            os.unlink(output)
        else:
            print(f"    ✗ TTSProcessor failed")
    except Exception as e:
        print(f"    ✗ TTSProcessor error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_tts())
