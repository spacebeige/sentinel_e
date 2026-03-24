#!/usr/bin/env python3
"""
Generate test audio with actual speech using established TTS services
"""

import os
import sys
import tempfile
from pathlib import Path

try:
    from gtts import gTTS
except ImportError:
    print("Installing gTTS dependency...")
    os.system("pip install -q gtts")
    from gtts import gTTS

# Create test speech
test_text = "Namaste. How can I help you today?"
print(f"Generating test audio with speech: '{test_text}'")

try:
    # Use gTTS to generate Hindi speech
    tts = gTTS(text=test_text, lang='hi', slow=False)
    
    output_file = Path(__file__).parent / "test_audio_speech.wav"
    
    # gTTS generates mp3, we need to convert to wav via temp file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tts.save(tmp.name)
        
        # Try to convert mp3 to wav using ffmpeg if available
        try:
            os.system(f"ffmpeg -i {tmp.name} -acodec pcm_s16le -ar 16000 {output_file} -y 2>/dev/null")
            if os.path.exists(output_file):
                size_kb = os.path.getsize(output_file) / 1024
                print(f"✓ Test audio created: {output_file}")
                print(f"  Size: {size_kb:.1f} KB")
                print(f"  Format: WAV (PCM 16-bit, 16kHz)")
                os.unlink(tmp.name)
                sys.exit(0)
        except Exception as e:
            print(f"Could not convert to WAV: {e}")
        
        # Fallback: save as mp3
        output_file_mp3 = Path(__file__).parent / "test_audio_speech.mp3"
        os.rename(tmp.name, output_file_mp3)
        print(f"✓ Test audio created: {output_file_mp3}")
        size_kb = os.path.getsize(output_file_mp3) / 1024
        print(f"  Size: {size_kb:.1f} KB")
        print(f"  Format: MP3 (to run test, use: python3 test_e2e_pipeline.py test_audio_speech.mp3)")

except Exception as e:
    print(f"❌ Error generating audio: {e}")
    sys.exit(1)
