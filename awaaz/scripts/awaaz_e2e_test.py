#!/usr/bin/env python3
"""
AWAAZ End-to-End Test Engine Loop
Reads a voice recording -> Groq Whisper STT -> LLM Engine -> ElevenLabs TTS

To test with a WAV file: record audio or drop in audio.wav to pipeline it.
"""

import os
import sys
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Ensure we use our core architecture modules 
sys.path.insert(0, str(Path(__file__).parent / "awaaz"))

from src.pipeline.stt import STTProcessor
from src.pipeline.nlp import ModelProcessor
from src.pipeline.tts import TTSProcessor, synthesize_speech
from src.session_store import AWAAZSession
import tempfile

try:
    from awaaz_recorder import MicCalibrator, VADEngine, Recorder, save_wav, TARGET_SR
except ImportError:
    TARGET_SR = 16000

load_dotenv(dotenv_path=Path(__file__).parent / "awaaz" / ".env")

async def run_pipeline(audio_file_path: str):
    print("\n" + "="*50)
    print("  AWAAZ Full Pipeline Processing")
    print("="*50)

    # Mock Session for the engine
    session = AWAAZSession(call_id="test_session_" + str(int(time.time())))
    session.lang = "hi"
    session.state = "active"
    
    # 1. Init Modules
    stt = STTProcessor()
    nlp = ModelProcessor(provider="groq", model="llama-3.1-8b-instant")
    tts = TTSProcessor()

    print("[SYSTEM] Loading Models...")
    await stt.load()
    if hasattr(tts, 'load') and asyncio.iscoroutinefunction(tts.load):
        await tts.load()

    # 2. STT Route
    print(f"\n[1/3] Transcribing Audio: {audio_file_path}")
    stt_start = time.time()
    
    transcript = await stt.transcribe(audio_file_path, language="auto")
    
    if not transcript:
        print("[ERROR] STT failed to transcribe any text.")
        return

    print(f"  -> Transcription Time: {time.time() - stt_start:.2f}s")
    print(f"  -> Recognized Text: '{transcript}'")

    # 3. NLP Route
    print("\n[2/3] Generating AI Response...")
    nlp_start = time.time()
    
    response_text, call_status = await nlp.process_turn(session, transcript)
    print(f"  -> NLP Generation Time: {time.time() - nlp_start:.2f}s")
    print(f"  -> AI Script (Response): '{response_text}'")

    # 4. TTS Route
    print("\n[3/3] Synthesizing Speech (ElevenLabs Indian Tone)...")
    tts_start = time.time()
    
    out_wav = f"/tmp/{session.call_id}_reply.wav"
    result = synthesize_speech(response if 'response' in locals() else test_text, getattr(session, 'lang', 'hi'), getattr(session, 'output_path', out_wav if 'out_wav' in locals() else 'out.wav'))
    success = result['path'] is not None
    
    if success:
        print(f"  -> Synthesis Time: {time.time() - tts_start:.2f}s")
        print(f"  -> Audio Saved: {out_wav}")
        if os.uname().sysname == "Darwin":
            os.system(f"afplay {out_wav}")
    else:
        print("[ERROR] Failed to synthesize TTS audio.")

def record_test_audio():
    print("\n[REC] Recording input audio from Microphone...")
    cal = MicCalibrator().calibrate()
    rec = Recorder(vad=VADEngine(aggressiveness=2), device_index=None)
    
    print(f"[REC] Speak now (max 5s)...")
    audio = rec.record(max_duration_s=5, silence_ms=1000, calibration=cal)
    if audio is None: raise RuntimeError("No audio captured")
        
    p = tempfile.mktemp(suffix=".wav")
    save_wav(audio, TARGET_SR, p)
    return p

def main():
    path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".wav") else None
    if not path or not os.path.exists(path):
        path = record_test_audio()

    asyncio.run(run_pipeline(path))

if __name__ == "__main__":
    main()
