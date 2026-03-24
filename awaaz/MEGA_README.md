<!--
  AWAAZ MEGA README
  This file combines all documentation for the AWAAZ system in one place.
  It covers setup, architecture, usage, voice options, phonetic features, API, troubleshooting, and more.
  Last updated: March 2026
-->

# AWAAZ Voice System – Unified Documentation

Welcome to the **AWAAZ** voice system! This document is a comprehensive guide to setting up, understanding, and using the AWAAZ multilingual, phonetic-aware, production-ready voice pipeline for Indian telephony and AI voice applications.

---

## Table of Contents


1. [Quick Start & Setup](#quick-start--setup)
2. [System Architecture Overview](#system-architecture-overview)
3. [Folder Structure](#folder-structure)
4. [Phonetic & Voice Features](#phonetic--voice-features)
5. [TTS Pipeline & Language Support](#tts-pipeline--language-support)
6. [API & FastAPI Usage](#api--fastapi-usage)
7. [Voice Options & Upgrades](#voice-options--upgrades)
8. [Usage Examples & Commands](#usage-examples--commands)
9. [Testing, Validation & Troubleshooting](#testing-validation--troubleshooting)
10. [Advanced: Accent, Phonetics, and Customization](#advanced-accent-phonetics-and-customization)
11. [Deployment & Production](#deployment--production)
12. [References & Further Reading](#references--further-reading)

---


## 1. Quick Start & Setup

### Install Dependencies

```bash
cd awaaz
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --upgrade
```

### Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API keys (GROQ_API_KEY required)
```

### Start the API Server

```bash
chmod +x start_api_server.sh
./start_api_server.sh
# Or run main.py for telephony
python main.py
```

### Run Live Voice Test

```bash
python3 test_live_voice.py --mode mic --output ./response.wav
```

---

## 2. System Architecture Overview

The AWAAZ system is a three-layer voice pipeline:

```
[Voice Input]
   ↓
[Layer 1: Transcription (STT)]
   ↓
[Layer 2: AI Processing (NLP)]
   ↓
[Layer 3: TTS Synthesis]
   ↓
[Voice Output]
```

**Key Features:**
- Multilingual (50+ languages, 19 Indic fully phonetic-aware)
- Unified feminine voice (ritu, or premium voices)
- Phonetic analysis: accent, IPA, English meanings
- Automatic provider fallback (Sarvam, Google, ElevenLabs, Groq, gTTS)
- Real-time, production-ready

---

## 3. Folder Structure

```
awaaz/
├── src/
│   ├── pipeline/
│   │   ├── phonetics.py          # Phonetic analysis engine
│   │   ├── tts.py                # TTS with unified voice, phonetic integration
│   │   ├── stt.py                # STT with phonetic awareness
│   │   ├── nlp.py                # Language model processing
│   │   └── ...
│   ├── session_store.py          # Session management
│   └── ...
├── test_live_voice.py            # Live mic/file test
├── test_phonetic_integration.py  # Phonetic system test suite
├── awaaz_recorder.py             # Audio recording with VAD
├── main.py                       # Main application
├── api_server.py                 # FastAPI app
├── start_api_server.sh           # API server start script
├── config.yaml, requirements.txt, .env, ...
└── Documentation/
    ├── PHONETIC_SYSTEM_IMPLEMENTATION.md
    ├── PHONETIC_QUICK_REFERENCE.md
    ├── FINAL_INTEGRATION_SUMMARY.md
    ├── LIVE_TESTING_INTEGRATION_REPORT.md
    └── ...
```

---

## 4. Phonetic & Voice Features

- **Unified Feminine Voice:** All Indic languages use "ritu" (or premium voices like priya, pooja, simran, etc.)
- **Phonetic Analysis:** IPA conversion, accent detection, English meaning extraction, native script transliteration
- **Accent Adaptation:** TTS pace/pitch/silence adapts to thick village vs standard accent
- **Debug Transparency:** Logs show what system understood (IPA, English, accent)
- **Production Ready:** 100% backward compatible, tested on 19 Indic languages

---

## 5. TTS Pipeline & Language Support

**Universal TTS Pipeline:**
- 50+ languages, automatic provider routing
- Sarvam, Google Cloud, ElevenLabs, Groq, gTTS fallback
- Language-specific voice, pace, pitch, loudness
- Script detection and validation

**Supported Languages:**
- Marathi, Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Punjabi, Odia, Assamese, English, Sinhala, Nepali, Sanskrit, Konkani, Bhojpuri, Maithili, Dogri, Bodo, and more

**Provider Priority:**
1. Google Cloud Neural (if enabled)
2. Sarvam (premium Indian voices)
3. ElevenLabs (multilingual)
4. Groq (fast, free)
5. gTTS (offline fallback)

---

## 6. API & FastAPI Usage

**Start the API server:**
```bash
./start_api_server.sh
# or
python3 api_server.py
```

**Swagger UI:** http://localhost:8000/docs

**Key Endpoints:**
- `/api/v1/transcription/upload` – Upload audio
- `/api/v1/transcription/process-async` – Start STT
- `/api/v1/ai-processing/process-async` – LLM
- `/api/v1/tts/synthesize-async` – TTS
- `/api/v1/tts/download/{job_id}` – Download audio

**Example Python client:**
```python
import requests
with open("recording.wav", "rb") as f:
    resp = requests.post("http://localhost:8000/api/v1/transcription/upload", files={"file": f})
    job_id = resp.json()["job_id"]
# ... see full examples above
```

---

## 7. Voice Options & Upgrades

- **Sarvam Premium Voices:** priya, pooja, simran, shreya, neha, ishita, etc.
- **Google Cloud Neural:** Most natural, free tier available
- **ElevenLabs:** Premium, expressive, emotional control
- **How to switch:** Edit `tts.py` SARVAM_SPEAKER_MAP or set up Google/ElevenLabs in `.env`

---

## 8. Usage Examples & Commands

**Record and save model response:**
```bash
python3 test_live_voice.py --mode mic --output ./response.wav
```
**With language override:**
```bash
python3 test_live_voice.py --mode mic --lang ta --output ./tamil.wav
```
**Batch process files:**
```bash
for audio in inputs/*.wav; do
  output="responses/$(basename $audio .wav)_reply.wav"
  python3 test_live_voice.py --mode file --input "$audio" --output "$output"
done
```

---

## 9. Testing, Validation & Troubleshooting

**Test phonetic system:**
```bash
python3 test_phonetic_integration.py
```
**Check logs for [PHONETIC-DEBUG] and [TTS-START]**

**Common issues:**
- No audio: Check mic permissions
- API error: Verify API keys in .env
- Wrong language: Use --lang to override
- Robotic voice: Switch to premium voices (see above)

---

## 10. Advanced: Accent, Phonetics, and Customization

- **Accent adaptation:**
  - Standard: pace=0.95, pitch=0dB
  - Thick village: pace=0.85, pitch=-5dB, silence=0.2s
- **Phonetic debug:**
  - Enable in `.env`: `ENABLE_PHONETIC_DEBUG=true`
- **Custom accent patterns:**
  - Edit `phonetics.py` REGIONAL_ACCENT_PATTERNS
- **Add new language:**
  - Add script range, speaker config, provider routing

---

## 11. Deployment & Production

**Production deployment:**
```bash
git add awaaz/
git commit -m "feat: AWAAZ system mega-readme"
git push
```
**Docker:**
```bash
docker build -t awaaz-api:latest .
docker run -p 8000:8000 -e GROQ_API_KEY=... awaaz-api:latest
```

---

## 12. References & Further Reading

All previous documentation is now consolidated here. For legacy or deep-dive details, refer to the project history or commit logs.

---

**Status:** PRODUCTION READY  
**Last Updated:** March 2026