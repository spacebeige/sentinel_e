# Pan-India Telephonic IVR System with LLM Orchestration

This repository now includes a complete IVR (Interactive Voice Response) system implementation for pan-India telephonic support with regional language support powered by LLM orchestration.

## Overview

The IVR system provides:
- **Regional Language Support**: Hindi (Mumbai, Delhi, Rural UP), Tamil, Marathi, English
- **LLM-Powered Responses**: Culturally and linguistically appropriate responses
- **Intent Classification**: Automatic categorization of caller needs
- **Urgency Detection**: Priority-based routing and response generation
- **WebSocket Audio Streaming**: Real-time audio processing

## Architecture

```
┌──────────┐     ┌──────────────┐     ┌───────────────────────┐
│  Caller  │────▶│ Asterisk PBX │────▶│   Colab Server        │
│          │◀────│  (Raw Audio) │◀────│ • Faster-Whisper STT  │
└──────────┘     └──────────────┘     │ • IndicBERT (Intent)  │
                                      │ • Wav2Vec (Accent)    │
                                      └───────────┬───────────┘
                                                  │
                                                  ▼
                                      ┌───────────────────────┐
                                      │  IVR LLM Orchestrator │
                                      │  (Sentinel-E Backend) │
                                      │ • Regional Profiling  │
                                      │ • Response Generation │
                                      └───────────┬───────────┘
                                                  │
                                                  ▼
                                      ┌───────────────────────┐
                                      │   Regional TTS Engine │
                                      │ • Hindi, Tamil, etc.  │
                                      │ • Emotion Control     │
                                      └───────────────────────┘
```

## Components

### 1. IVR Module (`backend/ivr/`)

#### `ivr_prompts.py`
- **IVR_SYSTEM_PROMPT**: Complete system prompt for LLM orchestration
- **REGIONAL_PROFILES**: Language/dialect mapping for 6+ regional profiles
- **INTENT_CATEGORIES**: 5 pre-defined intent types with urgency levels

#### `ivr_orchestrator.py`
- **IVROrchestrator**: Main class for LLM integration
  - Processes transcription + intent data from Colab pipeline
  - Generates culturally appropriate responses
  - Supports regional dialect matching
  - Handles urgency-based emotion selection

#### `ivr_client.py`
- **WebSocket Client**: Streams audio to Colab server
  - Supports 16kHz, 16-bit mono WAV files
  - Real-time chunk-based streaming
  - Non-blocking receive for pipeline responses
  - Automatic LLM processing of transcriptions

### 2. Example Script (`example_ivr_client.py`)

Demonstrates:
- Standalone orchestrator testing (no WebSocket)
- WebSocket client usage with Colab server
- Regional language response generation
- Intent-based routing

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

The key dependency is `websockets>=12.0` (already added to `requirements.txt`).

### 2. Test Standalone Orchestrator

```bash
python example_ivr_client.py
```

Output:
```
TEST 1: IVR Orchestrator Standalone Test
--- Test Case 1: Network Issue (Hindi - Mumbai) ---
Input: Mera network kaam nahi kar raha hai, please fix it.
Response: Mafi chahenge sir. Hum aapke network issue ko abhi check kar rahe hain.
Action: handoff_to_human
Emotion: apologetic
...
```

### 3. Test with Colab WebSocket Server

**Prerequisites:**
1. Colab notebook with Faster-Whisper WebSocket endpoint
2. ngrok URL for the Colab server
3. Test audio file (16kHz, 16-bit mono WAV)

**Usage:**

```bash
# Update the WebSocket URL in backend/ivr/ivr_client.py
# COLAB_WEBSOCKET_URL = "ws://YOUR_NGROK_URL.ngrok.app/stream"

# Run with WebSocket
python example_ivr_client.py --websocket \
  --ws-url ws://your-ngrok-url.ngrok.app/stream \
  --audio-file test_hindi_complaint.wav
```

## Regional Language Support

### Supported Profiles

| Profile | Language | Dialect | Use Case |
|---------|----------|---------|----------|
| `hi_mumbai` | Hindi | Colloquial | Urban Mumbai callers |
| `hi_delhi` | Hindi | Standard | Delhi NCR callers |
| `hi_rural_up` | Hindi | Formal | Rural UP callers |
| `ta_chennai` | Tamil | Standard | Chennai callers |
| `mr_mumbai` | Marathi | Standard | Marathi-speaking Mumbai |
| `en_bangalore` | English | Indian | Bangalore/tech callers |

### Intent Categories

| Category | Urgency | Example Response (Hindi) |
|----------|---------|-------------------------|
| Network_Issue | High | "Mafi chahenge sir. Hum aapke network issue ko abhi check kar rahe hain." |
| Billing_Query | Medium | "Ji bilkul. Main aapka bill detail abhi dekh raha hoon." |
| Account_Support | Medium | "Zaroor. Aapke account ki jaankari de raha hoon." |
| Service_Request | Low | "Theek hai. Main aapki request register kar raha hoon." |
| General_Inquiry | Low | "Ji kahiye. Main aapki madad karne ke liye yahaan hoon." |

## Input/Output Schema

### Input (from Colab Pipeline)

```json
{
  "text": "Mera network kaam nahi kar raha hai, please fix it.",
  "language": "hi",
  "detected_profile": {
    "lang": "hi_mumbai",
    "confidence": 0.89
  },
  "intent_data": {
    "category": "Network_Issue",
    "urgency": "High",
    "confidence": 0.92
  }
}
```

### Output (to TTS Engine)

```json
{
  "tts_script": "Mafi chahenge sir. Hum aapke network issue ko abhi check kar rahe hain.",
  "suggested_action": "handoff_to_human",
  "tts_overrides": {
    "emotion": "apologetic"
  }
}
```

## LLM Execution Rules

The orchestrator follows these rules:

1. **EMPATHY & URGENCY**: High-urgency issues get immediate acknowledgment
2. **LINGUISTIC MIRRORING**: Responses match caller's regional dialect
3. **TTS OPTIMIZATION**: Acronyms written phonetically ("A T and T" not "AT&T")
4. **BREVITY**: Responses under 15 seconds (2-3 short sentences)

## Integration with Sentinel-E

The IVR module leverages existing Sentinel-E infrastructure:

- ✅ **LLM Models**: Can use existing 9-model registry
- ✅ **Orchestration**: Integrates with `MetaCognitiveOrchestrator`
- ✅ **Database**: Can extend for call logging
- ✅ **Auth**: Can use existing JWT authentication
- ✅ **API Framework**: Built on FastAPI (same as backend)

### Future Integration Points

```python
# backend/ivr/ivr_orchestrator.py
from backend.metacognitive.orchestrator import MetaCognitiveOrchestrator

async def process_with_llm(self, pipeline_data):
    mco = MetaCognitiveOrchestrator()
    query = self._build_ivr_query(pipeline_data)
    result = await mco.run(query, mode="standard")
    return self._parse_llm_response(result)
```

## API Endpoints (Future)

Planned FastAPI endpoints:

```python
# POST /api/ivr/call/start
# POST /api/ivr/call/process
# POST /api/ivr/call/end
# WS /api/ivr/stream (WebSocket for audio)
```

## Testing

### Unit Tests

```bash
cd backend
python -m pytest ivr/tests/
```

### Manual Testing

**Test 1: Standalone Orchestrator**
```bash
python example_ivr_client.py
```

**Test 2: WebSocket Client**
```bash
python example_ivr_client.py --websocket \
  --ws-url ws://your-colab-url.ngrok.app/stream \
  --audio-file test_audio.wav
```

**Test 3: Python API**
```python
from backend.ivr import IVROrchestrator
import asyncio

orchestrator = IVROrchestrator()
pipeline_data = {
    "text": "Mera internet down hai",
    "language": "hi",
    "detected_profile": {"lang": "hi_mumbai"},
    "intent_data": {"category": "Network_Issue", "urgency": "High"}
}

response = asyncio.run(orchestrator.process_pipeline_data(pipeline_data))
print(response["tts_script"])
# Output: "Mafi chahenge sir. Hum aapke network issue ko abhi check kar rahe hain."
```

## Audio File Requirements

For testing with real audio:

**Format Requirements:**
- Sample Rate: 16kHz (or 8kHz for telephony)
- Channels: 1 (mono)
- Bit Depth: 16-bit
- Format: WAV (PCM)

**Convert Audio with `sox`:**
```bash
# Convert MP3 to compatible WAV
sox input.mp3 -r 16000 -c 1 -b 16 test_hindi_complaint.wav

# Convert from any format
sox input.m4a -r 16000 -c 1 -b 16 output.wav
```

## Development Roadmap

- [x] Core IVR module structure
- [x] Regional language profiles (6 profiles)
- [x] Intent categorization (5 categories)
- [x] WebSocket audio streaming client
- [x] LLM orchestrator integration (mock)
- [x] Example scripts and documentation
- [x] Standalone testing
- [ ] Real LLM API integration (Claude Opus 4.6)
- [ ] FastAPI WebSocket server endpoints
- [ ] Database models for call logging
- [ ] Asterisk PBX integration (SIP/VOIP)
- [ ] Regional TTS engine integration
- [ ] Real-time sentiment analysis
- [ ] Call recording and audit logging
- [ ] Production deployment guide

## Documentation

- **Detailed Guide**: See `backend/ivr/README.md`
- **API Reference**: See `backend/ivr/` module docstrings
- **Examples**: See `example_ivr_client.py`

## License

Part of the Sentinel-E project.

## Support

For issues or questions, please open a GitHub issue in the Sentinel-E repository.
