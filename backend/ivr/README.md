# IVR Telephonic System - Pan-India Interactive Voice Response

This module implements a telephonic IVR system with LLM-powered response generation for regional language support across India.

## Architecture Overview

```
┌──────────┐     ┌──────────────┐     ┌───────────────┐     ┌─────────┐     ┌──────────┐
│  Caller  │────▶│ Asterisk PBX │────▶│ Colab Server  │────▶│ IVR LLM │────▶│ Regional │
│          │     │  (Raw Audio) │     │ (Whisper STT) │     │Orchestr.│     │   TTS    │
└──────────┘     └──────────────┘     └───────────────┘     └─────────┘     └──────────┘
                                             │                     │                │
                                             ├─ IndicBERT (Intent)│                │
                                             ├─ Wav2Vec (Accent)  │                │
                                             └──────────┬──────────┘                │
                                                       │                            │
                                                       ▼                            │
                                             ┌──────────────────┐                  │
                                             │  ivr_client.py   │◀─────────────────┘
                                             │ (WebSocket Client)│
                                             └──────────────────┘
```

## Components

### 1. `ivr_prompts.py`
Contains the system prompt and configuration for the LLM orchestrator:
- **IVR_SYSTEM_PROMPT**: Complete system prompt for Opus 4.6/Claude orchestrator
- **REGIONAL_PROFILES**: Mapping of regional accents to linguistic preferences
  - `hi_mumbai`: Colloquial Hindi (Mumbai)
  - `hi_delhi`: Standard Hindi (Delhi)
  - `hi_rural_up`: Formal Hindi (Rural UP)
  - `ta_chennai`: Tamil (Chennai)
  - `mr_mumbai`: Marathi (Mumbai)
  - `en_bangalore`: Indian English (Bangalore)
- **INTENT_CATEGORIES**: Pre-defined intent mappings with urgency levels
  - Network_Issue (High urgency)
  - Billing_Query (Medium urgency)
  - Account_Support (Medium urgency)
  - Service_Request (Low urgency)
  - General_Inquiry (Low urgency)

### 2. `ivr_orchestrator.py`
LLM integration for IVR response generation:
- **IVROrchestrator**: Main class for processing pipeline data
  - `process_pipeline_data()`: Processes transcription/intent data
  - `process_with_llm()`: Integration point for actual LLM API calls
  - Generates culturally appropriate responses based on regional profiles

### 3. `ivr_client.py`
WebSocket client for audio streaming:
- **stream_audio_to_pipeline()**: Streams audio to Colab WebSocket server
- **mock_llm_processing()**: Processes pipeline data through orchestrator
- **run_ivr_client()**: Main entry point

## Usage

### Basic Usage

```python
from backend.ivr import run_ivr_client

# Run with default configuration
run_ivr_client()
```

### Advanced Usage

```python
from backend.ivr import stream_audio_to_pipeline
import asyncio

# Custom WebSocket URL and audio file
websocket_url = "ws://your-ngrok-url.ngrok.app/stream"
audio_file = "/path/to/your/audio.wav"

asyncio.run(stream_audio_to_pipeline(websocket_url, audio_file))
```

### As a Module

```python
from backend.ivr import IVROrchestrator
import asyncio

# Initialize orchestrator
orchestrator = IVROrchestrator()

# Process pipeline data
pipeline_data = {
    "text": "Mera network kaam nahi kar raha hai",
    "language": "hi",
    "detected_profile": {"lang": "hi_mumbai"},
    "intent_data": {"category": "Network_Issue", "urgency": "High"}
}

response = asyncio.run(orchestrator.process_pipeline_data(pipeline_data))
print(response)
# Output: {
#   "tts_script": "Mafi chahenge sir. Hum aapke network issue ko abhi check kar rahe hain.",
#   "suggested_action": "handoff_to_human",
#   "tts_overrides": {"emotion": "apologetic"}
# }
```

## Configuration

### WebSocket Server URL

Update the `COLAB_WEBSOCKET_URL` in `ivr_client.py` or pass it as a parameter:

```python
# In ivr_client.py
COLAB_WEBSOCKET_URL = "ws://YOUR_NGROK_URL.ngrok.app/stream"

# Or via parameter
run_ivr_client(websocket_url="ws://your-url.ngrok.app/stream")
```

### Test Audio File

The client expects a 16kHz, 16-bit mono WAV file. Place your test file in the repository root:

```bash
# Example using sox to convert audio
sox input.mp3 -r 16000 -c 1 -b 16 test_hindi_complaint.wav
```

Or specify a custom path:

```python
run_ivr_client(audio_file="/path/to/your/test.wav")
```

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

## Execution Rules

The LLM orchestrator follows these rules when generating responses:

1. **EMPATHY & URGENCY**: High-urgency issues receive immediate acknowledgment of frustration
2. **LINGUISTIC MIRRORING**: Responses match the caller's regional dialect
3. **TTS OPTIMIZATION**: English acronyms written phonetically (e.g., "A T and T" not "AT&T")
4. **BREVITY**: Responses kept under 15 seconds of spoken time (2-3 short sentences)

## Dependencies

Required packages (add to `backend/requirements.txt`):

```txt
websockets>=12.0
```

Optional packages for full functionality:

```txt
faster-whisper  # For local STT testing
edge-tts        # For local TTS testing
```

## Integration with Sentinel-E

The IVR module integrates with the existing Sentinel-E architecture:

- **LLM Orchestration**: Can use `backend.metacognitive.orchestrator.MetaCognitiveOrchestrator`
- **Model Registry**: Leverages existing `COGNITIVE_MODEL_REGISTRY` for model selection
- **Database**: Can extend `backend.database.models` for call logging
- **Authentication**: Can use existing JWT auth from `backend.gateway.auth`

### TODO: Full Integration

```python
# backend/ivr/ivr_orchestrator.py
from backend.metacognitive.orchestrator import MetaCognitiveOrchestrator

async def process_with_llm(self, pipeline_data: Dict[str, Any], model_id: Optional[str] = None):
    mco = MetaCognitiveOrchestrator()
    # Build IVR-specific query
    query = self._build_ivr_query(pipeline_data)
    # Use MCO for response generation
    result = await mco.run(query, mode="standard", selected_model=model_id)
    return self._parse_llm_response(result)
```

## Testing

### Unit Tests

```bash
cd backend
python -m pytest ivr/tests/
```

### Manual Testing

1. Start a Colab server with Faster-Whisper WebSocket endpoint
2. Get the ngrok URL from Colab
3. Update `COLAB_WEBSOCKET_URL` in `ivr_client.py`
4. Run the client:

```bash
cd backend
python -m ivr.ivr_client
```

## Examples

### Example 1: Network Issue (Hindi - Mumbai)

**Input:**
```json
{
  "text": "Mera internet down hai yaar",
  "language": "hi",
  "detected_profile": {"lang": "hi_mumbai"},
  "intent_data": {"category": "Network_Issue", "urgency": "High"}
}
```

**Output:**
```json
{
  "tts_script": "Mafi chahenge sir. Hum aapke network issue ko abhi check kar rahe hain.",
  "suggested_action": "handoff_to_human",
  "tts_overrides": {"emotion": "apologetic"}
}
```

### Example 2: Billing Query (English - Bangalore)

**Input:**
```json
{
  "text": "I want to check my bill for this month",
  "language": "en",
  "detected_profile": {"lang": "en_bangalore"},
  "intent_data": {"category": "Billing_Query", "urgency": "Medium"}
}
```

**Output:**
```json
{
  "tts_script": "Sure, let me check your billing details right now.",
  "suggested_action": "resolve_automated",
  "tts_overrides": {"emotion": "helpful"}
}
```

## Roadmap

- [ ] Full integration with Sentinel-E Meta-Cognitive Orchestrator
- [ ] Real-time LLM API calls (Claude Opus 4.6)
- [ ] Database logging for call history and analytics
- [ ] FastAPI endpoints for IVR WebSocket server
- [ ] Asterisk PBX integration via SIP/VOIP
- [ ] Multi-language TTS support (Hindi, Tamil, Marathi, Telugu, Kannada)
- [ ] Call recording and audit logging
- [ ] Real-time sentiment analysis
- [ ] Escalation path configuration

## License

Part of the Sentinel-E project.
