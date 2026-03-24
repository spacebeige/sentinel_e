# Voice Transcription & NLP Routing Integration Guide

## Overview

The Sentinel-E system now includes integrated voice transcription powered by AWAAZ with intelligent NLP routing and multilingual support. This guide explains how to use and integrate these features.

## Architecture

### Components

1. **awaazService** (`frontend/src/services/awaazService.js`)
   - Handles direct communication with the AWAAZ API backend
   - Manages audio recording, uploading, and transcription polling
   - Methods:
     - `initializeAudio()` - Request microphone access
     - `startRecording()` - Begin audio capture
     - `stopRecording()` - Finish recording and return audio blob
     - `uploadAudio(blob)` - Upload to AWAAZ API
     - `getTranscriptionStatus(jobId)` - Poll transcription result
     - `pollJobWithBackoff()` - Exponential backoff polling

2. **nlpRouter** (`frontend/src/engines/nlpRouter.js`)
   - Analyzes transcribed text for intent and grievance classification
   - Determines optimal model mode (standard/experimental)
   - Selects sub-mode (debate/evidence/glass/synthesis)
   - Categories: Water Supply, Sanitation, Road, Health, Electricity, Education, Documents, Other
   - Intents: Complaint, Question, Feedback, Suggestion, General

3. **languageDetector** (`frontend/src/engines/languageDetector.js`)
   - Detects language from Unicode script patterns
   - Supports 20+ Indian languages + English, Urdu
   - Maps languages to models and TTS voices
   - Handles RTL language detection

4. **useVoiceTranscription** Hook (`frontend/src/hooks/useVoiceTranscription.js`)
   - React hook encapsulating complete voice pipeline
   - Returns recording state, transcription, NLP analysis

5. **InputArea** (`frontend/src/components/InputArea.js`)
   - Enhanced text input with voice button
   - Displays recording timer and transcription status
   - Shows NLP classification hints

## Usage Examples

### Using the Voice Button in UI

```jsx
import InputArea from './components/InputArea';

// In your component:
<InputArea
  onSend={handleSend}
  loading={isLoading}
  mode={mode}
  subMode={subMode}
  onModeChange={setMode}
  onSubModeChange={setSubMode}
/>
```

### Using the useVoiceTranscription Hook Directly

```jsx
import { useVoiceTranscription } from './hooks/useVoiceTranscription';

export function MyVoiceComponent() {
  const {
    isRecording,
    isTranscribing,
    recordingTime,
    nlpAnalysis,
    transcribedText,
    error,
    startRecording,
    stopRecording,
    reset,
  } = useVoiceTranscription();

  const handleRecord = async () => {
    if (isRecording) {
      try {
        const result = await stopRecording();
        console.log('Transcribed:', result.text);
        console.log('NLP Analysis:', result.routing);
        console.log('Language:', result.languageSelection);
      } catch (err) {
        console.error('Transcription error:', err);
      }
    } else {
      await startRecording();
    }
  };

  const handleReset = () => {
    reset();
  };

  return (
    <div>
      <button onClick={handleRecord}>
        {isRecording ? 'Stop Recording' : 'Start Recording'}
      </button>
      {recordingTime > 0 && <p>Recording: {recordingTime}s</p>}
      {isTranscribing && <p>Transcribing...</p>}
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}
      {transcribedText && <p>Transcribed: {transcribedText}</p>}
      {nlpAnalysis && (
        <div>
          <p>Intent: {nlpAnalysis.routing.detected_intent}</p>
          <p>Grievance: {nlpAnalysis.routing.detected_grievance.category_name}</p>
          <p>Recommended Mode: {nlpAnalysis.routing.recommended_mode}</p>
        </div>
      )}
      <button onClick={handleReset}>Reset</button>
    </div>
  );
}
```

### Direct API Usage

```javascript
import awaazService from './services/awaazService';

// Manual transcription workflow
async function transcribeAudio(audioBlob) {
  try {
    // Upload audio
    const upload = await awaazService.uploadAudio(audioBlob);
    
    // Poll for result
    const result = await awaazService.pollJobWithBackoff(
      upload.job_id,
      'transcription',
      (id) => awaazService.getTranscriptionStatus(id)
    );
    
    return result.transcribed_text;
  } catch (error) {
    console.error('Transcription failed:', error);
  }
}
```

### Using NLP Router

```javascript
import { analyzeAndRoute, GRIEVANCE_CATEGORIES } from './engines/nlpRouter';

const query = "मेरी सड़क में बहुत सारे गड्ढे हैं"; // "My road has many potholes"
const language = "hi"; // Hindi

const routing = analyzeAndRoute(query, language);

console.log('Intent:', routing.detected_intent); // 'complaint'
console.log('Grievance:', routing.detected_grievance.category_name); // 'Road Infrastructure'
console.log('Recommended Mode:', routing.recommended_mode); // 'standard'
console.log('Confidence:', routing.confidence); // 0.85
```

### Using Language Detector

```javascript
import { 
  analyzeLanguageAndSelectModel,
  formatLanguageDetection,
  detectLanguageFromText 
} from './engines/languageDetector';

const text = "तमिल पाठ यह है";
const detection = detectLanguageFromText(text);

console.log(detection);
// {
//   language: 'ta',
//   confidence: 0.95,
//   detectedScripts: ['Tamil'],
//   isMultilingual: false
// }

// Get appropriate model
const selection = analyzeLanguageAndSelectModel(text);
console.log(selection.selectedModel); // 'gpt-4' or 'claude-3-opus'
console.log(selection.ttsVoice); // TTS configuration
```

## Supported Languages

| Code | Language | Script | Models |
|------|----------|--------|--------|
| en | English | Latin | GPT-4, Claude, Llama |
| hi | Hindi | Devanagari | GPT-4, Claude, Llama |
| ta | Tamil | Tamil | GPT-4, Claude |
| te | Telugu | Telugu | GPT-4, Claude |
| kn | Kannada | Kannada | GPT-4, Claude |
| ml | Malayalam | Malayalam | GPT-4, Claude |
| mr | Marathi | Devanagari | GPT-4, Claude |
| gu | Gujarati | Gujarati | GPT-4, Claude |
| bn | Bengali | Bengali | GPT-4, Claude |
| pa | Punjabi | Gurmukhi | GPT-4, Claude |
| or | Odia | Odia | GPT-4, Claude |
| ur | Urdu | Nastaliq | GPT-4, Claude |

## Grievance Categories

| Code | Category | Keywords |
|------|----------|----------|
| GR-01 | Water Supply | water, tap, jal, नल |
| GR-02 | Sanitation & Sewerage | sewerage, sanitation, waste |
| GR-03 | Road Infrastructure | road, pothole, street |
| GR-04 | Public Health | health, hospital, medicines |
| GR-05 | Electricity | electricity, power, light |
| GR-06 | Education | education, school, college |
| GR-07 | Documents & Permits | document, certificate, permit |
| GR-08 | Other | miscellaneous complaints |

## Mode Selection Logic

### Standard Mode
- **When**: Complaints, simple questions, documents-related issues
- **Use**: Direct grievance processing
- **Examples**: "पानी की समस्या है", "My electricity is not working"

### Experimental Mode - Debate
- **When**: Why/How questions requiring reasoning
- **Use**: Multi-sided analysis
- **Example**: "Why is public transportation important?"

### Experimental Mode - Evidence
- **When**: Feedback or suggestion requests
- **Use**: Source-backed analysis
- **Example**: "How can we improve water quality?"

### Experimental Mode - Glass
- **When**: Need to understand AI reasoning
- **Use**: Transparency-focused responses

### Experimental Mode - Synthesis
- **When**: Complex queries needing model collaboration
- **Use**: Multi-model consensus building

## Integration with ChatEngine

The system automatically routes to ChatEngine:

```javascript
// In InputArea or custom component
const result = await stopRecording();

// This automatically populates:
// 1. Text field with transcribed text
// 2. Sets appropriate mode/subMode based on NLP analysis
// 3. Stores language preference
// 4. User clicks send to process through ChatEngine
```

## Backend Requirements

Ensure the AWAAZ API server is running:

```bash
cd /Users/ashwinagarkhed/sentinel_e/awaaz
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

Or run the backend Sentinel-E server which includes AWAAZ integration:

```bash
cd /Users/ashwinagarkhed/sentinel_e/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## Environment Variables

```bash
# Frontend (.env)
REACT_APP_AWAAZ_API=http://localhost:8000/api/v1

# Backend (.env)
SARVAM_API_KEY=your_key_here
SARVAM_LANG_DETECT_API_URL=https://api.sarvam.ai/language-detect
SARVAM_TRANSLATE_API_URL=https://api.sarvam.ai/translate
SARVAM_TTS_API_URL=https://api.sarvam.ai/text-to-speech
```

## API Endpoints

### Transcription
- `POST /api/v1/transcription/upload` - Upload audio file
- `GET /api/v1/transcription/status/{jobId}` - Check transcription status

### NLP Processing
- `POST /api/v1/nlp/analyze` - Analyze text for grievance/intent

### AI Processing
- `POST /api/v1/ai/process` - Send to model with NLP context

### TTS Synthesis
- `POST /api/v1/tts/synthesize` - Generate speech from text

### Language Detection
- `POST /api/v1/language/detect` - Detect language from text

## Error Handling

```javascript
try {
  const result = await stopRecording();
  handleSuccess(result);
} catch (error) {
  if (error.message.includes('timed out')) {
    // Handle timeout
  } else if (error.message.includes('access denied')) {
    // Handle microphone permission issue
  } else {
    // Handle other errors
  }
}
```

## Performance Considerations

1. **Polling Timeout**: Default 60 seconds, adjustable in `pollJobWithBackoff()`
2. **Recording Limit**: Keep recordings under 60 seconds for optimal performance
3. **Language Detection**: Fastest on text with 10+ characters
4. **Model Selection**: Automatically adapts to language capabilities

## Testing

```javascript
// Test with mock audio
const testAudio = new Blob(['test'], { type: 'audio/webm' });
const upload = await awaazService.uploadAudio(testAudio);

// Test NLP routing
const queries = [
  { text: "पानी नहीं आ रहा है", language: "hi" },
  { text: "Road has potholes", language: "en" },
  { text: "எப்படி படிக்கலாம்?", language: "ta" },
];

queries.forEach(q => {
  const result = analyzeAndRoute(q.text, q.language);
  console.log(result);
});
```

## Troubleshooting

### Microphone Access Denied
- Check browser permissions
- Make sure HTTPS is used (or localhost)
- Restart browser and try again

### Transcription Timeout
- Check AWAAZ server is running
- Check network connectivity
- Increase timeout in `pollJobWithBackoff()` call

### NLP Misclassification
- More keywords can be added to `GRIEVANCE_CATEGORIES`
- Confidence score can be checked before routing
- Manual mode selection overrides NLP

### Language Detection Issues
- Ensure text contains actual language scripts
- Mixed language text may detect dominant language
- Can manually specify language if needed

## Future Enhancements

1. Real-time transcription feedback
2. Custom grievance categories per region
3. Multi-language response generation
4. Voice feedback (TTS responses)
5. Conversation context preservation
6. Custom model fine-tuning per grievance type
7. Analytics dashboard for grievance trends
