# Voice Integration Architecture Diagram

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION                         │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
         ┌────────────────┐
         │  Click Mic Btn │
         │  in InputArea  │
         └────────┬───────┘
                  │
                  ▼
         ┌──────────────────────┐
         │  Start Recording     │┄┄┄┐
         │  (awaazService)      │   │ Display Timer
         └──────┬───────────────┘   │ & Status
                │                   │
        [Recording Audio...]         │ User speaks...
                │                   │
                ▼◄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┘
         ┌──────────────────────┐
         │  Stop Recording      │
         │  (awaazService)      │
         └──────┬───────────────┘
                │
                ▼
         ┌──────────────────────────────┐
         │   Upload Audio to AWAAZ      │
         │   POST /transcription/upload │
         └──────┬───────────────────────┘
                │
                ▼
         ┌──────────────────────────────┐
         │   Poll Transcription Result  │
         │   GET /transcription/status  │
         │   (Exponential Backoff)      │
         └──────┬───────────────────────┘
                │
      ┌─────────┴─────────┐
      │                   │
      ▼                   ▼
  Transcribed Text    Detected Language
      +                   +
      │                   │
      └─────────┬─────────┘
                │
                ▼
    ┌────────────────────────────────┐
    │   NLP Analysis (nlpRouter)     │
    │   ────────────────────────     │
    │  • Detect Intent               │
    │  • Classify Grievance          │
    │  • Recommend Mode/SubMode      │
    └────────┬───────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
Intent          Grievance
Category        Type
    +             +
    │             │
    └────────┬────┘
             │
             ▼
    ┌──────────────────────────┐
    │  Language Detection      │
    │  (languageDetector)      │
    │  ────────────────────    │
    │  • Map to Models         │
    │  • Select TTS Voice      │
    └────────┬─────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
Selected Model    TTS Voice
    +                 +
    │                 │
    └────────┬────────┘
             │
             ▼
    ┌──────────────────────────┐
    │   Update InputArea       │
    │   ────────────────────── │
    │  • Populate Text Text    │
    │  • Set Mode/SubMode      │
    │  • Show NLP Hints        │
    │  • Ready to Send         │
    └────────┬─────────────────┘
             │
             ▼
    ┌──────────────────────────┐
    │   User Reviews & Sends   │
    │   (Click Send Button)    │
    └────────┬─────────────────┘
             │
             ▼
    ┌──────────────────────────┐
    │   ChatEngine Processing  │
    │   (Standard Pipeline)    │
    │   ────────────────────── │
    │  • Mode/SubMode Selected │
    │  • Language Preserved    │
    │  • NLP Context Included  │
    └────────┬─────────────────┘
             │
             ▼
    ┌──────────────────────────┐
    │   AI Response            │
    │   (Language-Aware)       │
    └──────────────────────────┘
```

## Component Interaction Map

```
┌─────────────────────────────────────────────────────────────┐
│                   FRONTEND ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           components/InputArea.js                    │  │
│  │  ───────────────────────────────────────────────     │  │
│  │  • Microphone button                                 │  │
│  │  • Recording status display                          │  │
│  │  • Transcription integration                         │  │
│  │  • NLP hint display                                  │  │
│  └────────────┬──────────────────────────────────────┬─┘  │
│               │                                      │      │
│         calls │                                 hooks│      │
│               │                                      │      │
│        ┌──────▼──────────────────────────────────────┴──┐  │
│        │   hooks/useVoiceTranscription.js               │  │
│        │   ─────────────────────────────────────────    │  │
│        │   • Recording management                       │  │
│        │   • State management                           │  │
│        │   • Complete voice pipeline                    │  │
│        └──────┬────────────────────┬────────────────────┘  │
│               │                    │                        │
│         uses  │                 uses                        │
│               │                    │                        │
│    ┌──────────▼──┐        ┌────────▼──────────┐            │
│    │  SERVICES   │        │  ENGINES           │            │
│    ├─────────────┤        ├────────────────────┤            │
│    │ awaazService│        │ nlpRouter:         │            │
│    │             │        │ • analyzeAndRoute()│            │
│    │ • Record    │        │ • detectIntent()   │            │
│    │ • Upload    │        │ • classify()       │            │
│    │ • Poll      │        │                    │            │
│    │ • Cleanup   │        │ languageDetector:  │            │
│    │             │        │ • detectLanguage() │            │
│    │             │        │ • selectModel()    │            │
│    │             │        │ • fetchTTSVoice()  │            │
│    └─────┬───────┘        └────┬───────────────┘            │
│          │                     │                            │
│   sends  │                     │                            │
│   audio  │                 analyzes                         │
│          │                 text                             │
│          ▼                     │                            │
│         ┌────────────────────┐ │                            │
│         │  AWAAZ API Server  │◄┘                            │
│         │ ─────────────────  │                              │
│         │ • STT              │                              │
│         │ • Language Detect  │                              │
│         │ • NLP              │                              │
│         │ • TTS              │                              │
│         └─────────┬──────────┘                              │
│                   │                                         │
│           returns │ transcribed text                        │
│                   │ + language                              │
│                   │                                         │
│                   └────────────────────► InputArea updates  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow Sequence

```
Time → ─────────────────────────────────────────────────────────

User             InputArea        useVoiceTranscription    awaazService    AWAAZ API
  │                 │                      │                     │              │
  │──Click Mic──────│                      │                     │              │
  │                 │                      │                     │              │
  │                 │──startRecording()──→ │                     │              │
  │                 │                      │──initAudio()───────│              │
  │                 │                      │                   ◄─────────────── │
  │                 │                      │                     │ permission   │
  │                 │                      │──startRecording()──│              │
  │                 │◄──Show Recording──────│                     │              │
  │                 │  Timer               │                     │              │
  │                 │                      │                     │              │
  │  [SPEAKING...]  │                      │ [Recording...]      │              │
  │                 │                      │                     │              │
  │──Stop/Click Mic─│                      │                     │              │
  │                 │──stopRecording()────→│                     │              │
  │                 │◄──audio blob─────────│──stopRecording()────│              │
  │                 │                      │◄──audio blob────────│              │
  │                 │  ┏━━━━━━━━━━━━━━━━━━━│━━━━━━━━━━━━━━━━━━━━┓              │
  │                 │  ┃ UPLOADING & POLLING ┃                     │              │
  │                 │  ┣──uploadAudio()────→│──uploadAudio()────→ │──POST────→ │
  │                 │  │                     │                    │            │
  │                 │  │  ┏━ pollJob────────→│──GET────┐           │ [Queue]    │
  │                 │  │  ┃  with backoff     │         │──────────┤           │
  │                 │  │  ┃  (exponential)    │ [repeat]│           │ [Process] │
  │                 │  │  ┃ ◄──pending────────│◄────────┤           │           │
  │                 │  │  ▪ [wait 0.5s]      │         │           │           │
  │                 │  │  ▪ [repeat 0.75s]   │         │           │ [Transcrib│
  │                 │  │  ▪ [...exp backoff] │         │           │  ing...]  │
  │                 │  │   ◄──complete       │◄────────┘           │           │
  │                 │  ┗━━━━━━━━━━━━━━━━━━━━┛                     │           │
  │                 │                      │◄text + language──────│◄──JSON─── │
  │                 │                      │                     │              │
  │                 │  ┏━━━━━━━━━━━━━━━━━━━│━━━━━━━━━━━━━━━━━━┓              │
  │                 │  ┃ NLP ANALYSIS       ┃                    │              │
  │                 │  ├─analyzeAndRoute()──│ (nlpRouter)        │              │
  │                 │  │ └─selectLanguage───│ (languageDetector)│              │
  │                 │  ┗━━━━━━━━━━━━━━━━━━━┛                    │              │
  │                 │                      │◄nplAnalysis result───│              │
  │                 │                      │                     │              │
  │                 │◄─Update────────────────intent              │              │
  │                 │  grievance            model selection      │              │
  │                 │  language             confidence           │              │
  │                 │                                             │              │
  │◄──Text & Mode────│                      │                     │              │
  │  Populated       │                      │                     │              │
  │                 │  Show NLP Hint        │                     │              │
  │                 │  (Category/Mode)      │                     │              │
  │                 │                      │                     │              │
  │ [Reviews Text]  │                      │                     │              │
  │                 │                      │                     │              │
  │ [Clicks Send]   │                      │                     │              │
  │──onSend()───────│                      │                     │              │
  │                 │ (text + mode)        │                     │              │
  │                 │──ChatEngine Pipeline─→ [Process & Respond]  │              │
  │                 │                      │                     │              │
```

## Integration Points

```
ChatEngine
├── handleSend()
│   ├── Input: { text, file }
│   ├── Mode selected (set by nlpRouter)
│   ├── SubMode selected (set by nlpRouter)
│   └── Sends to /api/run/omega/{mode}
│
InputArea
├── Voice Button
│   ├── Calls useVoiceTranscription()
│   ├── Records audio
│   ├── Gets transcription
│   ├── Performs NLP analysis
│   ├── Auto-selects mode
│   └── Updates state
│
useVoiceTranscription Hook
├── startRecording()
├── stopRecording()
│   ├── Uploads to AWAAZ
│   ├── Polls transcription
│   ├── Calls analyzeAndRoute()
│   ├── Calls analyzeLanguageAndSelectModel()
│   └── Returns complete analysis
│
nlpRouter Engine
├── analyzeAndRoute(text, language)
│   ├── detectIntent() → Complaint/Question/Feedback/etc
│   ├── classifyGrievance() → GR-01 to GR-08
│   ├── determineMode() → standard/experimental
│   └── Returns routing decision
│
languageDetector Engine
├── detectLanguageFromText(text)
│   ├── Script pattern matching
│   └── Returns language code
│
├── analyzeLanguageAndSelectModel()
│   ├── Validate language support
│   ├── Select appropriate model
│   ├── Get TTS voice
│   └── Determine RTL status
│
awaazService
├── Audio Lifecycle
│   ├── initializeAudio()
│   ├── startRecording()
│   ├── stopRecording()
│   └── cleanup()
│
├── API Communication
│   ├── uploadAudio() → POST /transcription/upload
│   ├── getTranscriptionStatus() → GET /transcription/status/{id}
│   ├── pollJobWithBackoff() → Smart retry logic
│   └── Timeout handling → 60s default
```

## State Management Flow

```
InputArea Component State:
├── text (textarea content)
├── isRecording (boolean)
├── isTranscribing (boolean)
├── recordingTime (seconds)
├── nlpAnalysis (complete analysis result)
├── showNLPHint (display toggle)
└── file (attached file)

useVoiceTranscription Hook State:
├── isRecording (boolean)
├── isTranscribing (boolean)
├── recordingTime (seconds)
├── nlpAnalysis (object)
├── transcribedText (string)
└── error (string or null)

ChatEngine State:
├── mode ('standard' or 'experimental')
├── subMode ('debate'|'evidence'|'glass'|'synthesis'|null)
├── messages (array)
├── loading (boolean)
└── [other state...]

Props Flow:
InputArea ← onModeChange, onSubModeChange
         ← onSend callback

ChatEngine → mode, subMode to InputArea
         ← updates from voice system
         → setMode, setSubMode to update
```
