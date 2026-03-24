"""
AWAAZ — Voice Recognition + Response Loop
Single runnable file. Citizen speaks → detect lang/accent → STT → model → TTS → citizen hears.

INSTALL (run once):
    pip install faster-whisper silero-vad transformers torch groq gTTS soundfile numpy scipy pydub
    pip install TTS       # Coqui TTS (optional, better Hindi voice)
    pip install fasttext  # token-level language ID — replaces all heuristic lang detection

    # fastText model (lid.176.bin — 126MB, download once):
    #   wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -O /tmp/lid.176.bin
    # OR set env var: FASTTEXT_MODEL_PATH=/path/to/lid.176.bin

    # For Ollama local fallback: curl -fsSL https://ollama.com/install.sh | sh && ollama pull llama3.2:3b

    # IndicWav2Vec (accent correction — optional, improves regional accuracy):
    # Downloads automatically on first run from HuggingFace (~900MB)

USAGE:
    # With a mic (live call simulation):
    python awaaz_voice.py --mode mic

    # With a pre-recorded WAV file:
    python awaaz_voice.py --mode file --input path/to/audio.wav

    # With Asterisk AGI (production):
    python awaaz_voice.py --mode asterisk

ENV VARS (optional — only needed if using Groq):
    export GROQ_API_KEY=your_free_key_here
"""

import os
import sys
import uuid
import time
import json
import wave
import queue
import struct
import asyncio
import argparse
import tempfile
import threading
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

SAMPLE_RATE       = 16000       # Hz — Whisper expects 16kHz
CHUNK_SIZE        = 512         # samples per VAD chunk
SILENCE_THRESHOLD = 0.7         # seconds of silence = end of utterance
MAX_UTTERANCE_S   = 30          # max seconds before force-flush
GROQ_MODEL        = "llama-3.1-8b-instant"
OLLAMA_MODEL      = "llama3.2:3b"
OLLAMA_URL        = "http://localhost:11434/api/generate"
WHISPER_MODEL     = "small"     # tiny / base / small / medium
MAX_TURNS         = 10          # end call after N turns
REDIS_TTL         = 7200        # 2 hours
# fastText LID model path — override with env var FASTTEXT_MODEL_PATH
FASTTEXT_MODEL_PATH = os.environ.get("FASTTEXT_MODEL_PATH", "/tmp/lid.176.bin")
# Minimum fraction of words in a non-primary language to call input "mixed"
MIXED_LANG_THRESHOLD = 0.20

# ─────────────────────────────────────────────
# LANG META RESOLVER — replaces LANGUAGES dict
# ─────────────────────────────────────────────

class LangMetaResolver:
    """
    Resolves language metadata (name, script, gTTS code, pace) dynamically
    from the LLM instead of a hardcoded table.

    Why: any language Whisper or fastText detects — including Tulu,
    Chhattisgarhi, Saurashtra, Konkani, Bhojpuri, Bishnupriya Manipuri,
    Sadri, Maithili — is handled automatically. Adding a new language
    requires zero code changes.

    Responses are cached in-process (dict keyed by lang code) so the
    LLM is only called once per language per process lifetime.
    """

    def __init__(self):
        self._cache: dict = {}
        # gTTS only supports ~30 codes — for unsupported ones we use
        # the nearest supported language (determined by LLM, not hardcoded)
        self._gtts_supported = set([
            "af","ar","bg","bn","bs","ca","cs","cy","da","de","el","en",
            "eo","es","et","fi","fr","gu","hi","hr","hu","hy","id","is",
            "it","ja","jw","km","kn","ko","la","lv","mk","ml","mr","my",
            "ne","nl","no","pl","pt","ro","ru","si","sk","sq","sr","su",
            "sv","sw","ta","te","th","tl","tr","uk","ur","vi","zh-cn","zh-tw",
        ])

    def resolve(self, lang_code: str, llm_client=None) -> dict:
        """
        Returns dict with keys: name, script, gtts_lang, pace_rate, pause_ms.
        Uses cached value if available. Falls back to safe defaults if LLM
        unavailable.
        """
        if lang_code in self._cache:
            return self._cache[lang_code]

        meta = self._from_llm(lang_code, llm_client) if llm_client else {}
        if not meta:
            meta = self._safe_default(lang_code)

        self._cache[lang_code] = meta
        return meta

    def _from_llm(self, lang_code: str, llm_client) -> dict:
        """
        Ask the LLM for language metadata. Returns empty dict on any failure.
        The LLM knows every ISO 639 code — no table needed.
        """
        prompt = (
            f"Given the ISO 639 language code '{lang_code}', return a JSON object "
            f"with exactly these keys:\n"
            f"  name: full language name in English\n"
            f"  script: writing system name (e.g. Devanagari, Tamil, Latin)\n"
            f"  gtts_lang: the closest gTTS-supported language code from this list: "
            f"{sorted(self._gtts_supported)}\n"
            f"  pace_rate: speaking pace multiplier 0.8-1.1 (rural dialects slower, "
            f"urban faster, default 1.0)\n"
            f"  pause_ms: inter-sentence pause in ms (200-500, rural longer)\n"
            f"Return ONLY the JSON object, no explanation, no markdown."
        )
        try:
            resp = llm_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            raw = raw.replace("```json","").replace("```","").strip()
            data = json.loads(raw)
            # Validate keys present
            for k in ("name","script","gtts_lang","pace_rate","pause_ms"):
                if k not in data:
                    return {}
            print(f"[LANGMETA] {lang_code} → {data}")
            return data
        except Exception as e:
            print(f"[LANGMETA] LLM query failed for {lang_code}: {e}")
            return {}

    def _safe_default(self, lang_code: str) -> dict:
        """
        Pure-signal fallback — no hardcoded per-language values.
        Uses only the lang_code itself to make safe structural guesses.
        """
        DEVANAGARI_CODES = {"hi","mr","ne","mai","doi","kok","sa","bho","brx","awa","mag"}
        DRAVIDIAN_GTTS   = {"ta":"ta","te":"te","kn":"kn","ml":"ml","tcy":"kn","kod":"kn"}
        EASTERN_INDIC    = {"bn":"bn","as":"bn","or":"or","mni":"bn"}
        ARABIC_SCRIPT    = {"ur":"ur","sd":"ur","ks":"ur"}
        LATIN_FAMILY     = {"en":"en"}
        OTHER_MAP        = {"gu":"gu","pa":"pa"}

        base = lang_code.split("-")[0].lower()
        if base in DEVANAGARI_CODES:
            script, gtts = "Devanagari", "hi"
        elif base in DRAVIDIAN_GTTS:
            script, gtts = "Dravidian", DRAVIDIAN_GTTS[base]
        elif base in EASTERN_INDIC:
            script, gtts = "Eastern Indic", EASTERN_INDIC[base]
        elif base in ARABIC_SCRIPT:
            script, gtts = "Nastaliq/Arabic", ARABIC_SCRIPT[base]
        elif base in LATIN_FAMILY:
            script, gtts = "Latin", "en"
        elif base in OTHER_MAP:
            script, gtts = "Indic", OTHER_MAP[base]
        else:
            script, gtts = "Unknown", "hi"

        return {
            "name":       lang_code,
            "script":     script,
            "gtts_lang":  gtts,
            "pace_rate":  1.0,
            "pause_ms":   300,
        }

    def get_pace(self, lang_code: str, accent_region: str,
                  llm_client=None) -> dict:
        """
        Returns pace config. For known accent regions (detected by IndicWav2Vec),
        asks the LLM what pace adjustment makes sense.
        """
        cache_key = f"{lang_code}::{accent_region}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        meta = self.resolve(lang_code, llm_client)
        base_rate  = meta.get("pace_rate", 1.0)
        base_pause = meta.get("pause_ms", 300)

        # If we have a specific accent region, ask LLM for a refinement
        if accent_region and accent_region != "default" and llm_client:
            try:
                prompt = (
                    f"For a {lang_code} speaker from region '{accent_region}', "
                    f"return a JSON with: pace_rate (0.8-1.1) and pause_ms (150-500). "
                    f"Rural speakers slower, urban faster. "
                    f"Return ONLY JSON, no markdown."
                )
                resp = llm_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=60,
                    temperature=0.0,
                )
                raw = resp.choices[0].message.content.strip()
                raw = raw.replace("```json","").replace("```","").strip()
                adjustment = json.loads(raw)
                base_rate  = adjustment.get("pace_rate", base_rate)
                base_pause = adjustment.get("pause_ms", base_pause)
            except Exception:
                pass

        result = {"rate": float(base_rate), "pause_ms": int(base_pause)}
        self._cache[cache_key] = result
        return result


# Module-level singleton
_lang_meta = LangMetaResolver()


def get_lang_meta(lang_code: str, llm_client=None) -> dict:
    """Public accessor for the LangMetaResolver singleton."""
    return _lang_meta.resolve(lang_code, llm_client)


def get_pace(lang_code: str, accent_region: str, llm_client=None) -> dict:
    """Public accessor for pace config."""
    return _lang_meta.get_pace(lang_code, accent_region, llm_client)


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class CallerProfile:
    session_id:       str   = field(default_factory=lambda: str(uuid.uuid4()))
    lang:             str   = "hi"
    lang_name:        str   = "Hindi"
    accent_region:    str   = "default"
    formality_score:  float = 0.5
    formality_label:  str   = "STANDARD"
    script:           str   = "Devanagari"
    gtts_lang:        str   = "hi"
    detected_at_turn: int   = 1
    confidence:       float = 0.0
    turn_number:      int   = 0
    state:            str   = "GREETING"
    history:          list  = field(default_factory=list)
    is_emergency:     bool  = False
    lang_mode:        str   = "pure"
    lang_distribution: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# MODULE 1 — VAD (Voice Activity Detection)
# ─────────────────────────────────────────────

class VADProcessor:
    """
    Silero VAD — strips silence, finds utterance boundaries.
    Loads a 1MB model, runs fully on CPU.
    """
    def __init__(self):
        self.model = None
        self.is_loaded = False

    def load(self):
        try:
            import torch
            self.model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False
            )
            self.get_speech_ts = utils[0]
            self.is_loaded = True
            print("[VAD] Silero VAD loaded OK")
        except Exception as e:
            print(f"[VAD] WARNING: Silero VAD failed to load ({e}). Using energy-based fallback.")
            self.is_loaded = False

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Returns True if the chunk contains speech."""
        if self.is_loaded:
            try:
                import torch
                tensor = torch.FloatTensor(audio_chunk)
                confidence = self.model(tensor, SAMPLE_RATE).item()
                return confidence > 0.5
            except Exception:
                pass
        # Energy-based fallback
        rms = np.sqrt(np.mean(audio_chunk.astype(float) ** 2))
        return rms > 300

    def find_utterance_boundaries(self, audio: np.ndarray) -> list:
        """Find speech segments in a longer audio array."""
        if self.is_loaded:
            try:
                import torch
                tensor = torch.FloatTensor(audio)
                speeches = self.get_speech_ts(tensor, self.model, sampling_rate=SAMPLE_RATE)
                return [(s["start"], s["end"]) for s in speeches]
            except Exception:
                pass
        return [(0, len(audio))]


# ─────────────────────────────────────────────
# MODULE 2 — STT (Speech to Text)
# ─────────────────────────────────────────────

class STTProcessor:
    """
    faster-whisper — transcribes audio in 14 Indian languages.
    Loads once, never reloaded. asyncio 5-slot pool.
    """
    def __init__(self):
        self.model = None
        self._semaphore = asyncio.Semaphore(5)

    def load(self):
        try:
            from faster_whisper import WhisperModel
            device = "cpu"
            compute_type = "int8"
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                    compute_type = "float16"
                    print("[STT] GPU detected — using CUDA")
            except Exception:
                pass
            self.model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
            print(f"[STT] faster-whisper ({WHISPER_MODEL}) loaded on {device}")
        except ImportError:
            print("[STT] ERROR: faster-whisper not installed. Run: pip install faster-whisper")
            sys.exit(1)

    def detect_language(self, audio_path: str) -> tuple:
        """
        Detect language from first 30 seconds of audio.
        Returns (lang_code, confidence).
        """
        try:
            segments, info = self.model.transcribe(
                audio_path,
                task="transcribe",
                language=None,
                beam_size=1,
                best_of=1,
                vad_filter=True,
            )
            _ = list(segments)
            lang = info.language
            conf = info.language_probability
            if not lang:
                lang = "hi"
            print(f"[STT] Language detected: {lang} (confidence: {conf:.2f})")
            return lang, conf
        except Exception as e:
            print(f"[STT] Language detection failed: {e}")
            return "hi", 0.5

    def transcribe(self, audio_path: str, lang: str) -> str:
        """
        Transcribe audio file in the given language.
        Returns plain text string.
        """
        try:
            whisper_lang = lang if lang != "hi-en" else "hi"

            segments, info = self.model.transcribe(
                audio_path,
                language=whisper_lang,
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            text = " ".join(seg.text.strip() for seg in segments)
            print(f"[STT] Transcribed ({lang}): {text}")
            return text.strip()
        except Exception as e:
            print(f"[STT] Transcription failed: {e}")
            return ""


# ─────────────────────────────────────────────
# MODULE 2b — TOKEN-LEVEL LANGUAGE DETECTION
# ─────────────────────────────────────────────

class TokenLevelLangDetector:
    """
    fastText lid.176.bin — word-level language identification.
    Singleton pattern — model loads once per process.
    """

    _instance = None

    @classmethod
    def get(cls) -> "TokenLevelLangDetector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._model = None
        self._available = False
        self._load()

    def _load(self):
        """Load lid.176.bin once."""
        if not os.path.exists(FASTTEXT_MODEL_PATH):
            print(f"[LANGDET] fastText model not found at {FASTTEXT_MODEL_PATH}.")
            print("[LANGDET] Download: wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -O /tmp/lid.176.bin")
            print("[LANGDET] Falling back to Whisper sentence-level detection only.")
            return
        try:
            import fasttext
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = fasttext.load_model(FASTTEXT_MODEL_PATH)
            self._available = True
            print("[LANGDET] fastText LID model loaded OK")
        except ImportError:
            print("[LANGDET] fasttext not installed. Run: pip install fasttext")
        except Exception as e:
            print(f"[LANGDET] fastText load failed: {e}")

    def _predict_word(self, word: str) -> str:
        """Predict language for a single word."""
        if not self._available or not word.strip():
            return "xx"
        try:
            labels, _ = self._model.predict(word.strip(), k=1)
            return labels[0].replace("__label__", "")
        except Exception:
            return "xx"

    def detect(self, text: str, sentence_lang: str) -> tuple:
        """
        Run token-level detection on transcribed text.
        Returns (lang_mode, lang_distribution).
        """
        if not self._available:
            return "pure", {sentence_lang: 1.0}

        words = [w for w in text.split() if len(w) > 1]
        if not words:
            return "pure", {sentence_lang: 1.0}

        counts: dict = {}
        for w in words:
            lang_code = self._predict_word(w)
            if lang_code != "xx":
                counts[lang_code] = counts.get(lang_code, 0) + 1

        total = sum(counts.values()) or 1
        distribution = {k: round(v / total, 3) for k, v in sorted(
            counts.items(), key=lambda x: -x[1]
        )}

        primary_frac = distribution.get(sentence_lang, 0.0)
        non_primary  = 1.0 - primary_frac
        lang_mode    = "mixed" if non_primary >= MIXED_LANG_THRESHOLD else "pure"

        print(f"[LANGDET] mode={lang_mode} distribution={distribution}")
        return lang_mode, distribution

    def update_profile(self, text: str, profile: CallerProfile) -> None:
        """Run detection and mutate profile in-place."""
        mode, dist = self.detect(text, profile.lang)
        profile.lang_mode         = mode
        profile.lang_distribution = dist


# ─────────────────────────────────────────────
# MODULE 3 — ACCENT + FORMALITY DETECTION
# ─────────────────────────────────────────────

class ProfileDetector:
    """
    Detects accent region and formality score.
    Uses IndicWav2Vec (optional) + structural features.
    """

    def detect_accent(self, audio_path: str, lang: str) -> str:
        """
        Attempt IndicWav2Vec accent detection (Hindi only).
        Falls back to language-based default.
        """
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            import torch
            import soundfile as sf

            if not lang.startswith("hi"):
                return f"{lang}-default"

            audio, sr = sf.read(audio_path)
            if sr != SAMPLE_RATE:
                from scipy.signal import resample
                audio = resample(audio, int(len(audio) * SAMPLE_RATE / sr))

            processor = Wav2Vec2Processor.from_pretrained("ai4bharat/indicwav2vec-v2-all")
            model = Wav2Vec2Model.from_pretrained("ai4bharat/indicwav2vec-v2-all")

            inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            with torch.no_grad():
                embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze()

            norm = embedding.norm().item()
            if norm > 15:
                return "hi-UP-urban"
            else:
                return "hi-UP-rural"

        except Exception:
            defaults = {
                "hi": "hi-UP-rural", "ta": "ta-TN-urban",
                "mr": "mr-PUNE",     "bn": "bn-WB",
                "te": "default",     "en": "default",
            }
            return defaults.get(lang, "default")

    def detect_formality(self, text: str, lang: str) -> tuple:
        """
        Score formality 0.0–1.0 using structural features only.
        No per-language vocabulary lists.
        """
        if not text:
            return 0.3, "SIMPLE"

        words = text.split()
        total = len(words)
        if total == 0:
            return 0.3, "SIMPLE"

        import re
        sentences = [s.strip() for s in re.split(r'[।.!?]', text) if s.strip()]
        n_sent    = len(sentences) or 1
        avg_len   = sum(len(s.split()) for s in sentences) / n_sent

        punct_count  = sum(1 for c in text if c in "।.!?,;:")
        punct_score  = min(punct_count / max(total, 1) * 3, 1.0)

        length_score = min(avg_len / 15, 1.0)

        count_score  = min(total / 20, 1.0)

        score = round(length_score * 0.5 + punct_score * 0.3 + count_score * 0.2, 2)

        if score < 0.35:
            label = "SIMPLE"
        elif score < 0.65:
            label = "STANDARD"
        else:
            label = "FORMAL"

        print(f"[PROFILE] Formality: {score:.2f} → {label}")
        return score, label


# ─────────────────────────────────────────────
# MODULE 4 — MODEL (LLM reply generation)
# ─────────────────────────────────────────────

class ModelProcessor:
    """
    Generates reply using Groq (fast) or Ollama (local).
    Falls back automatically.
    """

    def __init__(self):
        self.groq_client = None
        self._load_groq()

    def _load_groq(self):
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            print("[MODEL] No GROQ_API_KEY set. Will use Ollama local fallback.")
            return
        try:
            from groq import Groq
            self.groq_client = Groq(api_key=api_key)
            print("[MODEL] Groq client loaded OK")
        except ImportError:
            print("[MODEL] groq package not installed. Run: pip install groq")

    def _build_system_prompt(self, profile: CallerProfile) -> str:
        max_sentences = {"SIMPLE": 2, "STANDARD": 3, "FORMAL": 4}.get(
            profile.formality_label, 2
        )
        history_text = "\n".join(
            f"Turn {i+1} - Citizen: {t['citizen']}\nAssistant: {t['assistant']}"
            for i, t in enumerate(profile.history[-3:])
        ) or "No previous turns."

        dist_str = ", ".join(
            f"{k}: {int(v*100)}%"
            for k, v in profile.lang_distribution.items()
        ) if profile.lang_distribution else f"{profile.lang}: 100%"

        if profile.lang_mode == "mixed":
            lang_style_rule = (
                f"The caller is speaking in a MIXED language style ({dist_str}). "
                f"Mirror their exact style, including any slang. "
                f"For {profile.lang_name} words, MUST use native script. "
                f"For English words, use Latin script."
            )
        else:
            lang_style_rule = (
                f"The caller is speaking pure {profile.lang_name}. "
                f"Reply ONLY in {profile.lang_name}. MUST use native {profile.script} script. "
                f"Avoid English words unless the caller used them. Mirror any slang used."
            )

        return f"""You are AWAAZ, a government grievance assistant on a live phone call in India.

CALLER PROFILE:
- Language: {profile.lang} ({profile.lang_name})
- Language mode: {profile.lang_mode} ({dist_str})
- Accent region: {profile.accent_region}
- Formality: {profile.formality_label} (score: {profile.formality_score})
- Script: {profile.script}
- Turn: {profile.turn_number}
- State: {profile.state}

CONVERSATION SO FAR:
{history_text}

LANGUAGE STYLE RULE:
{lang_style_rule}

CRITICAL TTS INSTRUCTIONS - DO NOT IGNORE:
1. When generating words in {profile.lang_name}, you MUST write them in their ACTUAL NATIVE SCRIPT (e.g. {profile.script}), NOT in English/Latin letters.
2. DO NOT use Romanized {profile.lang_name} (e.g. DO NOT write "Aap kaise hain"). 
3. The Text-To-Speech engine will mispronounce Romanized inputs.

STRICT RULES:
1. {lang_style_rule}
2. Use {profile.formality_label} vocabulary register
3. Maximum {max_sentences} sentences
4. ONE question per reply maximum — never ask two questions
5. NO markdown, NO bullet points, NO lists
6. If state is GREETING: greet and ask what problem to report
7. If state is GATHERING: ask ONE clarifying question about the complaint
8. If state is CONFIRMING: summarise complaint in 1 sentence and ask for confirmation
9. If state is EMERGENCY: acknowledge, say officer is being notified, give 2-hour callback promise
10. Numbers must be read digit by digit in the caller's language
11. Output ONLY the spoken reply — nothing else, no labels, no prefixes"""

    def generate(self, user_text: str, profile: CallerProfile) -> str:
        """Generate reply. Tries Groq first, falls back to Ollama, then rule-based."""
        system_prompt = self._build_system_prompt(profile)

        # Try Groq
        if self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_text},
                    ],
                    max_tokens=150,
                    temperature=0.4,
                )
                reply = response.choices[0].message.content.strip()
                print(f"[MODEL] Groq reply: {reply}")
                return reply
            except Exception as e:
                print(f"[MODEL] Groq failed ({e}), trying Ollama...")

        # Try Ollama local
        try:
            import urllib.request
            payload = json.dumps({
                "model": OLLAMA_MODEL,
                "prompt": f"SYSTEM: {system_prompt}\n\nUSER: {user_text}\nASSISTANT:",
                "stream": False,
                "options": {"num_predict": 150, "temperature": 0.4},
            }).encode()
            req = urllib.request.Request(
                OLLAMA_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                reply = data.get("response", "").strip()
                print(f"[MODEL] Ollama reply: {reply}")
                return reply
        except Exception as e:
            print(f"[MODEL] Ollama failed ({e}), using rule-based fallback...")

        return self._rule_based_reply(profile)

    def _rule_based_reply(self, profile: CallerProfile) -> str:
        """Last-resort fallback when both Groq and Ollama are unreachable."""
        state = profile.state

        state_messages = {
            "GREETING":   f"Hello. I am the government complaint assistant. Please tell me your problem.",
            "GATHERING":  f"Please describe your problem in more detail.",
            "CONFIRMING": f"Shall I register this complaint now? Please say yes or no.",
            "FILING":     f"Your complaint is being registered. Please wait.",
            "EMERGENCY":  f"I understand this is urgent. An officer is being notified immediately.",
            "CLOSING":    f"Thank you. Your complaint has been registered.",
        }
        msg = state_messages.get(state, state_messages["GREETING"])
        print(f"[MODEL] Rule-based fallback (all LLMs down): {msg}")
        return msg

    def normalize_for_tts_llm(self, text: str, profile: CallerProfile) -> str:
        """
        Convert LLM reply into clean, fully speakable form before TTS.
        LLM-based instead of regex/dictionary.
        """
        should_normalise = (
            profile.lang_mode == "mixed" and profile.formality_label == "SIMPLE"
        ) or self._has_foreign_script(text, profile.lang)

        if not should_normalise:
            return text

        lang_name = profile.lang_name
        norm_prompt = (
            f"You are a speech preparation assistant. "
            f"Convert the following text into clean, natural spoken {lang_name}. "
            f"Rules: "
            f"(1) Output must be fully speakable — no unpronounceable tokens. "
            f"(2) Preserve the meaning exactly. "
            f"(3) Keep it conversational, short. "
            f"(4) Do NOT add explanations or labels. "
            f"(5) Output ONLY the cleaned spoken text.\n\n"
            f"Input: {text}"
        )

        if self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": norm_prompt}],
                    max_tokens=120,
                    temperature=0.2,
                )
                normalised = response.choices[0].message.content.strip()
                if normalised:
                    print(f"[NORM] TTS normalised: {normalised}")
                    return normalised
            except Exception as e:
                print(f"[NORM] Groq normalisation failed ({e}), using original text")

        try:
            import urllib.request
            payload = json.dumps({
                "model": OLLAMA_MODEL,
                "prompt": norm_prompt,
                "stream": False,
                "options": {"num_predict": 120, "temperature": 0.2},
            }).encode()
            req = urllib.request.Request(
                OLLAMA_URL, data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
                normalised = data.get("response", "").strip()
                if normalised:
                    print(f"[NORM] Ollama normalised: {normalised}")
                    return normalised
        except Exception:
            pass

        return text

    def _has_foreign_script(self, text: str, lang: str) -> bool:
        """Detect if text contains characters from a different script."""
        SCRIPT_RANGES = {
            "hi": (0x0900, 0x097F),   # Devanagari
            "mr": (0x0900, 0x097F),   # Devanagari
            "ta": (0x0B80, 0x0BFF),   # Tamil
            "te": (0x0C00, 0x0C7F),   # Telugu
            "kn": (0x0C80, 0x0CFF),   # Kannada
            "ml": (0x0D00, 0x0D7F),   # Malayalam
            "bn": (0x0980, 0x09FF),   # Bengali
            "as": (0x0980, 0x09FF),   # Bengali
            "gu": (0x0A80, 0x0AFF),   # Gujarati
            "pa": (0x0A00, 0x0A7F),   # Gurmukhi
            "or": (0x0B00, 0x0B7F),   # Odia
            "ur": (0x0600, 0x06FF),   # Arabic
        }
        primary_range = SCRIPT_RANGES.get(lang)
        if not primary_range:
            return False

        for ch in text:
            cp = ord(ch)
            if 0x0041 <= cp <= 0x007A and lang not in ("en", "hi-en"):
                return True
            if (
                cp > 0x007F
                and not (primary_range[0] <= cp <= primary_range[1])
                and not (0x0900 <= cp <= 0x097F and lang in ("hi", "mr"))
                and ch not in " \t\n।॥,।.!?:;()-\""
            ):
                return True
        return False


# ─────────────────────────────────────────────
# MODULE 5 — TTS (Text to Speech)
# ─────────────────────────────────────────────

class TTSProcessor:
    """Text to speech in 14 Indian languages."""

    def __init__(self):
        self.coqui_available = False
        self._try_load_coqui()

    def _try_load_coqui(self):
        try:
            from TTS.api import TTS
            self.TTS = TTS
            self.coqui_available = True
            print("[TTS] Coqui TTS available")
        except ImportError:
            print("[TTS] Coqui TTS not installed — using gTTS for all languages")

    def synthesize(self, text: str, profile: CallerProfile, output_path: str) -> bool:
        """Convert text to speech file. Returns True on success."""
        lang      = profile.lang
        gtts_lang = get_lang_meta(lang).get("gtts_lang", "hi")
        pace_cfg  = get_pace(lang, profile.accent_region)

        if self.coqui_available and lang in ("hi", "en"):
            try:
                return self._coqui_synth(text, lang, output_path, pace_cfg)
            except Exception as e:
                print(f"[TTS] Coqui failed ({e}), falling back to gTTS")

        return self._gtts_synth(text, gtts_lang, output_path, pace_cfg)

    def _coqui_synth(self, text: str, lang: str, output_path: str, pace_cfg: dict) -> bool:
        tts_model = {
            "hi": "tts_models/hi/custom/vits",
            "en": "tts_models/en/ljspeech/tacotron2-DDC",
        }.get(lang, "tts_models/en/ljspeech/tacotron2-DDC")

        tts = self.TTS(model_name=tts_model, progress_bar=False)
        tts.tts_to_file(text=text, file_path=output_path)

        self._apply_pace(output_path, pace_cfg["rate"])
        print(f"[TTS] Coqui synthesized → {output_path}")
        return True

    def _gtts_synth(self, text: str, gtts_lang: str, output_path: str, pace_cfg: dict) -> bool:
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang=gtts_lang, slow=(pace_cfg["rate"] < 0.9))
            mp3_path = output_path.replace(".wav", ".mp3")
            tts.save(mp3_path)
            self._mp3_to_wav(mp3_path, output_path)
            print(f"[TTS] gTTS synthesized ({gtts_lang}) → {output_path}")
            return True
        except ImportError:
            print("[TTS] ERROR: gTTS not installed. Run: pip install gTTS")
            return False
        except Exception as e:
            print(f"[TTS] gTTS failed: {e}")
            return False

    def _apply_pace(self, wav_path: str, rate: float):
        """Adjust playback speed."""
        if rate == 1.0:
            return
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_wav(wav_path)
            new_frame_rate = int(audio.frame_rate * rate)
            adjusted = audio._spawn(
                audio.raw_data,
                overrides={"frame_rate": new_frame_rate}
            ).set_frame_rate(SAMPLE_RATE)
            adjusted.export(wav_path, format="wav")
        except Exception:
            pass

    def _mp3_to_wav(self, mp3_path: str, wav_path: str):
        try:
            from pydub import AudioSegment
            AudioSegment.from_mp3(mp3_path).export(wav_path, format="wav")
            os.remove(mp3_path)
        except Exception:
            import shutil
            shutil.move(mp3_path, wav_path)

    def play(self, wav_path: str):
        """Play audio to speaker."""
        try:
            import subprocess
            for cmd in [
                ["aplay", wav_path],
                ["ffplay", "-nodisp", "-autoexit", wav_path],
                ["mpg123", wav_path],
            ]:
                try:
                    subprocess.run(cmd, check=True,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                    return
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            print(f"[TTS] Audio saved to {wav_path} — no player found")
        except Exception as e:
            print(f"[TTS] Playback error: {e}")


# ─────────────────────────────────────────────
# MODULE 5b — TRANSLITERATION HOOK (PLUGGABLE)
# ─────────────────────────────────────────────

class TransliterationHook:
    """Pluggable transliteration layer (passthrough by default)."""

    def __init__(self):
        self._backend = None
        self._src_lang = None
        self._tgt_lang = None

    def enable_indictrans2(self, src: str, tgt: str):
        """Plug in IndicTrans2 as the backend."""
        try:
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import transliterate
            self._backend     = transliterate
            self._src_lang    = src
            self._tgt_lang    = tgt
            print(f"[TRANSLIT] IndicTrans2 enabled: {src} → {tgt}")
        except ImportError:
            print("[TRANSLIT] indic-transliteration not installed.")

    def process(self, text: str, profile: CallerProfile) -> str:
        """Apply transliteration if configured."""
        if self._backend is None:
            return text

        has_latin = any(0x0041 <= ord(c) <= 0x007A for c in text)
        if not has_latin or profile.lang in ("en", "hi-en"):
            return text

        return self._transliterate(text)

    def _transliterate(self, text: str) -> str:
        """Internal dispatch."""
        if self._backend is None:
            return text
        try:
            from indic_transliteration.sanscript import ITRANS, DEVANAGARI
            result = self._backend(text, ITRANS, DEVANAGARI)
            print(f"[TRANSLIT] {text} → {result}")
            return result
        except Exception as e:
            print(f"[TRANSLIT] Failed ({e}), returning original")
            return text


transliteration_hook = TransliterationHook()


# ─────────────────────────────────────────────
# MODULE 6 — AUDIO INPUT
# ─────────────────────────────────────────────

class AudioInput:
    """Handles mic recording, file reading, and Asterisk AGI input."""

    def __init__(self, mode: str, input_file: Optional[str] = None):
        self.mode       = mode
        self.input_file = input_file
        self._file_pos  = 0

    def record_utterance(self, session_id: str) -> Optional[str]:
        """Record one utterance. Returns path to WAV file."""
        if self.mode == "file":
            return self._next_file_chunk()
        elif self.mode == "mic":
            return self._record_from_mic(session_id)
        elif self.mode == "asterisk":
            return self._read_asterisk_pipe(session_id)
        return None

    def _record_from_mic(self, session_id: str) -> Optional[str]:
        """Record from microphone until silence."""
        try:
            import pyaudio
        except ImportError:
            print("[AUDIO] pyaudio not installed. Run: pip install pyaudio")
            return None

        pa       = pyaudio.PyAudio()
        stream   = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        print("[AUDIO] Listening... (speak now)")
        frames          = []
        silent_chunks   = 0
        speech_started  = False
        silence_limit   = int(SILENCE_THRESHOLD * SAMPLE_RATE / CHUNK_SIZE)
        max_chunks      = int(MAX_UTTERANCE_S * SAMPLE_RATE / CHUNK_SIZE)

        vad = VADProcessor()
        vad.load()

        for _ in range(max_chunks):
            data  = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            if vad.is_speech(audio):
                speech_started = True
                silent_chunks  = 0
                frames.append(data)
            else:
                if speech_started:
                    silent_chunks += 1
                    frames.append(data)
                    if silent_chunks >= silence_limit:
                        break

        stream.stop_stream()
        stream.close()
        pa.terminate()

        if not frames or not speech_started:
            print("[AUDIO] No speech detected")
            return None

        out_path = f"/tmp/awaaz_{session_id}_{int(time.time())}.wav"
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))

        print(f"[AUDIO] Recorded utterance → {out_path}")
        return out_path

    def _next_file_chunk(self) -> Optional[str]:
        """Return the input file path."""
        if self.input_file and os.path.exists(self.input_file):
            return self.input_file
        print(f"[AUDIO] File not found: {self.input_file}")
        return None

    def _read_asterisk_pipe(self, session_id: str) -> Optional[str]:
        """Read audio from Asterisk AGI stdin pipe."""
        pipe_path = f"/tmp/asterisk_audio_{session_id}.raw"
        out_path  = f"/tmp/awaaz_{session_id}_{int(time.time())}.wav"

        if not os.path.exists(pipe_path):
            print(f"[AUDIO] Asterisk pipe not found: {pipe_path}")
            return None

        try:
            with open(pipe_path, "rb") as f:
                raw = f.read(SAMPLE_RATE * 2 * 30)

            with wave.open(out_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(raw)

            return out_path
        except Exception as e:
            print(f"[AUDIO] Asterisk read error: {e}")
            return None


# ─────────────────────────────────────────────
# EMERGENCY DETECTION — fully model-based
# ─────────────────────────────────────────────

def check_emergency(text: str, lang: str,
                    model_processor: "ModelProcessor" = None) -> bool:
    """
    Detect emergency from transcribed text using LLM.
    Falls back to conservative False if LLM unavailable.
    """
    if not text or not text.strip():
        return False

    if model_processor and model_processor.groq_client:
        try:
            confirm_prompt = (
                f"Does this message describe an active emergency requiring "
                f"immediate government or medical help? "
                f"Consider ALL languages including regional Indian ones. "
                f"Answer only YES or NO.\n\nMessage: {text}"
            )
            resp = model_processor.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": confirm_prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            answer = resp.choices[0].message.content.strip().upper()
            is_emergency = answer.startswith("YES")
            print(f"[EMERGENCY] LLM: '{answer}' → {is_emergency}")
            return is_emergency
        except Exception as e:
            print(f"[EMERGENCY] LLM unavailable ({e})")

    # Ollama fallback
    if model_processor:
        try:
            import urllib.request
            payload = json.dumps({
                "model": OLLAMA_MODEL,
                "prompt": (
                    f"Is this an active emergency needing immediate help? "
                    f"Answer YES or NO only.\nMessage: {text}"
                ),
                "stream": False,
                "options": {"num_predict": 5, "temperature": 0.0},
            }).encode()
            req = urllib.request.Request(
                OLLAMA_URL, data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                answer = json.loads(r.read()).get("response","").strip().upper()
            return answer.startswith("YES")
        except Exception:
            pass

    return False


# ─────────────────────────────────────────────
# MAIN VOICE LOOP
# ─────────────────────────────────────────────

class AWAAZVoiceLoop:
    """The complete voice conversation loop."""

    def __init__(self, mode: str = "mic", input_file: Optional[str] = None):
        print("\n" + "="*60)
        print("  AWAAZ — Voice Grievance Assistant")
        print("  Loading models... (first run may take a minute)")
        print("="*60 + "\n")

        self.audio    = AudioInput(mode=mode, input_file=input_file)
        self.vad      = VADProcessor()
        self.stt      = STTProcessor()
        self.profiler = ProfileDetector()
        self.model    = ModelProcessor()
        self.tts      = TTSProcessor()
        self.profile  : Optional[CallerProfile] = None

        self.langdet  = TokenLevelLangDetector.get()
        self.translit = transliteration_hook

        self.vad.load()
        self.stt.load()

        print("\n[AWAAZ] All models loaded. Starting call.\n")

    def run(self):
        """Main loop — runs one full call session."""
        session_id   = str(uuid.uuid4())[:8]
        self.profile = CallerProfile(session_id=session_id)

        print(f"[SESSION] ID: {session_id}")
        print("-" * 40)

        for turn in range(1, MAX_TURNS + 1):
            self.profile.turn_number = turn
            print(f"\n[TURN {turn}] State: {self.profile.state}")

            # STEP 1: Record utterance
            audio_path = self.audio.record_utterance(session_id)

            if audio_path is None:
                silent_reply = self._handle_silence()
                self._speak(silent_reply)
                if turn >= 3:
                    print("[AWAAZ] Call ended — no speech detected.")
                    break
                continue

            # STEP 2: Detect language (turn 1 only)
            if turn == 1:
                lang, conf = self.stt.detect_language(audio_path)
                meta = get_lang_meta(lang, self.model.groq_client)
                self.profile.lang       = lang
                self.profile.lang_name  = meta.get("name", lang)
                self.profile.gtts_lang  = meta.get("gtts_lang", "hi")
                self.profile.script     = meta.get("script", "Unknown")
                self.profile.confidence = conf
                self.profile.detected_at_turn = turn
                print(f"[PROFILE] Language: {self.profile.lang_name} ({lang}) conf:{conf:.2f}")

            # STEP 3: Transcribe
            text = self.stt.transcribe(audio_path, self.profile.lang)

            if not text:
                self._speak(self._handle_silence())
                continue

            print(f"[CITIZEN] {text}")

            # STEP 4a: Token-level language detection
            self.langdet.update_profile(text, self.profile)

            # STEP 4b: Detect accent + formality (turn 1-2)
            if turn <= 2:
                self.profile.accent_region = self.profiler.detect_accent(
                    audio_path, self.profile.lang
                )
                self.profile.formality_score, self.profile.formality_label = \
                    self.profiler.detect_formality(text, self.profile.lang)
                print(f"[PROFILE] Accent: {self.profile.accent_region} | "
                      f"Formality: {self.profile.formality_label}")

            # STEP 5: Emergency check
            if check_emergency(text, self.profile.lang, self.model):
                self.profile.is_emergency = True
                self.profile.state = "EMERGENCY"

            # STEP 6: Update state machine
            self._update_state(text)

            # STEP 7: Generate reply
            reply = self.model.generate(text, self.profile)

            # STEP 8: Update history
            self.profile.history.append({
                "citizen":   text,
                "assistant": reply,
                "turn":      turn,
            })

            # STEP 8b: Normalise reply for TTS
            reply_for_tts = self.model.normalize_for_tts_llm(reply, self.profile)

            # STEP 8c: Transliteration hook (pluggable)
            reply_for_tts = self.translit.process(reply_for_tts, self.profile)

            # STEP 9: Speak reply
            print(f"[AWAAZ]   {reply_for_tts}")
            self._speak(reply_for_tts)

            # STEP 10: Check for call end
            if self.profile.state in ("CLOSING", "EMERGENCY") and turn > 1:
                print("\n[AWAAZ] Call complete.")
                self._print_summary()
                break

            # File mode: single turn only
            if self.audio.mode == "file":
                print("\n[AWAAZ] File mode — single turn complete.")
                self._print_summary()
                break

        else:
            print("\n[AWAAZ] Max turns reached. Ending call.")

    def _update_state(self, text: str):
        """Simple state machine transitions."""
        state = self.profile.state
        turn  = self.profile.turn_number

        if state == "GREETING":
            self.profile.state = "GATHERING"

        elif state == "GATHERING":
            if turn >= 2 and len(text.split()) >= 5:
                self.profile.state = "CONFIRMING"

        elif state == "CONFIRMING":
            confirmed = self._llm_detect_confirmation(text)
            if confirmed:
                self.profile.state = "FILING"
            else:
                self.profile.state = "GATHERING"

        elif state == "FILING":
            self.profile.state = "CLOSING"

    def _llm_detect_confirmation(self, text: str) -> bool:
        """Ask LLM if the citizen's response is an affirmation."""
        if not text.strip():
            return False
        try:
            if self.model and self.model.groq_client:
                resp = self.model.groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Is the following a confirmation or agreement? "
                            f"Works in any language. Answer YES or NO only.\n"
                            f"Text: {text}"
                        )
                    }],
                    max_tokens=5,
                    temperature=0.0,
                )
                answer = resp.choices[0].message.content.strip().upper()
                return answer.startswith("YES")
        except Exception:
            pass
        
        words = text.strip().split()
        return len(words) <= 3 and "?" not in text

    def _handle_silence(self) -> str:
        """Reply for silent turns."""
        lang      = self.profile.lang if self.profile else "hi"
        lang_name = (self.profile.lang_name or lang) if self.profile else lang

        try:
            if self.model and self.model.groq_client:
                resp = self.model.groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Generate a short phone-call prompt asking if the caller "
                            f"is still there. Write in {lang_name} language ({lang}). "
                            f"Maximum 8 words. Friendly tone. No markdown."
                        )
                    }],
                    max_tokens=30,
                    temperature=0.3,
                )
                return resp.choices[0].message.content.strip()
        except Exception:
            pass
        
        return "Hello? Are you still there?"

    def _speak(self, text: str):
        """Synthesize and play TTS reply."""
        if not self.profile:
            return
        out_path = f"/tmp/awaaz_reply_{self.profile.session_id}_{self.profile.turn_number}.wav"
        success  = self.tts.synthesize(text, self.profile, out_path)
        if success and os.path.exists(out_path):
            self.tts.play(out_path)

    def _print_summary(self):
        """Print session summary at call end."""
        p = self.profile
        print("\n" + "="*60)
        print("  CALL SUMMARY")
        print("="*60)
        print(f"  Session ID    : {p.session_id}")
        print(f"  Language      : {p.lang_name} ({p.lang})")
        print(f"  Lang mode     : {p.lang_mode}")
        dist_str = ", ".join(f"{k}:{int(v*100)}%" for k, v in p.lang_distribution.items())
        print(f"  Distribution  : {dist_str or 'n/a'}")
        print(f"  Accent        : {p.accent_region}")
        print(f"  Formality     : {p.formality_label} ({p.formality_score})")
        print(f"  Emergency     : {'YES' if p.is_emergency else 'No'}")
        print(f"  Turns         : {p.turn_number}")
        print(f"  Final state   : {p.state}")
        print("-"*60)
        for i, h in enumerate(p.history, 1):
            print(f"  Turn {i}")
            print(f"    Citizen : {h['citizen']}")
            print(f"    AWAAZ   : {h['assistant']}")
        print("="*60 + "\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AWAAZ — Voice Grievance Assistant (single file)"
    )
    parser.add_argument(
        "--mode",
        choices=["mic", "file", "asterisk"],
        default="mic",
        help="Input mode: mic (live), file (WAV), asterisk (AGI pipe)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input WAV file (required for --mode file)"
    )
    args = parser.parse_args()

    if args.mode == "file" and not args.input:
        print("ERROR: --mode file requires --input path/to/audio.wav")
        sys.exit(1)

    loop = AWAAZVoiceLoop(mode=args.mode, input_file=args.input)
    loop.run()


if __name__ == "__main__":
    main()
