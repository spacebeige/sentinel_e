"""
AWAAZ — Speech-to-Text Only Pipeline
Simplified version: Captures audio → VAD → Language Detection → Transcription → Output

INSTALL:
    pip install faster-whisper silero-vad fasttext soundfile numpy scipy

    # fastText model (126MB, download once):
    wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin \\
         -O /tmp/lid.176.bin

USAGE:
    # Live microphone:
    python awaaz_voice_stt_only.py --mode mic

    # Pre-recorded WAV file:
    python awaaz_voice_stt_only.py --mode file --input path/to/audio.wav

    # Asterisk AGI pipe:
    python awaaz_voice_stt_only.py --mode asterisk
"""

import os
import sys
import uuid
import json
import wave
import struct
import argparse
import tempfile
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

try:
    import soundfile as sf
except ImportError:
    sf = None

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

SAMPLE_RATE = 16000
CHUNK_SIZE = 512
SILENCE_THRESHOLD = 0.7
MAX_UTTERANCE_S = 30
WHISPER_MODEL = "small"
MAX_TURNS = 10
FASTTEXT_MODEL_PATH = os.environ.get("FASTTEXT_MODEL_PATH", "/tmp/lid.176.bin")
MIXED_LANG_THRESHOLD = 0.20


# ─────────────────────────────────────────────
# CALLER PROFILE
# ─────────────────────────────────────────────

@dataclass
class CallerProfile:
    """State container for a single call."""
    session_id: str
    lang: Optional[str] = None
    lang_name: Optional[str] = None
    lang_mode: str = "pure"  # "pure" or "mixed"
    lang_distribution: Dict[str, float] = field(default_factory=dict)
    accent_region: Optional[str] = None
    formality_score: float = 0.5
    formality_label: str = "STANDARD"
    script: Optional[str] = None
    gtts_lang: Optional[str] = None
    confidence: float = 0.0
    turn_number: int = 0
    history: list = field(default_factory=list)
    is_emergency: bool = False


# ─────────────────────────────────────────────
# VAD PROCESSOR
# ─────────────────────────────────────────────

class VADProcessor:
    """Voice Activity Detection using Silero VAD."""

    def __init__(self):
        self.silero_model = None
        self.silero_utils = None
        self.energy_threshold = 300

    def load(self):
        """Load Silero VAD model from HuggingFace."""
        try:
            import torch
            self.silero_model, self.silero_utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            print(f"  ✓ Silero VAD loaded (mode: PyTorch)")
        except Exception as e:
            print(f"  ⚠ Silero VAD failed: {e} — falling back to energy-based VAD")
            self.silero_model = None

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Detect if audio chunk contains speech."""
        if self.silero_model is not None:
            try:
                import torch
                audio_tensor = torch.from_numpy(audio_chunk)
                confidence = self.silero_model(audio_tensor, SAMPLE_RATE).item()
                return confidence > 0.5
            except Exception:
                pass

        # Fallback: energy-based detection
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        return rms > self.energy_threshold

    def find_utterance_boundaries(self, audio: np.ndarray) -> list:
        """Find list of (start_frame, end_frame) tuples for detected utterances."""
        frames = len(audio) // CHUNK_SIZE
        boundaries = []
        in_speech = False
        speech_start = 0
        silence_frames = 0

        for i in range(frames):
            chunk = audio[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
            if len(chunk) < CHUNK_SIZE:
                break

            speech = self.is_speech(chunk)

            if speech and not in_speech:
                speech_start = i * CHUNK_SIZE
                in_speech = True
                silence_frames = 0
            elif speech:
                silence_frames = 0
            elif in_speech and not speech:
                silence_frames += 1
                if silence_frames > int(SILENCE_THRESHOLD * SAMPLE_RATE / CHUNK_SIZE):
                    boundaries.append((speech_start, i * CHUNK_SIZE))
                    in_speech = False

        if in_speech:
            boundaries.append((speech_start, len(audio)))

        return boundaries


# ─────────────────────────────────────────────
# STT PROCESSOR
# ─────────────────────────────────────────────

class STTProcessor:
    """Speech-to-Text using Faster Whisper."""

    def __init__(self):
        self.model = None
        self.model_size = WHISPER_MODEL

    def load(self):
        """Load faster-whisper model."""
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                self.model_size,
                device="cuda" if self._has_cuda() else "cpu",
                compute_type="float32",
            )
            print(f"  ✓ Faster-Whisper loaded ({self.model_size}, CPU)")
        except Exception as e:
            print(f"  ✗ Failed to load Whisper: {e}")
            raise

    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def detect_language(self, audio_path: str) -> Tuple[str, float]:
        """Detect language from audio file."""
        if not self.model:
            return "en", 0.0

        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=None,  # auto-detect
                vad_filter=True,
                beam_size=5,
            )
            return info.language, info.language_probability
        except Exception as e:
            print(f"Error detecting language: {e}")
            return "en", 0.0

    def transcribe(self, audio_path: str, lang: str) -> str:
        """Transcribe audio file to text."""
        if not self.model:
            return ""

        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=lang,
                beam_size=5,
                vad_filter=False,
            )
            text = " ".join([segment.text.strip() for segment in segments])
            return text
        except Exception as e:
            print(f"Error transcribing: {e}")
            return ""


# ─────────────────────────────────────────────
# TOKEN-LEVEL LANGUAGE DETECTOR
# ─────────────────────────────────────────────

class TokenLevelLangDetector:
    """Detect mixed language at word level using fastText."""

    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    def __init__(self):
        self.model = None

    def _load(self):
        """Load fastText language identification model."""
        if not os.path.exists(FASTTEXT_MODEL_PATH):
            print(f"\n  ⚠ fastText model not found at {FASTTEXT_MODEL_PATH}")
            print(f"    Download with: wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -O {FASTTEXT_MODEL_PATH}")
            return

        try:
            import fasttext
            fasttext.FastText.eprint = lambda x: None  # suppress warnings
            self.model = fasttext.load_model(FASTTEXT_MODEL_PATH)
            print(f"  ✓ fastText language model loaded")
        except Exception as e:
            print(f"  ⚠ Failed to load fastText: {e}")

    def _predict_word(self, word: str) -> str:
        """Predict language for a single word."""
        if not self.model or not word.strip():
            return None

        try:
            pred = self.model.predict(word.replace("\n", ""), k=1)
            if pred and pred[0]:
                lang_code = pred[0][0].replace("__label__", "")
                return lang_code
        except:
            pass
        return None

    def detect(self, text: str, sentence_lang: str) -> Tuple[str, dict]:
        """
        Detect mixed languages at token level.
        Returns (lang_mode, distribution) where:
        - lang_mode: "pure" or "mixed"
        - distribution: {"hi": 0.7, "en": 0.3, ...}
        """
        if not self.model:
            return "pure", {sentence_lang: 1.0}

        words = text.split()
        lang_counts = {}

        for word in words:
            lang = self._predict_word(word)
            if lang:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1

        total = sum(lang_counts.values())
        if total == 0:
            return "pure", {sentence_lang: 1.0}

        dist = {lang: count / total for lang, count in lang_counts.items()}
        primary_lang = max(dist, key=dist.get)
        primary_fraction = dist[primary_lang]

        # Check if any non-primary language exceeds threshold
        for lang, frac in dist.items():
            if lang != primary_lang and frac >= MIXED_LANG_THRESHOLD:
                return "mixed", dist

        return "pure", {primary_lang: 1.0}

    def update_profile(self, text: str, profile: CallerProfile) -> None:
        """Update caller profile with language distribution."""
        if not profile.lang:
            return

        lang_mode, dist = self.detect(text, profile.lang)
        profile.lang_mode = lang_mode
        profile.lang_distribution = dist


# ─────────────────────────────────────────────
# AUDIO INPUT
# ─────────────────────────────────────────────

class AudioInput:
    """Capture audio from mic, file, or Asterisk."""

    def __init__(self, mode: str = "mic", input_file: Optional[str] = None):
        self.mode = mode
        self.input_file = input_file
        self.file_obj = None
        self.wav_reader = None

        if mode == "file" and input_file:
            try:
                self.file_obj = open(input_file, "rb")
                self.wav_reader = wave.open(self.file_obj, "rb")
            except Exception as e:
                print(f"Error opening {input_file}: {e}")

    def record_utterance(self, session_id: str) -> Optional[str]:
        """Record a single utterance and return path to WAV file."""
        audio = None

        if self.mode == "mic":
            audio = self._record_from_mic()
        elif self.mode == "file":
            audio = self._read_from_file()
        elif self.mode == "asterisk":
            audio = self._read_asterisk_pipe()

        if audio is None or len(audio) == 0:
            return None

        # Save to temp file
        out_path = f"/tmp/{session_id}_utterance.wav"
        self._save_wav(out_path, audio)
        return out_path

    def _record_from_mic(self) -> Optional[np.ndarray]:
        """Record from microphone."""
        try:
            import pyaudio
        except ImportError:
            print("pyaudio not installed: pip install pyaudio")
            return None

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        print("  [Recording... speak now, or press Ctrl+C to stop]")
        frames = []
        silent_chunks = 0

        try:
            while silent_chunks < int(SILENCE_THRESHOLD * SAMPLE_RATE / CHUNK_SIZE):
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.float32)
                frames.append(chunk)

                rms = np.sqrt(np.mean(chunk ** 2))
                if rms < 300:
                    silent_chunks += 1
                else:
                    silent_chunks = 0
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        return np.concatenate(frames) if frames else None

    def _read_from_file(self) -> Optional[np.ndarray]:
        """Read pre-recorded WAV file."""
        if not self.wav_reader:
            return None

        try:
            frames = self.wav_reader.readframes(self.wav_reader.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            return audio
        except Exception as e:
            print(f"Error reading WAV: {e}")
            return None

    def _read_asterisk_pipe(self) -> Optional[np.ndarray]:
        """Read from Asterisk AGI raw audio pipe."""
        try:
            audio_data = sys.stdin.buffer.read(CHUNK_SIZE * 2)
            if not audio_data:
                return None
            audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            return audio
        except Exception as e:
            print(f"Error reading from Asterisk: {e}")
            return None

    def _save_wav(self, path: str, audio: np.ndarray):
        """Save audio to WAV file."""
        try:
            sf.write(path, audio, SAMPLE_RATE)
        except Exception as e:
            print(f"Error saving WAV: {e}")


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

class AWAAZVoiceLoop:
    """STT-only voice loop."""

    def __init__(self, mode: str = "mic", input_file: Optional[str] = None):
        self.mode = mode
        self.session_id = str(uuid.uuid4())
        self.audio_input = AudioInput(mode=mode, input_file=input_file)
        self.vad = VADProcessor()
        self.stt = STTProcessor()
        self.lid = TokenLevelLangDetector.get()
        self.profile = CallerProfile(session_id=self.session_id)

        print("Loading models for STT-only pipeline...")
        self.vad.load()
        self.stt.load()
        print("Models loaded successfully.\n")

    def run(self):
        print(f"[{self.session_id}] AWAAZ STT-only pipeline started in {self.mode} mode.")
        print("Listening for speech...\n")

        try:
            while self.profile.turn_number < MAX_TURNS:
                audio_path = self.audio_input.record_utterance(self.session_id)
                if not audio_path:
                    break

                self.profile.turn_number += 1
                print(f"\n--- Turn {self.profile.turn_number} ---")

                # Detect language (first turn only)
                if not self.profile.lang:
                    print("Detecting primary language...")
                    lang, conf = self.stt.detect_language(audio_path)
                    self.profile.lang = lang
                    self.profile.confidence = conf
                    print(f"  Detected: {lang} (confidence: {conf:.2f})")

                # Transcribe
                print("Transcribing...")
                text = self.stt.transcribe(audio_path, self.profile.lang)
                print(f"  Transcript: {text.strip()}")

                if not text.strip():
                    print("  (No speech detected)")
                    continue

                # Update language profile
                self.lid.update_profile(text, self.profile)
                print(f"  Language Mode: {self.profile.lang_mode}")
                if self.profile.lang_distribution:
                    print(f"  Distribution: {self.profile.lang_distribution}")

        except KeyboardInterrupt:
            print("\n\nPipeline interrupted by user.")
        finally:
            print("\n[Pipeline finished]")


def main():
    parser = argparse.ArgumentParser(description="AWAAZ STT-Only Pipeline")
    parser.add_argument(
        "--mode",
        choices=["mic", "file", "asterisk"],
        default="mic",
        help="Input mode: mic (live), file (WAV), asterisk (AGI pipe)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input WAV file (required for --mode file)",
    )
    args = parser.parse_args()

    if args.mode == "file" and not args.input:
        print("ERROR: --mode file requires --input /path/to/audio.wav")
        sys.exit(1)

    loop = AWAAZVoiceLoop(mode=args.mode, input_file=args.input)
    loop.run()


if __name__ == "__main__":
    main()
