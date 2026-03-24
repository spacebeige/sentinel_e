"""Integration-oriented unit tests for ElevenLabs STT/TTS routing and language mirroring."""

import os
import sys
import asyncio
from types import MethodType

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.stt import STTProcessor
from src.pipeline.tts import TTSProcessor
from src.pipeline.nlp import ModelProcessor
from test_live_voice import _normalize_lang_code


class DummySession:
    def __init__(self, lang: str = "en", gtts_lang: str = "en"):
        self.lang = lang
        self.gtts_lang = gtts_lang


def test_stt_prioritizes_elevenlabs_when_configured(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "dummy")
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    monkeypatch.setenv("HF_API_KEY", "dummy")

    stt = STTProcessor(preferred_provider="elevenlabs")
    assert stt.providers[0].name == "elevenlabs_stt"


def test_stt_can_prioritize_groq(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "dummy")
    monkeypatch.setenv("GROQ_API_KEY", "dummy")

    stt = STTProcessor(preferred_provider="groq")
    assert stt.providers[0].name == "groq_whisper"


def test_tts_prefers_elevenlabs_before_groq(monkeypatch, tmp_path):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "dummy")
    monkeypatch.setenv("GROQ_API_KEY", "dummy")

    tts = TTSProcessor(preferred_provider="elevenlabs")

    calls = []

    async def fake_elevenlabs_synthesize(self, text, output_path):
        calls.append("elevenlabs")
        with open(output_path, "wb") as f:
            f.write(b"RIFF....WAVE")
        return True

    async def fake_groq_synthesize(self, text, output_path, language="en"):
        calls.append("groq")
        return True

    tts.elevenlabs.synthesize = MethodType(fake_elevenlabs_synthesize, tts.elevenlabs)
    tts.groq.synthesize = MethodType(fake_groq_synthesize, tts.groq)

    out_path = tmp_path / "reply.wav"
    ok = asyncio.run(tts.synthesize("Hello there", DummySession(lang="en", gtts_lang="en"), str(out_path)))

    assert ok is True
    assert calls == ["elevenlabs"]


def test_language_alignment_guard_for_english_and_hindi():
    mp = ModelProcessor()

    assert mp._is_reply_language_aligned("Hello, I can help you today.", "en") is True
    assert mp._is_reply_language_aligned("नमस्ते, मैं आपकी सहायता कर सकता हूँ।", "en") is False

    assert mp._is_reply_language_aligned("नमस्ते, आपकी शिकायत दर्ज कर दी गई है।", "hi") is True
    assert mp._is_reply_language_aligned("I have registered your complaint.", "hi") is False


def test_language_alignment_guard_for_mixed_profile():
    mp = ModelProcessor()
    assert mp._is_reply_language_aligned("Aapka request approved hai.", "hi-en") is True


def test_english_alias_normalization():
    assert _normalize_lang_code("english_us") == "en"
    assert _normalize_lang_code("eng") == "en"
