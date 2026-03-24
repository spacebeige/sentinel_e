"""Test suite for AWAAZ core modules."""

import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.session_store import SessionStore, AWAAZSession
from src.vad import VADProcessor
from src.barge_in_gate import BargeInGate
from src.pipeline.nlp import parse_llm_output, check_emergency
import numpy as np


@pytest.mark.asyncio
async def test_session_store_create():
    """Test SessionStore creation."""
    store = SessionStore()
    session = await store.create("test_channel_123")
    
    assert session.channel_id == "test_channel_123"
    assert session.is_active is True
    assert session.state == "GREETING"


@pytest.mark.asyncio
async def test_session_store_get_by_channel():
    """Test SessionStore retrieval by channel."""
    store = SessionStore()
    session1 = await store.create("ch_1")
    
    retrieved = await store.get_by_channel("ch_1")
    assert retrieved is not None
    assert retrieved.session_id == session1.session_id


@pytest.mark.asyncio
async def test_session_store_update():
    """Test SessionStore field update."""
    store = SessionStore()
    session = await store.create("ch_1")
    
    await store.update(session.session_id, lang="ta", state="GATHERING")
    
    updated = await store.get(session.session_id)
    assert updated.lang == "ta"
    assert updated.state == "GATHERING"


def test_vad_silence_detection():
    """Test VAD silence boundary detection."""
    vad = VADProcessor(silence_ms=700, sample_rate=16000)
    
    # Create 1 second of silence (should not trigger utterance)
    silence = np.zeros(16000, dtype=np.int16)
    result = vad.process_chunk(silence.tobytes())
    
    assert result is None  # No speech, so no utterance


def test_parse_llm_output_json():
    """Test JSON output parsing."""
    json_output = '{"reply": "Hello", "meta": {"grievance_category": "GR-01"}}'
    reply, meta = parse_llm_output(json_output)
    
    assert reply == "Hello"
    assert meta["grievance_category"] == "GR-01"


def test_parse_llm_output_plain_text():
    """Test plain text output parsing."""
    plain_output = "This is a simple reply"
    reply, meta = parse_llm_output(plain_output)
    
    assert reply == plain_output
    assert meta == {}


def test_emergency_detection_hindi():
    """Test emergency keyword detection in Hindi."""
    text = "bachao, iska koi hal nahi"
    is_emergency = check_emergency(text, "hi", None)
    
    assert is_emergency is True


def test_emergency_detection_false_positive():
    """Test that normal text is not flagged as emergency."""
    text = "mera pani bill nahi aa raha"
    is_emergency = check_emergency(text, "hi", None)
    
    assert is_emergency is False


def test_barge_in_gate():
    """Test barge-in gate logic."""
    session = AWAAZSession()
    gate = BargeInGate(session)
    
    # Start playback
    asyncio.run(gate.enable())
    assert gate._playing is True
    
    # Caller speaks during playback
    should_stop = gate.on_vad_activity(has_speech=True)
    assert should_stop is True
    assert session.barge_in_pending is True
    
    # Stop playback
    asyncio.run(gate.disable())
    assert gate._playing is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
