import pytest
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memory.behavioral_memory import BehavioralMemoryManager

@pytest.mark.asyncio
async def test_get_profile_defaults():
    # Test that get_profile returns default dictionary when get_memory_by_key returns None
    db_mock = AsyncMock()
    with patch("memory.behavioral_memory.get_memory_by_key", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = None
        
        profile = await BehavioralMemoryManager.get_profile(db_mock, "user-123")
        
        assert profile["preferred_reasoning_depth"] == "balanced"
        assert profile["correction_frequency"] == 0
        assert "model_satisfaction" in profile
        mock_get.assert_called_once_with(db_mock, user_id="user-123", key="behavioral_profile")

@pytest.mark.asyncio
async def test_get_profile_existing():
    # Test that get_profile loads JSON from memory entry
    db_mock = AsyncMock()
    memory_mock = MagicMock()
    memory_mock.value = {
        "preferred_reasoning_depth": "deep",
        "correction_frequency": 3,
        "model_satisfaction": {}
    }
    
    with patch("memory.behavioral_memory.get_memory_by_key", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = memory_mock
        
        profile = await BehavioralMemoryManager.get_profile(db_mock, "user-123")
        
        assert profile["preferred_reasoning_depth"] == "deep"
        assert profile["correction_frequency"] == 3

@pytest.mark.asyncio
async def test_update_profile_async():
    # Test updating profile with interaction metrics
    db_mock = AsyncMock()
    
    # Mock get_profile to return default dict
    default_profile = {
        "interaction_patterns": {},
        "preferred_reasoning_depth": "balanced",
        "model_satisfaction": {},
        "workflow_habits": {},
        "correction_frequency": 0
    }
    
    with patch.object(BehavioralMemoryManager, "get_profile", new_callable=AsyncMock) as mock_get_profile, \
         patch("memory.behavioral_memory.upsert_memory", new_callable=AsyncMock) as mock_upsert:
         
        mock_get_profile.return_value = default_profile
        
        metrics = {
            "model": "gemini-3.5-pro",
            "latency_ms": 1200,
            "vote": "down",
            "query_complexity": "complex"
        }
        
        await BehavioralMemoryManager.update_profile_async(db_mock, "user-123", metrics)
        
        # Verify the updated profile was upserted correctly
        mock_upsert.assert_called_once()
        call_kwargs = mock_upsert.call_args[1]
        assert call_kwargs["user_id"] == "user-123"
        assert call_kwargs["key"] == "behavioral_profile"
        
        updated_val = call_kwargs["value"]
        assert updated_val["preferred_reasoning_depth"] == "deep"
        assert updated_val["correction_frequency"] == 1
        assert updated_val["model_satisfaction"]["gemini-3.5-pro"]["negative"] == 1

def test_format_profile_for_prompt():
    # Test semantic prompt formatting
    profile_deep = {"preferred_reasoning_depth": "deep", "correction_frequency": 6}
    formatted_deep = BehavioralMemoryManager.format_profile_for_prompt(profile_deep)
    assert "User prefers deep, analytical reasoning." in formatted_deep
    assert "User frequently corrects outputs" in formatted_deep

    profile_concise = {"preferred_reasoning_depth": "concise", "correction_frequency": 1}
    formatted_concise = BehavioralMemoryManager.format_profile_for_prompt(profile_concise)
    assert "User prefers concise, direct synthesis." in formatted_concise
    assert "User frequently corrects outputs" not in formatted_concise
