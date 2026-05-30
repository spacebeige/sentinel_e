import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any

# Ensure tests can import backend modules
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from metacognitive.cognitive_gateway import CognitiveModelGateway, CognitiveGatewayOutput


@pytest.fixture
def mock_db():
    """Provides a mocked async database session."""
    db = AsyncMock()
    # Mock specific SQLAlchemy methods if needed
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    return db


@pytest.fixture
def mock_user() -> Dict[str, Any]:
    """Provides a standard mocked user payload."""
    return {
        "uid": "test_user_123",
        "email": "test@example.com",
        "email_verified": True
    }


@pytest.fixture
def mock_gateway():
    """Provides a mocked CognitiveModelGateway that returns successful default outputs."""
    gateway = CognitiveModelGateway()
    
    async def mock_invoke(model_key: str, gateway_input: Any, **kwargs):
        return CognitiveGatewayOutput(
            model_name=model_key,
            raw_output=f"[MOCKED] Output from {model_key} for: {gateway_input.user_query}",
            success=True,
            tokens_used=100,
            input_tokens=50,
            output_tokens=50,
            confidence_estimate=0.85
        )
        
    gateway.invoke_model = AsyncMock(side_effect=mock_invoke)
    
    async def mock_parallel(model_keys, gateway_input, **kwargs):
        tasks = [mock_invoke(key, gateway_input) for key in model_keys]
        return await asyncio.gather(*tasks)
        
    gateway.invoke_parallel = AsyncMock(side_effect=mock_parallel)
    gateway.invoke_parallel_failsafe = AsyncMock(side_effect=mock_parallel)
    
    return gateway
