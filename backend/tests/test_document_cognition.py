import asyncio
from core.document_cognition import build_document_cognition


def test_document_cognition_handles_missing_document():
    result = asyncio.run(build_document_cognition(None, None))
    assert result["available"] is False
    assert result["reason"] == "no_document"


def test_document_cognition_never_returns_raw_payload_for_invalid_image():
    result = asyncio.run(build_document_cognition("not-base64", "image/png", filename="scan.png"))
    assert result["available"] is True
    assert result["document_type"] == "image"
    assert "semantic_context" in result
    assert "not-base64" not in str(result)
    assert result["requires_model_escalation"] is True
