"""Regression tests for multimodal auditor phases and end-to-end audit payload."""

import asyncio

from core.multimodal_auditor import (
    InputType,
    phase1_inspect_input,
    phase2_capability_check,
    phase3_model_availability_audit,
    phase5_build_pipeline,
    audit_request,
)


def test_phase1_input_classification():
    text_only = phase1_inspect_input("What is quantum computing?")
    assert text_only.input_type == InputType.TEXT_ONLY
    assert not text_only.multimodal_required

    image_plus_text = phase1_inspect_input(
        "Analyze this image", image_b64="abc123", image_mime="image/png"
    )
    assert image_plus_text.input_type == InputType.MULTIMODAL
    assert image_plus_text.multimodal_required

    image_only = phase1_inspect_input("", image_b64="abc123", image_mime="image/jpeg")
    assert image_only.input_type == InputType.IMAGE_INPUT

    pdf_plus_text = phase1_inspect_input(
        "Analyze this document", file_mime="application/pdf"
    )
    assert pdf_plus_text.input_type == InputType.DOCUMENT_ANALYSIS

    with_url = phase1_inspect_input("Check this https://example.com/page and tell me")
    assert with_url.has_urls
    assert with_url.detected_urls


def test_phase2_and_phase5_pipeline_building():
    image_req = phase1_inspect_input(
        "Analyze this image", image_b64="abc123", image_mime="image/png"
    )
    image_caps = phase2_capability_check(image_req)
    assert "vision" in image_caps.required_capabilities

    text_req = phase1_inspect_input("What is deep learning?")
    text_caps = phase2_capability_check(text_req)
    assert "text_reasoning" in text_caps.required_capabilities

    image_pipeline = phase5_build_pipeline(image_req, image_caps)
    text_pipeline = phase5_build_pipeline(text_req, text_caps)

    # In CI/dev without API keys, selected model count can be zero.
    if image_caps.preferred_models:
        assert image_pipeline.model_count >= 3
    if text_caps.preferred_models:
        assert text_pipeline.model_count >= 3


def test_phase3_registry_audit_shape():
    entries, disabled = phase3_model_availability_audit()
    assert isinstance(entries, list)
    assert isinstance(disabled, list)



def test_full_audit_payload():
    text_result = asyncio.run(
        audit_request(
            query="What is deep learning?",
            image_b64=None,
            image_mime=None,
        )
    )
    assert "SYSTEM_AUDIT" in text_result
    assert "MODEL_PIPELINE" in text_result
    assert "EXECUTION_STATUS" in text_result
    assert "audit_latency_ms" in text_result

    mm_result = asyncio.run(
        audit_request(
            query="What do you see in this image?",
            image_b64="dummy_base64",
            image_mime="image/png",
        )
    )
    assert mm_result["SYSTEM_AUDIT"]["multimodal_required"] is True
    assert "analysis_model" in mm_result["MODEL_PIPELINE"]
