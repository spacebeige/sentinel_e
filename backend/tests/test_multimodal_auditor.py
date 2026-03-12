"""Quick validation of multimodal auditor phases."""
import asyncio
from core.multimodal_auditor import (
    MultimodalAuditor,
    phase1_inspect_input,
    phase2_capability_check,
    phase3_model_availability_audit,
    phase4_auto_recovery,
    phase5_build_pipeline,
    phase6_integration_check,
    InputType,
    AuditStatus,
    audit_request,
)

print("=" * 60)
print("MULTIMODAL AUDITOR VALIDATION")
print("=" * 60)

# Phase 1: Text-only
r1 = phase1_inspect_input("What is quantum computing?")
print(f"\n[Phase 1] Text-only: type={r1.input_type.value}, multimodal={r1.multimodal_required}")
assert r1.input_type == InputType.TEXT_ONLY
assert not r1.multimodal_required

# Phase 1: Image input
r2 = phase1_inspect_input("Analyze this image", image_b64="abc123", image_mime="image/png")
print(f"[Phase 1] Image+text: type={r2.input_type.value}, multimodal={r2.multimodal_required}")
assert r2.input_type == InputType.MULTIMODAL
assert r2.multimodal_required

# Phase 1: Image only (no text)
r2b = phase1_inspect_input("", image_b64="abc123", image_mime="image/jpeg")
print(f"[Phase 1] Image-only:  type={r2b.input_type.value}, multimodal={r2b.multimodal_required}")
assert r2b.input_type == InputType.IMAGE_INPUT

# Phase 1: PDF
r4 = phase1_inspect_input("Analyze this document", file_mime="application/pdf")
print(f"[Phase 1] PDF+text: type={r4.input_type.value}, multimodal={r4.multimodal_required}")
assert r4.input_type == InputType.DOCUMENT_ANALYSIS

# Phase 1: URL detection
r3 = phase1_inspect_input("Check this https://example.com/page and tell me")
print(f"[Phase 1] URL detected: has_urls={r3.has_urls}, urls={r3.detected_urls}")
assert r3.has_urls

# Phase 2: Capability check for image
cap = phase2_capability_check(r2)
print(f"\n[Phase 2] Image caps: required={cap.required_capabilities}")
print(f"  preferred={cap.preferred_models}")
print(f"  fallback={cap.fallback_models}")
assert "vision" in cap.required_capabilities

# Phase 2: Capability check for text
cap_text = phase2_capability_check(r1)
print(f"[Phase 2] Text caps: required={cap_text.required_capabilities}")
print(f"  preferred={cap_text.preferred_models}")
assert "text_reasoning" in cap_text.required_capabilities

# Phase 3: Model availability audit
entries, disabled = phase3_model_availability_audit()
print(f"\n[Phase 3] Registry: {len(entries)} models, {len(disabled)} disabled")
for d in disabled:
    print(f"  DISABLED: {d.reason}")

# Phase 5: Pipeline construction (image)
pipeline_img = phase5_build_pipeline(r2, cap)
print(f"\n[Phase 5] Image pipeline:")
print(f"  analysis     = {pipeline_img.analysis_model}")
print(f"  critique     = {pipeline_img.critique_models}")
print(f"  synthesis    = {pipeline_img.synthesis_model}")
print(f"  verification = {pipeline_img.verification_model}")
print(f"  total models = {pipeline_img.model_count}")

# Models may be 0 if no API keys are configured (CI/test environment)
has_models = len(cap.preferred_models) > 0
if has_models:
    assert pipeline_img.model_count >= 3, f"Expected >= 3 models, got {pipeline_img.model_count}"
else:
    print("  (no API keys configured — pipeline empty as expected)")

# Phase 5: Pipeline construction (text)
pipeline_text = phase5_build_pipeline(r1, cap_text)
print(f"\n[Phase 5] Text pipeline:")
print(f"  analysis     = {pipeline_text.analysis_model}")
print(f"  critique     = {pipeline_text.critique_models}")
print(f"  synthesis    = {pipeline_text.synthesis_model}")
print(f"  verification = {pipeline_text.verification_model}")
print(f"  total models = {pipeline_text.model_count}")
if has_models:
    assert pipeline_text.model_count >= 3, f"Expected >= 3 models, got {pipeline_text.model_count}"
else:
    print("  (no API keys configured — pipeline empty as expected)")


# Full audit (no execution)
async def test_full_audit():
    result = await audit_request(
        query="What is deep learning?",
        image_b64=None,
        image_mime=None,
    )
    print(f"\n[Full Audit - Text]")
    print(f"  input_type: {result['SYSTEM_AUDIT']['input_type']}")
    print(f"  selected_models: {result['SYSTEM_AUDIT']['selected_models']}")
    print(f"  disabled_models: {len(result['SYSTEM_AUDIT']['disabled_models'])}")
    print(f"  execution_status: {result['EXECUTION_STATUS']}")
    print(f"  latency: {result['audit_latency_ms']:.2f}ms")

    result2 = await audit_request(
        query="What do you see in this image?",
        image_b64="dummy_base64",
        image_mime="image/png",
    )
    print(f"\n[Full Audit - Multimodal]")
    print(f"  input_type: {result2['SYSTEM_AUDIT']['input_type']}")
    print(f"  multimodal_required: {result2['SYSTEM_AUDIT']['multimodal_required']}")
    print(f"  selected_models: {result2['SYSTEM_AUDIT']['selected_models']}")
    print(f"  pipeline analysis: {result2['MODEL_PIPELINE']['analysis_model']}")
    print(f"  pipeline critique: {result2['MODEL_PIPELINE']['critique_models']}")
    print(f"  execution_status: {result2['EXECUTION_STATUS']}")

asyncio.run(test_full_audit())

print("\n" + "=" * 60)
print("ALL VALIDATIONS PASSED")
print("=" * 60)
