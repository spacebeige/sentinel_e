"""TTS Router Module.

Selects segment language/priority without directly invoking providers.
"""

from dataclasses import dataclass


@dataclass
class EngineScore:
    engine_id: str
    score: float
    reason: str


class TTSRouter:
    def select_engine(self, lang: str, is_emergency: bool = False) -> str:
        if is_emergency:
            return "low_latency"
        if "en" in (lang or "").lower():
            return "english_pref"
        return "indic_pref"


def route_tts(segments: list, session_lang: str) -> list:
    """Compatibility API expected by earlier pipeline.

    Args:
        segments: list[dict] where each dict has text/lang_type or list[(text, lang)]
        session_lang: session language profile

    Returns:
        list of route descriptors containing synthesizer_lang.
    """
    routed = []
    base_lang = (session_lang or "hi").split("-")[0]

    for seg in segments:
        if isinstance(seg, tuple):
            text, lang = seg
            lang_type = "indic" if lang != "en" else "english"
        else:
            text = seg.get("text", "")
            lang_type = seg.get("lang_type", "indic")

        mapped_lang = base_lang if lang_type == "indic" else "en"
        routed.append(
            {
                "text": text,
                "engine": "primary" if lang_type == "indic" else "fallback_or_mixed",
                "synthesizer_lang": mapped_lang,
            }
        )

    return routed


def route_and_generate(text: str, chunk_metadata: dict, context: dict) -> bytes:
    """Legacy compatibility shim.

    Actual synthesis is handled in tts_pipeline_v2 using existing_tts.
    """
    raise RuntimeError("route_and_generate is deprecated in this build; use EnhancedTTSProcessor.synthesize")
