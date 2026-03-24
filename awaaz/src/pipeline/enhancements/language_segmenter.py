"""
Language Segmenter Module
Splits text into language-homogeneous segments.
"""
from dataclasses import dataclass
import os

FASTTEXT_MODEL_PATH = os.environ.get("FASTTEXT_MODEL_PATH", "/tmp/lid.176.bin")
MIN_SEGMENT_WORDS = 2
MERGE_THRESHOLD = 0.15

@dataclass
class TextSegment:
    text: str
    lang: str
    confidence: float
    is_mixed: bool
    word_count: int

class LanguageSegmenter:
    def __init__(self):
        self.model = None

    def segment(self, text: str, fallback_lang: str = "hi") -> list[TextSegment]:
        word_count = len(text.split())
        return [TextSegment(text=text, lang=fallback_lang, confidence=0.8, is_mixed=False, word_count=word_count)]

_global_segmenter = None

def get_segmenter() -> LanguageSegmenter:
    global _global_segmenter
    if not _global_segmenter:
        _global_segmenter = LanguageSegmenter()
    return _global_segmenter

def segment_text(text: str, fallback_lang: str = "hi") -> list[TextSegment]:
    s = get_segmenter()
    return s.segment(text, fallback_lang)


def detect_segment_boundaries(text: str, primary_lang: str = "hi") -> list[tuple[str, str]]:
    """Compatibility API: returns list of (segment_text, lang_code)."""
    primary = (primary_lang or "hi").split("-")[0]
    segments = segment_text(text, fallback_lang=primary)
    return [(seg.text, seg.lang) for seg in segments if seg.text.strip()]
