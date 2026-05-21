"""
Document cognition normalization for Sentinel-E.

Local-first strategy:
  1. PyMuPDF text extraction for PDFs.
  2. OCR for typed images/screenshots.
  3. Compact semantic structure for orchestration context.
  4. Escalation flags only when local extraction is insufficient.

No raw document bytes or raw HTML are sent through this layer.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from core.file_preprocessor import MIN_TEXT_THRESHOLD, preprocess_file


HEADING_RE = re.compile(r"^([A-Z][A-Z0-9 ,:/()&.-]{5,}|#{1,4}\s+.+)$")


def _compact_text(text: Optional[str], max_chars: int) -> str:
    if not text:
        return ""
    normalized = re.sub(r"\n{3,}", "\n\n", str(text)).strip()
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    return normalized[:max_chars]


def _extract_sections(text: str, max_sections: int = 8) -> List[Dict[str, Any]]:
    if not text:
        return []

    sections: List[Dict[str, Any]] = []
    current_title = "Document Body"
    current_lines: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        is_heading = bool(HEADING_RE.match(line)) and len(line) <= 120
        if is_heading and current_lines:
            sections.append({
                "title": current_title,
                "text": _compact_text("\n".join(current_lines), 700),
            })
            current_lines = []
            current_title = line.lstrip("#").strip()
            if len(sections) >= max_sections:
                break
        elif is_heading:
            current_title = line.lstrip("#").strip()
        else:
            current_lines.append(line)

    if current_lines and len(sections) < max_sections:
        sections.append({
            "title": current_title,
            "text": _compact_text("\n".join(current_lines), 700),
        })

    return sections[:max_sections]


def _document_type(file_mime: Optional[str]) -> str:
    if file_mime == "application/pdf":
        return "pdf"
    if file_mime and file_mime.startswith("image/"):
        return "image"
    return "unknown"


def build_document_cognition(
    file_b64: Optional[str],
    file_mime: Optional[str] = None,
    filename: Optional[str] = None,
    max_context_chars: int = 5000,
) -> Dict[str, Any]:
    """
    Return compact document cognition metadata for orchestration.

    The `semantic_context` string is safe to inject into model prompts. The
    original bytes/base64 are never included in the returned structure.
    """
    if not file_b64:
        return {
            "available": False,
            "reason": "no_document",
        }

    processed = preprocess_file(file_b64, file_mime)
    extracted_text = _compact_text(processed.get("extracted_text"), max_context_chars)
    doc_type = _document_type(file_mime)
    local_sufficient = len(extracted_text) >= MIN_TEXT_THRESHOLD
    extraction_method = "pymupdf" if doc_type == "pdf" and extracted_text else "ocr" if extracted_text else "none"
    requires_ocr = doc_type in {"pdf", "image"} and not local_sufficient
    requires_model_escalation = doc_type == "image" and not local_sufficient

    sections = _extract_sections(extracted_text)
    metadata = processed.get("metadata") if isinstance(processed.get("metadata"), dict) else {}

    semantic_context_parts = [
        f"Document type: {doc_type}",
        f"Filename: {filename or 'attached document'}",
        f"Extraction method: {extraction_method}",
    ]
    if metadata:
        semantic_context_parts.append(f"Metadata: {metadata}")
    if sections:
        semantic_context_parts.append("Semantic sections:")
        semantic_context_parts.extend(
            f"- {section['title']}: {section['text']}"
            for section in sections
            if section.get("text")
        )
    elif extracted_text:
        semantic_context_parts.append(f"Extracted text: {extracted_text}")

    return {
        "available": True,
        "document_type": doc_type,
        "filename": filename,
        "extraction_method": extraction_method,
        "local_extraction_sufficient": local_sufficient,
        "requires_ocr_fallback": requires_ocr,
        "requires_model_escalation": requires_model_escalation,
        "skip_vision_api": bool(processed.get("skip_vision_api")),
        "text_char_count": len(extracted_text),
        "metadata": metadata,
        "sections": sections,
        "semantic_context": _compact_text("\n".join(semantic_context_parts), max_context_chars),
    }
