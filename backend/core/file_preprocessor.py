"""
Local file preprocessing for images and PDFs.
Uses OCR (pytesseract + OpenCV) and PDF extraction (PyMuPDF) to extract text
BEFORE sending to vision APIs — minimizing token usage and API costs.

Strategy:
  1. PDF → PyMuPDF text extraction (free, no API call)
  2. Image → OpenCV preprocessing + pytesseract OCR
  3. Image → resize/compress to reduce base64 size for vision API
  4. If local extraction yields sufficient text → skip vision API entirely
"""

import base64
import io
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Minimum chars of extracted text to consider "sufficient" (skip vision API)
MIN_TEXT_THRESHOLD = 100


# ═══════════════════════════════════════════════════════════
# PDF EXTRACTION (PyMuPDF / fitz)
# ═══════════════════════════════════════════════════════════

def extract_pdf_text(pdf_b64: str, max_pages: int = 10, max_chars: int = 8000) -> Optional[str]:
    """Extract text from a base64-encoded PDF using PyMuPDF.
    Returns extracted text or None if extraction fails."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not installed — skipping local PDF extraction")
        return None

    try:
        pdf_bytes = base64.b64decode(pdf_b64)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        pages = min(len(doc), max_pages)
        text_parts = []
        total_chars = 0

        for i in range(pages):
            page = doc[i]
            page_text = page.get_text("text").strip()
            if page_text:
                text_parts.append(f"[Page {i+1}]\n{page_text}")
                total_chars += len(page_text)
                if total_chars >= max_chars:
                    break

        doc.close()

        if not text_parts:
            logger.info("PDF has no extractable text (likely scanned/image-based)")
            return None

        result = "\n\n".join(text_parts)
        if len(result) > max_chars:
            result = result[:max_chars] + "\n...[truncated]"

        logger.info(f"PDF text extracted: {len(result)} chars from {pages} pages")
        return result

    except Exception as e:
        logger.warning(f"PDF text extraction failed: {e}")
        return None


def extract_pdf_metadata(pdf_b64: str) -> dict:
    """Extract PDF metadata (page count, title, author)."""
    try:
        import fitz
        pdf_bytes = base64.b64decode(pdf_b64)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        meta = {
            "page_count": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
        }
        doc.close()
        return meta
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════
# IMAGE OCR (OpenCV + pytesseract)
# ═══════════════════════════════════════════════════════════

def extract_image_text(image_b64: str) -> Optional[str]:
    """Run OCR on a base64-encoded image using OpenCV preprocessing + pytesseract."""
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError:
        logger.warning("OpenCV/Pillow not installed — skipping OCR")
        return None

    try:
        import pytesseract
    except ImportError:
        logger.warning("pytesseract not installed — skipping OCR")
        return None

    try:
        image_bytes = base64.b64decode(image_b64)
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Convert to OpenCV format
        img_array = np.array(pil_image.convert("RGB"))
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Preprocessing for better OCR accuracy
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        # Adaptive threshold for varying lighting
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, h=10)

        # Run OCR
        text = pytesseract.image_to_string(denoised).strip()

        if text and len(text) > 10:
            logger.info(f"OCR extracted {len(text)} chars from image")
            return text

        return None

    except Exception as e:
        logger.warning(f"Image OCR failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════
# IMAGE COMPRESSION (reduce base64 size for vision API)
# ═══════════════════════════════════════════════════════════

def compress_image_b64(
    image_b64: str,
    max_dimension: int = 1024,
    quality: int = 75,
) -> Tuple[str, str]:
    """Compress and resize a base64 image to reduce token usage.
    Returns (compressed_b64, mime_type)."""
    try:
        from PIL import Image
    except ImportError:
        return image_b64, "image/jpeg"

    try:
        image_bytes = base64.b64decode(image_b64)
        pil_image = Image.open(io.BytesIO(image_bytes))

        original_size = len(image_b64)

        # Resize if larger than max_dimension
        w, h = pil_image.size
        if max(w, h) > max_dimension:
            ratio = max_dimension / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)

        # Convert to RGB (drop alpha channel) and compress as JPEG
        if pil_image.mode in ("RGBA", "P"):
            pil_image = pil_image.convert("RGB")

        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=quality, optimize=True)
        compressed_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        compressed_size = len(compressed_b64)
        reduction = ((original_size - compressed_size) / original_size) * 100
        logger.info(
            f"Image compressed: {original_size//1024}KB → {compressed_size//1024}KB "
            f"({reduction:.0f}% reduction)"
        )

        return compressed_b64, "image/jpeg"

    except Exception as e:
        logger.warning(f"Image compression failed: {e}")
        return image_b64, "image/jpeg"


# ═══════════════════════════════════════════════════════════
# UNIFIED PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════

def preprocess_file(
    file_b64: str,
    file_mime: Optional[str] = None,
) -> dict:
    """Preprocess a file (image or PDF) locally before sending to vision API.

    Returns dict with:
        extracted_text: str or None — text extracted locally (OCR or PDF)
        compressed_b64: str — compressed version of image (or original for PDF)
        compressed_mime: str — MIME type after compression
        metadata: dict — file metadata
        skip_vision_api: bool — True if local extraction is sufficient
    """
    result = {
        "extracted_text": None,
        "compressed_b64": file_b64,
        "compressed_mime": file_mime or "image/png",
        "metadata": {},
        "skip_vision_api": False,
    }

    is_pdf = file_mime and file_mime == "application/pdf"

    if is_pdf:
        # PDF: extract text with PyMuPDF
        text = extract_pdf_text(file_b64)
        meta = extract_pdf_metadata(file_b64)
        result["extracted_text"] = text
        result["metadata"] = meta
        # If we got substantial text, skip the vision API
        if text and len(text) >= MIN_TEXT_THRESHOLD:
            result["skip_vision_api"] = True
            logger.info(f"PDF preprocessing: {len(text)} chars extracted — skipping vision API")
        else:
            logger.info("PDF has minimal text — will use vision API for analysis")
    else:
        # Image: OCR + compression
        ocr_text = extract_image_text(file_b64)
        result["extracted_text"] = ocr_text

        # Compress image for vision API
        compressed_b64, compressed_mime = compress_image_b64(file_b64)
        result["compressed_b64"] = compressed_b64
        result["compressed_mime"] = compressed_mime

        # If OCR found substantial text, we can still use vision API but augment
        # Don't skip vision for images — visual context matters beyond text
        if ocr_text and len(ocr_text) >= MIN_TEXT_THRESHOLD:
            logger.info(f"Image OCR: {len(ocr_text)} chars — will augment vision API prompt")

    return result
