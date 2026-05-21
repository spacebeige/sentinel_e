from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("BrowserRuntime.Observer")


@dataclass
class BrowserState:
    title: str = ""
    url: str = ""
    visible_text: str = ""
    buttons: List[Dict[str, Any]] = field(default_factory=list)
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "visible_text": self.visible_text,
            "buttons": self.buttons,
            "inputs": self.inputs,
            "links": self.links,
            "metadata": self.metadata,
        }


class BrowserObserver:
    """Extracts compact structured browser state. Raw HTML is never returned."""

    def __init__(self, max_text_chars: int = 6000, max_elements: int = 80):
        self.max_text_chars = max_text_chars
        self.max_elements = max_elements

    async def observe(self, page, *, allow_ocr: bool = True) -> BrowserState:
        dom_state = await self._extract_dom_state(page)
        if self._state_has_content(dom_state):
            return dom_state

        if allow_ocr:
            ocr_text = await self._extract_ocr_text(page)
            if ocr_text:
                dom_state.visible_text = self._compact_text(ocr_text)
                dom_state.metadata["extraction_method"] = "ocr"
                return dom_state

        dom_state.metadata["extraction_method"] = "empty_dom"
        return dom_state

    async def extract_pdf_from_page(self, page) -> Dict[str, Any]:
        url = page.url or ""
        response = await page.context.request.get(url)
        if not response.ok:
            return {"ok": False, "error": f"Unable to fetch PDF: HTTP {response.status}"}
        pdf_bytes = await response.body()
        return self.extract_pdf_bytes(pdf_bytes, source=url)

    def extract_pdf_bytes(self, pdf_bytes: bytes, source: str = "") -> Dict[str, Any]:
        try:
            import fitz
        except Exception as exc:
            return {"ok": False, "error": f"PyMuPDF unavailable: {exc}"}

        try:
            document = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_texts: List[str] = []
            for page in document:
                text = page.get_text("text") or ""
                if text.strip():
                    page_texts.append(text)
            joined = self._compact_text("\n\n".join(page_texts), limit=self.max_text_chars)
            return {
                "ok": True,
                "source": source,
                "page_count": len(document),
                "text": joined,
                "scanned": not bool(joined.strip()),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "source": source}

    async def _extract_dom_state(self, page) -> BrowserState:
        data = await page.evaluate(
            """
            () => {
              const visible = (el) => {
                if (!el) return false;
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style && style.visibility !== 'hidden' && style.display !== 'none' && rect.width > 0 && rect.height > 0;
              };
              const textOf = (el) => (el.innerText || el.getAttribute('aria-label') || el.getAttribute('title') || '').trim();
              const cssPath = (el) => {
                if (!el || !el.tagName) return '';
                if (el.id) return `#${CSS.escape(el.id)}`;
                const attr = ['name','aria-label','placeholder','href'].find((name) => el.getAttribute(name));
                if (attr) return `${el.tagName.toLowerCase()}[${attr}="${CSS.escape(el.getAttribute(attr)).slice(0,80)}"]`;
                let path = el.tagName.toLowerCase();
                if (el.className && typeof el.className === 'string') {
                  const cls = el.className.trim().split(/\\s+/).slice(0,2).map((c) => `.${CSS.escape(c)}`).join('');
                  path += cls;
                }
                return path;
              };
              const collect = (selector, mapper, limit=80) =>
                Array.from(document.querySelectorAll(selector)).filter(visible).slice(0, limit).map(mapper);
              return {
                title: document.title || '',
                url: location.href,
                visible_text: (document.body?.innerText || '').replace(/\\s+/g, ' ').trim(),
                buttons: collect('button, [role="button"], input[type="button"], input[type="submit"]', (el) => ({
                  text: textOf(el) || el.value || '',
                  selector: cssPath(el),
                  type: el.getAttribute('type') || el.tagName.toLowerCase(),
                  disabled: Boolean(el.disabled) || el.getAttribute('aria-disabled') === 'true'
                })),
                inputs: collect('input, textarea, select', (el) => ({
                  selector: cssPath(el),
                  type: el.getAttribute('type') || el.tagName.toLowerCase(),
                  name: el.getAttribute('name') || '',
                  label: el.getAttribute('aria-label') || '',
                  placeholder: el.getAttribute('placeholder') || '',
                  value: el.type === 'password' ? '' : (el.value || ''),
                  disabled: Boolean(el.disabled)
                })),
                links: collect('a[href]', (el) => ({
                  text: textOf(el),
                  href: el.href,
                  selector: cssPath(el)
                }))
              };
            }
            """
        )
        return BrowserState(
            title=str(data.get("title", ""))[:180],
            url=str(data.get("url", ""))[:1000],
            visible_text=self._compact_text(data.get("visible_text", "")),
            buttons=self._compact_elements(data.get("buttons", [])),
            inputs=self._compact_elements(data.get("inputs", [])),
            links=self._compact_elements(data.get("links", [])),
            metadata={"extraction_method": "dom"},
        )

    async def _extract_ocr_text(self, page) -> str:
        try:
            screenshot = await page.screenshot(full_page=False, type="png")
        except Exception as exc:
            logger.debug("Screenshot OCR capture failed: %s", exc)
            return ""

        try:
            import easyocr
            from PIL import Image
            import numpy as np
        except Exception as exc:
            logger.debug("EasyOCR fallback unavailable: %s", exc)
            return ""

        try:
            image = Image.open(io.BytesIO(screenshot)).convert("RGB")
            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            results = reader.readtext(np.array(image), detail=0, paragraph=True)
            return "\n".join(str(item) for item in results if str(item).strip())
        except Exception as exc:
            logger.debug("EasyOCR extraction failed: %s", exc)
            return ""

    def _compact_elements(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compacted = []
        for item in (elements or [])[: self.max_elements]:
            cleaned = {}
            for key, value in item.items():
                if isinstance(value, str):
                    cleaned[key] = self._compact_text(value, limit=240)
                else:
                    cleaned[key] = value
            compacted.append(cleaned)
        return compacted

    def _compact_text(self, value: str, limit: Optional[int] = None) -> str:
        value = re.sub(r"\s+", " ", str(value or "")).strip()
        max_len = limit or self.max_text_chars
        return value[:max_len]

    @staticmethod
    def _state_has_content(state: BrowserState) -> bool:
        return bool(
            state.visible_text.strip()
            or state.buttons
            or state.inputs
            or state.links
        )
