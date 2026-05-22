"""
============================================================
PDF Streaming Engine — Sentinel-E v8.1
============================================================
Progressive, chunked, and hierarchical summary pipeline for large PDFs.
Avoids full-context loads into the main model, reducing token usage and latency.
"""

import base64
import logging
from typing import AsyncGenerator, Dict, Any, Optional

logger = logging.getLogger("PDFStreaming")

class PDFStreamingProcessor:
    """
    Processes large PDFs progressively by chunking and yielding intermediate summaries.
    """
    
    def __init__(self, chunk_size: int = 4000):
        self.chunk_size = chunk_size

    async def extract_and_summarize_stream(self, pdf_b64: str, max_pages: int = 100) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Yields chunked text and metadata progressively.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF not installed — cannot stream PDF")
            yield {"error": "PyMuPDF missing"}
            return

        try:
            pdf_bytes = base64.b64decode(pdf_b64)
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = min(len(doc), max_pages)

            current_chunk_text = ""
            current_chunk_pages = []
            chunk_index = 0

            for i in range(total_pages):
                page_text = doc[i].get_text("text").strip()
                if not page_text:
                    continue

                current_chunk_text += f"\n[Page {i+1}]\n{page_text}\n"
                current_chunk_pages.append(i + 1)

                if len(current_chunk_text) >= self.chunk_size:
                    yield {
                        "chunk_index": chunk_index,
                        "pages": current_chunk_pages,
                        "text": current_chunk_text,
                        "progress_pct": int(((i + 1) / total_pages) * 100)
                    }
                    chunk_index += 1
                    current_chunk_text = ""
                    current_chunk_pages = []

            # Yield remaining
            if current_chunk_text:
                yield {
                    "chunk_index": chunk_index,
                    "pages": current_chunk_pages,
                    "text": current_chunk_text,
                    "progress_pct": 100
                }

            doc.close()

        except Exception as e:
            logger.error(f"PDF Streaming failed: {e}")
            yield {"error": str(e)}

    async def summarize_hierarchically(self, pdf_b64: str) -> str:
        """
        Consumes the stream, calls a fast model to summarize each chunk,
        and then summarizes the summaries.
        """
        from metacognitive.cognitive_gateway import CognitiveModelGateway
        from metacognitive.schemas import CognitiveGatewayInput, QueryMode
        gateway = CognitiveModelGateway()
        
        summaries = []
        async for chunk in self.extract_and_summarize_stream(pdf_b64):
            if "error" in chunk:
                break
            
            prompt = f"Summarize the following section of a document covering pages {chunk['pages']}:\n\n{chunk['text']}\n\nFocus on key claims, data, and decisions."
            
            # Fast summary pass
            try:
                # Use fastest available model
                gw_input = CognitiveGatewayInput(user_query=prompt, mode=QueryMode.RAW)
                output = await gateway.invoke_model("llama31-8b", gw_input)
                if output.success and output.raw_output:
                    summaries.append(output.raw_output)
            except Exception as e:
                logger.warning(f"Chunk summary failed: {e}")
                
        if not summaries:
            return "No extractable text found in document."
            
        if len(summaries) == 1:
            return summaries[0]
            
        # Final hierarchical synthesis
        meta_prompt = "Synthesize the following section summaries into a single cohesive executive summary of the document:\n\n"
        for i, s in enumerate(summaries):
            meta_prompt += f"--- Section {i+1} ---\n{s}\n\n"
            
        try:
            gw_meta_input = CognitiveGatewayInput(user_query=meta_prompt, mode=QueryMode.RAW)
            meta_output = await gateway.invoke_model("llama31-8b", gw_meta_input)
            if meta_output.success and meta_output.raw_output:
                return meta_output.raw_output
            return "\n\n".join(summaries)
        except Exception:
            return "\n\n".join(summaries)
