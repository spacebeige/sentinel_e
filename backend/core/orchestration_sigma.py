import logging
import asyncio
from typing import Dict, Any

try:
    from backend.sentinel.sentinel_sigma import SentinelSigmaOrchestrator
    from backend.core.ingestion import IngestionEngine
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from backend.sentinel.sentinel_sigma import SentinelSigmaOrchestrator
    from backend.core.ingestion import IngestionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PolyMath-Orchestrator")

class Orchestrator:
    def __init__(self):
        logger.info("Initializing Sentinel-Sigma Orchestrator...")
        self.sigma = SentinelSigmaOrchestrator()
        try:
            self.ingestion = IngestionEngine()
        except Exception:
            logger.warning("IngestionEngine failed to init (likely missing dep). Skipping.")
            self.ingestion = None
    
    async def process_request(self, text: str = None, file_path: str = None) -> Dict[str, Any]:
        """
        Unified entry point for analysis.
        Currently focused on Text Evidence for Sigma Protocol.
        """
        evidence_text = ""
        
        # 1. Handle File Input (e.g. PDF)
        if file_path:
            logger.info(f"Processing PDF: {file_path}")
            pass
            
        # 2. Handle Text Input
        if text:
            evidence_text += text + "\n"
            
        if not evidence_text.strip():
            return {"error": "No evidence provided."}

        # 3. Run Sigma Protocol
        logger.info("Running Sigma Diagnosis...")
        diagnosis = await self.sigma.diagnose(evidence_text)
        
        return {
            "status": "success",
            "protocol": "Sentinel-Sigma",
            "diagnosis": diagnosis
        }
