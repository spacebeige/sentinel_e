import logging
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
        self.ingestion = IngestionEngine()
    
    async def process_request(self, text: str = None, file_path: str = None) -> Dict[str, Any]:
        """
        Unified entry point for analysis.
        Currently focused on Text Evidence for Sigma Protocol.
        """

 
