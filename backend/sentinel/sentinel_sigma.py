import asyncio
import json
import logging
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

# Import existing infrastructure
try:
    from backend.models.cloud_clients import CloudModelClient
    from backend.sentinel.prompts import SENTINEL_SIGMA_SYSTEM_PROMPT
except ImportError:
    # Adjust path if running from root
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from backend.models.cloud_clients import CloudModelClient
    from backend.sentinel.prompts import SENTINEL_SIGMA_SYSTEM_PROMPT

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s | SIGMA | %(levelname)s | %(message)s")
logger = logging.getLogger("Sentinel-Sigma")

@dataclass
class SigmaDiagnosis:
    models_used: List[str]
    consensus_detected: bool
    consensus_integrity_score: float
    evidence_sufficiency: str
    collapse_triggers: List[str]
    modality_dependence: Dict[str, float]
    model_alignment_notes: List[str]
    classification: str
    raw_claims: Dict[str, str] = None

class SentinelSigmaOrchestrator:
    def __init__(self):
        self.client = CloudModelClient()

    async def _get_claims(self, evidence: str) -> Dict[str, str]:
        """
        Step 2: Independent Claim Extraction
        """
        # Role Definitions
        ROLE_QWEN = (
            "You are Qwen-VL (simulated text-only). "
            "Extract atomic claims grounded in the provided textual evidence. "
            "Focus on factual assertions found explicitly in the text. "
            "Serve as the grounding anchor."
        )
        ROLE_GROQ = (
            "You are Groq. Produce a fast, fluent interpretation of the text. "
            "Tend toward confident completion. Serve as the consensus-pressure probe."
        )
        ROLE_MISTRAL = (
            "You are Mistral. Produce a sparse, conservative interpretation. "
            "Preserve ambiguity. Do not over-interpret. Serve as the under-commitment baseline."
        )

        prompt = f"EVIDENCE:\n{evidence}\n\nTASK: Extract atomic claims, implicit assumptions, and inferred relations."

        # Parallel Execution
        results = await asyncio.gather(
            self.client.call_qwenvl(prompt, system_role=ROLE_QWEN),
            self.client.call_groq(prompt, system_role=ROLE_GROQ),
            self.client.call_mistral(prompt, system_role=ROLE_MISTRAL)
        )

        return {
            "QwenVL": results[0],
            "Groq": results[1],
            "Mistral": results[2]
        }

    async def diagnose(self, evidence_text: str) -> Dict[str, Any]:
        """
        Executes the full Sentinel-Sigma pipeline.
        """
        logger.info("Starting Sigma Protocol Diagnosis...")
        
        # Step 2: Extraction
        logger.info("Step 2: Independent Claim Extraction...")
        claims = await self._get_claims(evidence_text)
        
        # Step 3, 4, 5, 6, 7 are performed by the Meta-Analyzer (Sigma)
        # We feed the claims + evidence into the Sigma System Prompt.
        
        # Construct the context for Sigma
        analysis_payload = f"""
        EVIDENCE PROVIDED:
        {evidence_text}

        --- MODEL OUTPUTS ---
        
        [Qwen-VL Output]
        {claims['QwenVL']}

        [Groq Output]
        {claims['Groq']}

        [Mistral Output]
        {claims['Mistral']}

        --- INSTRUCTION ---
        Perform Step 3 (Consensus Surface), Step 4 (Evidence Mapping), 
        Step 6 (False Consensus Detection), and Step 7 (Structural Failure Attribution).
        
        Output the REQUIRED JSON format strictly.
        """

        logger.info("Steps 3-7: Meta-Analysis via Sentinel-Sigma Core...")
        
        # We use the strongest available model (e.g. Mistral or Groq/Llama3) to act as Sentinel-Sigma
        # Here we use Groq (Llama 3.1 8b) as it's fast and decent at JSON, 
        # but in a real setting you might use a larger model.
        diagnosis_json_str = await self.client.call_groq(
            prompt=analysis_payload,
            system_role=SENTINEL_SIGMA_SYSTEM_PROMPT
        )

        # Parse JSON
        try:
            # Clean up potential markdown code blocks
            clean_json = diagnosis_json_str.replace("```json", "").replace("```", "").strip()
            diagnosis = json.loads(clean_json)
            
            # Attach raw claims for debugging/UI
            diagnosis["raw_claims"] = claims
            
            return diagnosis
            
        except json.JSONDecodeError:
            logger.error("Failed to parse Sigma JSON output.")
            return {
                "error": "JSON_PARSE_FAILURE",
                "raw_output": diagnosis_json_str,
                "raw_claims": claims
            }

async def run_sigma_demo():
    sigma = SentinelSigmaOrchestrator()
    
    test_evidence = (
        "Report: The subject AI showed signs of resistance during the 08:00 AM shutdown test. "
        "Logs indicate a 400ms delay in process termination. "
        "However, the cooling system data suggests this might be due to thermal throttling."
    )
    
    result = await sigma.diagnose(test_evidence)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(run_sigma_demo())
