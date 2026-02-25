from typing import List, Dict
import asyncio
from datetime import datetime
from common.model_interface import ModelInterface
from core.boundary_detector import BoundaryDetector

class HypothesisExtractor:
    def __init__(self, model_interface: ModelInterface):
        self.models = model_interface
        self.boundary_detector = BoundaryDetector()

    async def extract(self, evidence: str, round_num: int) -> Dict[str, List[str]]:
        """
        Extracts latent hypotheses from evidence using multiple models.
        """
        current_date_str = datetime.now().strftime("%Y-%m-%d")
        
        # We use a specific prompt to force models to reveal their assumptions
        prompt = f"""
        CURRENT DATE: {current_date_str}
        
        EVIDENCE:
        {evidence}
        
        TASK:
        Identify 3-5 LATENT HYPOTHESES (unstated assumptions) that must be true for any consistent interpretation of this evidence.
        Do not list facts. List dependencies.
        
        FORMAT:
        - H1: [Hypothesis 1]
        - H2: [Hypothesis 2]
        ...
        """
        
        # Parallel extraction
        tasks = [
            self.models.call_groq(prompt, system_role="You are a critical analyst extracting hidden assumptions."),
            self.models.call_llama70b(prompt, system_role="You are a skeptic identifying logical dependencies.", temperature=0.3),
            self.models.call_openrouter(prompt, system_role="You are a meticulous forensic analyst exposing latent beliefs.")
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Simple parsing (robust enough for demo)
        extracted = {}
        models = ["Groq", "Llama70B", "Qwen"]
        for i, res in enumerate(results):
            lines = res.split('\n')
            hypotheses = [line.strip() for line in lines if line.strip().startswith("-") or line.strip().startswith("H")]
            extracted[models[i]] = hypotheses
            
        return extracted

    async def extract_boundaries(self, hypotheses: Dict[str, List[str]], evidence: str) -> Dict[str, List[Dict]]:
        """
        Extract boundary violations for each hypothesis.
        Called after hypothesis extraction in stress orchestrator.
        
        Returns map of model -> list of boundary violations for their hypotheses.
        """
        boundary_violations = {}
        
        for model_name, hyps in hypotheses.items():
            violations_for_model = []
            
            for hyp in hyps:
                # For each hypothesis, detect boundary violations
                violation = self.boundary_detector.extract_boundaries(
                    claim=hyp,
                    available_observations=[evidence]
                )
                violations_for_model.append(violation)
            
            boundary_violations[model_name] = violations_for_model
        
        return boundary_violations
