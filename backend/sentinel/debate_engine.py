import logging
import asyncio
import json
from typing import List, Dict, Any

logger = logging.getLogger("Sentinel-Debate-Engine")

class SentinelDebateEngine:
    def __init__(self, client):
        self.client = client

    async def run_debate(self, topic: str, rounds: int = 3) -> Dict[str, Any]:
        """
        Orchestrates multi-round debate between models based on strict rules:
        Round 1: Model A generates, B & C critique.
        Round 2: Model B revises, A & C critique.
        Round 3: Model C synthesizes.
        """
        logger.info(f"Starting debate on: {topic}")
        
        debate_history = []
        
        # --- ROUND 1 ---
        logger.info("Debate Round 1: Initial Position")
        # Model A (Mistral) generates structured position
        pos_a = await self.client.call_mistral(
            prompt=f"Topic: {topic}\nGenerate a structured position statement.",
            system_role="You are Model A (Analytical). Be precise and structured."
        )
        debate_history.append({"round": 1, "step": "position_a", "content": pos_a})
        
        # Model B (Groq) & C (Qwen via OpenRouter) critique
        critique_tasks = [
            self.client.call_groq(
                prompt=f"Critique this position:\n{pos_a}",
                system_role="You are Model B (Critical). Find flaws."
            ),
            self.client.call_qwenvl(
                prompt=f"Critique this position:\n{pos_a}",
                system_role="You are Model C (Qwen - Synthesizer/Validation). Assess valid points and nuances."
            )
        ]
        critiques = await asyncio.gather(*critique_tasks, return_exceptions=True)
        critique_b = critiques[0] if isinstance(critiques[0], str) else "Error"
        critique_c = critiques[1] if isinstance(critiques[1], str) else "Error"
        
        debate_history.append({"round": 1, "step": "critiques", "model_b": critique_b, "model_c": critique_c})

        
        if rounds < 2:
             return self._synthesize_early(topic, debate_history)

        # --- ROUND 2 ---
        logger.info("Debate Round 2: Revision")
        # Model B (Groq) Revises based on critiques (acting as proposer of counter-position or revision)
        # Actually prompt says "Model B revises". Let's have B propose a revised stance.
        revision_prompt = f"""
        Original Position (A): {pos_a}
        Critiques:
        B: {critique_b}
        C: {critique_c}
        
        Revise the position to address validity of critiques.
        """
        revision_b = await self.client.call_groq(
            prompt=revision_prompt,
            system_role="You are Model B. Revise the position to be stronger."
        )
        debate_history.append({"round": 2, "step": "revision_b", "content": revision_b})
        
        # A and C critique the revision
        r2_critique_tasks = [
            self.client.call_mistral(
                prompt=f"Critique this revised position:\n{revision_b}",
                system_role="You are Model A. Defend original nuance or accept improvement."
            ),
            self.client.call_qwenvl(
                prompt=f"Critique this revised position:\n{revision_b}",
                system_role="You are Model C (Qwen). Detailed validation of changes."
            )
        ]
        r2_critiques = await asyncio.gather(*r2_critique_tasks, return_exceptions=True)
        r2_critique_a = r2_critiques[0] if isinstance(r2_critiques[0], str) else "Error"
        r2_critique_c = r2_critiques[1] if isinstance(r2_critiques[1], str) else "Error"
        
        debate_history.append({"round": 2, "step": "critiques", "model_a": r2_critique_a, "model_c": r2_critique_c})

        if rounds < 3:
             return self._synthesize_early(topic, debate_history)

        # --- ROUND 3 ---
        logger.info("Debate Round 3: Synthesis")
        # Model C Synthesizes everything
        synthesis_prompt = f"""
        Debate Topic: {topic}
        
        Round 1 (A): {pos_a}
        Critiques: {critique_b}, {critique_c}
        
        Round 2 (B - Revised): {revision_b}
        Critiques: {r2_critique_a}, {r2_critique_c}
        
        Synthesize the final position.
        Identify:
        1. Stable consensus
        2. Remaining divergence
        3. Confidence shift
        
        Output valid JSON with keys: "synthesis_text", "consensus" (list), "divergence" (list).
        """
        
        synthesis = await self.client.call_qwenvl(synthesis_prompt, system_role="You are Model C (Final Synthesizer - Qwen). Synthesize fairly.")
        
        return {
            "history": debate_history,
            "synthesis": synthesis,
            "rounds_executed": 3
        }

    def _synthesize_early(self, topic: str, history: List) -> Dict:
        # Fallback for fewer rounds
        return {
            "history": history,
            "synthesis": json.dumps({
                "synthesis_text": "Debate concluded early.",
                "consensus": [],
                "divergence": []
            }),
            "rounds_executed": len(history)
        }
