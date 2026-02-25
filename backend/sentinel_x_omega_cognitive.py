import os
import json
from datetime import datetime
from statistics import mean
from tavily import TavilyClient
from dotenv import load_dotenv

from core.boundary_detector import BoundaryDetector
from core.neural_executive import NeuralExecutive

load_dotenv()


class SentinelXOmegaCognitive:

    def __init__(self):
        self.detector = BoundaryDetector()
        self.executive = NeuralExecutive()
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

        # Cloud LLM client
        from models.cloud_clients import CloudModelClient
        self.cloud_client = CloudModelClient()

        # Lightweight session memory (no full replay)
        self.last_summary = None

    # --------------------------------------------------
    # Retrieval (Strictly Limited)
    # --------------------------------------------------

    def retrieve(self, query):
        response = self.tavily.search(
            query=query,
            search_depth="advanced",
            max_results=3
        )
        return response.get("results", [])

    # --------------------------------------------------
    # Aggressive Compression
    # --------------------------------------------------

    def condense(self, results):

        condensed = []

        for r in results[:2]:
            text = (r.get("content") or "")[:180]
            condensed.append(text)

        return condensed

    # --------------------------------------------------
    # Short Reasoning (Default Mode)
    # --------------------------------------------------

    def short_reason(self, claim, condensed):

        context = "\n".join(condensed)

        prompt = f"""
You are a CTO-level advisor.

Previous Context Summary:
{self.last_summary or "None"}

Question:
{claim}

Context:
{context}

Respond clearly in under 80 words.
If uncertain, say so explicitly.
"""

        return self.call_reasoning_model(prompt, max_tokens=120)

    # --------------------------------------------------
    # Deep Structured Reasoning (Escalation)
    # --------------------------------------------------

    def deep_reason(self, claim, condensed):

        context = "\n".join(condensed)

        prompt = f"""
You are a senior systems architect.

Previous Context Summary:
{self.last_summary or "None"}

Problem:
{claim}

Context:
{context}

Return strict JSON:
{{
  "executive_summary": "",
  "probable_causes": [],
  "risk_zones": [],
  "uncertainty_level": "low | medium | high",
  "recommended_actions": []
}}

Be precise and concise.
"""

        response = self.call_reasoning_model(prompt, max_tokens=250)

        try:
            return json.loads(response)
        except:
            return {
                "executive_summary": "Model response formatting issue.",
                "probable_causes": [],
                "risk_zones": [],
                "uncertainty_level": "high",
                "recommended_actions": []
            }

    # --------------------------------------------------
    # Reasoning Model Call (YOU CONNECT YOUR MODEL HERE)
    # --------------------------------------------------

    def call_reasoning_model(self, prompt, max_tokens=150):
        """
        Calls Groq LLaMA-3.1 via CloudModelClient (async to sync).
        You can switch to Llama 3.3 70B or Qwen by changing the method.
        """
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Default: Groq
            result = loop.run_until_complete(
                self.cloud_client.call_groq(prompt)
            )
            return result
        except Exception as e:
            return f"Cloud LLM error: {e}"
        finally:
            loop.close()

    # --------------------------------------------------
    # Main Cognitive Flow
    # --------------------------------------------------

    def run(self, claim):

        # Step 1: Retrieval
        results = self.retrieve(claim)

        # Step 2: Compression
        condensed = self.condense(results)

        # Step 3: Primary short reasoning
        primary_output = self.short_reason(claim, condensed)

        # Step 4: Executive evaluation (placeholder similarity)
        similarities = [0.8, 0.75]
        sentiment_divergence = 0.2

        executive_state = self.executive.evaluate(
            similarities=similarities,
            sentiment_divergence=sentiment_divergence
        )

        # Step 5: Escalation
        if executive_state["escalate"]:
            final_output = self.deep_reason(claim, condensed)
        else:
            final_output = primary_output

        # Step 6: Governance check (independent)
        core = self.detector.extract_boundaries(claim, condensed)
        violations = core.get("boundary_violations", [])
        avg_severity = mean(
            [v.get("severity_score", 0) for v in violations]
        ) if violations else 0

        # Update lightweight memory
        if isinstance(final_output, str):
            self.last_summary = final_output[:200]
        else:
            self.last_summary = json.dumps(final_output)[:200]

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "answer": final_output,
            "executive_state": executive_state,
            "omega_metrics": {
                "average_severity": avg_severity,
                "structural_integrity": max(0.0, 1 - avg_severity / 100),
                "knowledge_sources_used": len(condensed)
            }
        }


# --------------------------------------------------
# Fully Interactive Mode
# --------------------------------------------------

if __name__ == "__main__":

    engine = SentinelXOmegaCognitive()

    print("\n🛡️ Sentinel-XΩ CTO Cognitive Interactive Mode")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting Sentinel-XΩ.")
            break

        result = engine.run(user_input)

        print("\n🔎 Answer:")
        print(result["answer"])

        print("\n🧭 Executive State:")
        print(json.dumps(result["executive_state"], indent=4))

        print("\n🔐 Omega Metrics:")
        print(json.dumps(result["omega_metrics"], indent=4))
        print("\n---\n")