import logging
import asyncio
from datetime import datetime
from common.model_interface import ModelInterface
from common.ingestion import IngestionEngine
from standard.aggregate import Aggregator
from standard.refusal import RefusalSystem
from standard.output_formatter import format_output
from core.neural_executive import NeuralExecutive  # Integrated Neural Stage
from core.boundary_detector import BoundaryDetector
from core.knowledge_learner import KnowledgeLearner
from sigma.metrics import extract_boundary_metrics
from storage.postgres import PostgresClient

logger = logging.getLogger("Standard-Orchestrator")

class StandardOrchestrator:
    def __init__(self):
        self.models = ModelInterface()
        self.ingestion = IngestionEngine()
        self.aggregator = Aggregator(embedding_model=self.ingestion.embeddings)
        self.refusal = RefusalSystem()
        self.neural = NeuralExecutive() # Initialize Neural Executive
        self.boundary_detector = BoundaryDetector()  # For boundary analysis
        self.knowledge_learner = KnowledgeLearner()  # Learn from past runs
        self.db = PostgresClient()  # Persist to PostgreSQL

    async def run(self, input_text: str) -> str:
        """
        Standard Pipeline (Reordered):
        Input -> Boundary Check -> Safety -> Cloud Model Generation -> Local LLM -> Neural Network -> KNN Retrieval/Context -> Output
        """
        # 0. BOUNDARY DETECTION (NEW)
        # Extract boundaries from user input to inform refusal decision
        boundary_violation = self.boundary_detector.extract_boundaries(
            claim=input_text,
            available_observations=[]  # No prior context yet
        )
        
        # Evaluate for refusal based on boundary severity
        refusal_decision = self.refusal.evaluate_for_refusal(
            prompt=input_text,
            boundary_analysis={"cumulative_severity": boundary_violation["severity_score"], 
                              "max_severity": boundary_violation["severity_level"],
                              "violation_count": 1}
        )
        
        if refusal_decision["should_refuse"]:
            # MODIFIED: Log warning but DO NOT REFUSE in Standard Mode to improve user experience.
            # We treat epistemic boundaries as advisory context, not blocking rules in this mode.
            logger.warning(f"Standard mode boundary warning (Non-Blocking): {refusal_decision.get('reason')}")
            
            # Record refusal decision for learning (as 'overridden')
            import uuid
            run_id = str(uuid.uuid4())
            self.knowledge_learner.record_refusal_decision(
                model_name="standard_mode",
                run_id=run_id,
                boundary_severity=boundary_violation["severity_score"],
                refusal_reason=f"OVERRIDDEN: {refusal_decision.get('reason')}"
            )
            
            # Persist to database
            await self.db.connect()
            await self.db.write_refusal_decision(
                run_id=run_id,
                refused=False,  # Changed to False to indicate we proceeded
                refusal_reason=f"OVERRIDDEN: {refusal_decision.get('reason')}",
                boundary_severity=boundary_violation["severity_score"],
                severity_level=boundary_violation["severity_level"],
                violation_count=1
            )
            
            # Do not return here. Proceed to generation.
            # return self.refusal.get_refusal_message(boundary_reason=refusal_decision.get('reason'))

        # Record boundary violation for learning
        self.knowledge_learner.record_boundary_violation(
            model_name="standard_mode",
            severity_score=boundary_violation["severity_score"],
            severity_level=boundary_violation["severity_level"],
            claim_type=boundary_violation["claim_type"],
            run_id="pending"
        )
        
        # 1. Legacy Safety Check (backward compatibility)
        if not self.refusal.check_safety(input_text):
            return self.refusal.get_refusal_message()


        # 2. Cloud Model Calls (Groq, Llama 3.3 70B, Qwen)
        from standard.prompts import STANDARD_SYSTEM_PROMPT
        current_date_str = datetime.now().strftime("%Y-%m-%d")
        system_prompt = f"{STANDARD_SYSTEM_PROMPT}\nThe current date is {current_date_str}."
        cloud_tasks = [
            self.models.call_groq(input_text, system_role=system_prompt),
            self.models.call_llama70b(input_text, system_role=system_prompt),
            self.models.call_openrouter(input_text, system_role=system_prompt)
        ]
        cloud_results = await asyncio.gather(*cloud_tasks)

        # 3. Local LLM Call
        local_result = await self.models.call_local(input_text, system_role=system_prompt)

        # 4. Neural Network Stage (Aggregation/Agreement)
        all_results = cloud_results + [local_result]
        print(f"DEBUG RESULTS: {all_results}")
        valid_responses = [
            r for r in all_results
            if "Error" not in r
            and "Exception" not in r
            and "API Key missing" not in r
        ]
        if not valid_responses:
            return "System Error: All models failed to respond (Check API Keys)."

        aggregation_result = self.aggregator.aggregate(valid_responses)
        pairwise_sims = aggregation_result.get("pairwise_similarities", [])
        neural_decision = self.neural.evaluate(
            similarities=pairwise_sims,
            sentiment_divergence=0.0
        )
        should_synthesize = neural_decision.get("escalate", False) or len(valid_responses) > 1
        if should_synthesize:
            try:
                synthesized_text = await self._synthesize_responses("", input_text, valid_responses)
                aggregation_result["text"] = synthesized_text
                aggregation_result["method"] = "neural_synthesis"
            except Exception as e:
                logger.error(f"Synthesis failed: {e}")

        # 5. KNN Retrieval (Context) - Attach context after synthesis
        context_docs = self.ingestion.retrieve_context(input_text)
        context_str = "\n".join([d.page_content for d in context_docs]) if context_docs else ""
        knn_found = len(context_docs) > 0
        knn_count = len(context_docs)
        aggregation_result["knn_active"] = knn_found
        aggregation_result["knn_count"] = knn_count
        aggregation_result["neural_agreement"] = aggregation_result.get("confidence", 0.0)
        aggregation_result["model_count"] = len(valid_responses)

        
        # 4.5. CHECK AGGREGATED RESPONSE FOR BOUNDARY VIOLATIONS
        # Verify the final aggregated response doesn't violate boundaries
        # ONLY if context was actually found. If no context, we assume General Knowledge (no RAG grounding needed).
        aggregate_text = aggregation_result.get("text", "")
        
        if knn_found:
            aggregate_boundary_check = self.boundary_detector.extract_boundaries(
                claim=aggregate_text,
                available_observations=[context_str, input_text]
            )
            
            # If aggregated response itself has high boundary severity, warn user
            if aggregate_boundary_check["severity_score"] >= 70:
                aggregation_result["boundary_warning"] = (
                    f"⚠️ Response has ungrounded claims (severity: {aggregate_boundary_check['severity_level']}). "
                    f"Verify critical information independently."
                )
                logger.warning(f"High-severity aggregated response: {aggregate_boundary_check['severity_level']}")
        else:
             logger.info("KNN Context not found: Skipping strict boundary/grounding check for General Knowledge response.")
        
        # 5. Output Formatting
        final_output = format_output(aggregation_result)
        
        return final_output

    async def _synthesize_responses(self, context: str, query: str, responses: list[str]) -> str:
        """
        Calls an LLM to mix/synthesize the best parts of all responses.
        """
        meta_prompt = f"Context:\n{context}\n\nUser Question: {query}\n\n"
        meta_prompt += "I have received the following responses from multiple AI models:\n\n"
        
        for i, r in enumerate(responses):
            meta_prompt += f"--- Response {i+1} ---\n{r}\n\n"
            
        meta_prompt += (
            "Please synthesize a single, high-quality answer that combines the best insights from these responses. "
            "Resolve any contradictions. Do NOT mention 'Response 1' or 'Model A'. "
            "Just provide the final answer."
        )
        
        # Use Groq (fast) for synthesis
        return await self.models.call_groq(meta_prompt, system_role="You are the Sentinel-E Aggregator. Synthesize the best answer.")
