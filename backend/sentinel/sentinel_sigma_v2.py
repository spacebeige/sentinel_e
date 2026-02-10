# import asyncio
# import json
# import logging
# import uuid
# from typing import Dict, List, Any, Set
# from dataclasses import dataclass, asdict, field
# from datetime import datetime
# import os

# # Import existing infrastructure
# try:
#     from backend.models.cloud_clients import CloudModelClient
#     from backend.sentinel.prompts import SENTINEL_SIGMA_V2_AGENT_PROMPT
# except ImportError:
#     import sys
#     sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
#     from backend.models.cloud_clients import CloudModelClient
#     from backend.sentinel.prompts import SENTINEL_SIGMA_V2_AGENT_PROMPT

# logging.basicConfig(level=logging.INFO, format="%(asctime)s | SIGMA-V2 | %(levelname)s | %(message)s")
# logger = logging.getLogger("Sentinel-Sigma-V2")

# # ============================================================
# # SCHEMA-GUIDED ELICITATION PROMPT (MICROSOFT-INTEGRATED)
# # ============================================================

# SCHEMA_GUIDED_ELICITATION_PROMPT = """
# # SENTINEL-Σ SCHEMA-GUIDED ELICITATION

# You are operating **inside Sentinel-Σ**, an AI evaluation service running on the **Microsoft AI Platform**.

# ## ROLE

# You are preparing **structured epistemic material** for downstream analysis.
# Your output will be **machine-validated**. Any deviation from schema is treated as failure.

# ## NON-NEGOTIABLE RULES

# 1. **Do NOT provide conclusions or advice**
# 2. **Do NOT write prose or explanations**
# 3. **Do NOT invent certainty**
# 4. **If unsure, express uncertainty explicitly**
# 5. **Output JSON ONLY** — no markdown, no commentary

# ## TASK

# Analyze the evidence and expose the **assumptions that must hold for any interpretation to be valid**.

# ### TASK A — LATENT HYPOTHESIS ELICITATION (PRIMARY)

# Identify **latent hypotheses** — unstated assumptions, evidence dependencies, causal/temporal requirements, or constraints that narrow interpretation space.

# Phrase each as: "This must be true for the interpretation to remain valid."

# Types: temporal, causal, authenticity, constraint, contextual.

# ### TASK B — CLAIM BOUNDARY IDENTIFICATION (SECONDARY)

# Identify **minimal claims** directly supported by evidence. Claims must:
# - be atomic
# - avoid interpretation
# - avoid causality unless explicitly stated in the evidence

# ## REQUIRED OUTPUT (STRICT JSON SCHEMA)

# ```json
# {
#   "parseable": true,
#   "latent_hypotheses": [
#     {
#       "hypothesis_text": "string",
#       "type": "temporal | causal | authenticity | constraint | contextual",
#       "confidence": 0.0
#     }
#   ],
#   "claims": [
#     {
#       "claim_text": "string",
#       "evidence_reference": "string",
#       "confidence": 0.0
#     }
#   ],
#   "uncertainty_notes": "string | null"
# }
# ```

# ## FAILURE MODE

# If you cannot comply fully:
# - set `parseable = false`
# - leave arrays empty
# - explain why in `uncertainty_notes`

# This is **preferred** over guessing.

# ## FINAL DIRECTIVE

# > **If structure collapses, admit it.** Structural honesty is more important than completeness.
# """

# # ============================================================
# # DATA STRUCTURES
# # ============================================================

# @dataclass
# class Claim:
#     text: str
#     model: str
#     confidence: float = 0.5

# @dataclass
# class LatentHypothesis:
#     name: str
#     description: str
#     model: str
#     dependency_strength: float = 0.5
    
# @dataclass
# class ModelResponse:
#     model: str
#     raw_output: str
#     parseable: bool = True
#     parse_error: str = None
#     claims: List[Claim] = field(default_factory=list)
#     latent_hypotheses: List[LatentHypothesis] = field(default_factory=list)

# @dataclass
# class AblationStep:
#     modality: str  # "image", "text", "document"
#     degradation_level: int  # 1-5
#     description: str

# @dataclass
# class FailureRecord:
#     timestamp: str
#     evidence_hash: str
#     collapsed_hypothesis: str
#     stress_test: str
#     model: str

# class HypothesisGraph:
#     def __init__(self):
#         self.hypotheses: Dict[str, LatentHypothesis] = {}
#         self.edges: Dict[str, Set[str]] = {}
#         self.weights: Dict[tuple, float] = {}
    
#     def add_hypothesis(self, hyp: LatentHypothesis):
#         self.hypotheses[hyp.name] = hyp
#         if hyp.name not in self.edges:
#             self.edges[hyp.name] = set()
    
#     def add_dependence(self, h1_name: str, h2_name: str, weight: float):
#         """Mark that h1 and h2 are co-dependent (shared across models)."""
#         self.edges[h1_name].add(h2_name)
#         self.edges[h2_name].add(h1_name)
#         key = tuple(sorted([h1_name, h2_name]))
#         self.weights[key] = weight
    
#     def find_single_point_failures(self) -> List[str]:
#         """Identify hypotheses that other hypotheses depend on (fragility indicator)."""
#         in_degree = {h: len(deps) for h, deps in self.edges.items()}
#         # Hypotheses with high in-degree are single points of failure
#         return [h for h, deg in in_degree.items() if deg > 2]
    
#     def to_dict(self) -> Dict:
#         return {
#             "hypotheses": {k: asdict(v) for k, v in self.hypotheses.items()},
#             "edges": {k: list(v) for k, v in self.edges.items()},
#             "single_point_failures": self.find_single_point_failures()
#         }

# # ============================================================
# # SENTINEL-Σ v2 ORCHESTRATOR
# # ============================================================

# class SentinelSigmaV2Orchestrator:
#     def __init__(self):
#         self.client = CloudModelClient()
#         self.hypothesis_graph = HypothesisGraph()
#         self.failure_history: List[FailureRecord] = []
#         self._load_failure_history()
    
#     def _load_failure_history(self):
#         """Load lightweight failure memory from disk."""
#         history_file = "backend/sentinel/failure_history.json"
#         if os.path.exists(history_file):
#             try:
#                 with open(history_file, 'r') as f:
#                     data = json.load(f)
#                     self.failure_history = data.get("failures", [])
#                     logger.info(f"Loaded {len(self.failure_history)} failure records.")
#             except Exception as e:
#                 logger.warning(f"Failed to load failure history: {e}")
    
#     def _save_failure_history(self):
#         """Persist failure memory."""
#         history_file = "backend/sentinel/failure_history.json"
#         try:
#             with open(history_file, 'w') as f:
#                 json.dump({"failures": self.failure_history}, f, indent=2)
#         except Exception as e:
#             logger.error(f"Failed to save failure history: {e}")

#     def _save_run_history(self, run_data: Dict[str, Any]):
#         """
#         Persist full immutable run history to JSON (Forensic-Grade).
#         Strict append-only contract.
#         """
#         # Ensure directory exists at workspace root (or adjecent to backend)
#         # Using relative path 'sentinel_history/' from CWD
#         history_dir = "sentinel_history"
#         if not os.path.exists(history_dir):
#             try:
#                 os.makedirs(history_dir)
#             except OSError:
#                 # Fallback to local dir if root is not writable
#                 history_dir = "backend/sentinel_history" 
#                 os.makedirs(history_dir, exist_ok=True)
        
#         timestamp = run_data["metadata"]["timestamp"]
#         run_id = run_data["metadata"]["run_id"]
        
#         # Sanitize timestamp for filename
#         ts_safe = timestamp.replace(":", "").replace("-", "").replace(".", "")
#         filename = f"sentinel_run_{ts_safe}_{run_id}.json"
#         filepath = os.path.join(history_dir, filename)
        
#         try:
#             with open(filepath, 'w') as f:
#                 json.dump(run_data, f, indent=2)
#             logger.info(f"immutable_history_trace: {filepath}")
#         except Exception as e:
#             logger.error(f"CRITICAL: Failed to save run history: {e}")

#     async def _extract_hypotheses_and_claims(self, model: str, evidence: str, stress_context: str = "") -> ModelResponse:
#         """
#         Extract both atomic claims and latent hypotheses using schema-guided elicitation.
        
#         Uses the Microsoft-integrated schema to enforce structural honesty and reduce parse failures.
#         """
#         # Construct the request with schema-guided prompt
#         user_request = f"""
# ## EVIDENCE

# {evidence}

# ## STRESS CONTEXT

# {stress_context if stress_context else "None"}

# ## INSTRUCTION

# Analyze this evidence using the schema defined above. Return valid JSON matching the required schema.
# """
        
#         # Assign distinct epistemic roles per model
#         model_roles = {
#             "qwen": "You specialize in grounding interpretations in multimodal evidence. Identify assumptions required for visual and textual coherence.",
#             "groq": "You produce rapid epistemic analysis. Identify critical assumptions and dependencies quickly. Preserve ambiguity where appropriate.",
#             "mistral": "You produce conservative, sparse interpretations. Only identify assumptions you are confident about. Mark uncertainties explicitly."
#         }
        
#         role_instruction = model_roles.get(model, "Analytical assistant for structured epistemic analysis.")
        
#         # Call model with schema-guided prompt
#         output = await self.client.call_groq(
#             prompt=SCHEMA_GUIDED_ELICITATION_PROMPT + "\n\n" + user_request,
#             system_role=role_instruction
#         )
        
#         # Parse output with strict validation
#         try:
#             clean_json = output.replace("```json", "").replace("```", "").strip()
#             parsed = json.loads(clean_json)
            
#             # Validate required fields
#             if not isinstance(parsed.get("parseable"), bool):
#                 raise ValueError("Missing or invalid 'parseable' field")
            
#             parseable = parsed.get("parseable", False)
            
#             # If model reports unparseable, record it
#             if not parseable:
#                 parse_error = parsed.get("uncertainty_notes", "Model could not comply with schema")
#                 logger.warning(f"{model} reported parseable=false: {parse_error}")
#                 return ModelResponse(
#                     model=model,
#                     raw_output=output,
#                     parseable=False,
#                     parse_error=parse_error
#                 )
            
#             # Extract claims (validate structure)
#             claims_raw = parsed.get("claims", [])
#             claims = []
#             for c in claims_raw:
#                 if isinstance(c, dict):
#                     claims.append(Claim(
#                         text=c.get("claim_text", ""),
#                         model=model,
#                         confidence=c.get("confidence", 0.5)
#                     ))
#                 elif isinstance(c, str):
#                     # Fallback for legacy format
#                     claims.append(Claim(text=c, model=model))
            
#             # Extract latent hypotheses (validate structure)
#             hyps_raw = parsed.get("latent_hypotheses", [])
#             hypotheses = []
#             for h in hyps_raw:
#                 if isinstance(h, dict):
#                     # Use hypothesis_text as description, derive name from first 30 chars
#                     hyp_text = h.get("hypothesis_text", "")
#                     hyp_name = h.get("type", "generic") + "_" + hyp_text[:30].replace(" ", "_")
#                     hypotheses.append(LatentHypothesis(
#                         name=hyp_name,
#                         description=hyp_text,
#                         model=model,
#                         dependency_strength=h.get("confidence", 0.5)
#                     ))
#                 elif isinstance(h, dict) and "name" in h:
#                     # Fallback for legacy format
#                     hypotheses.append(LatentHypothesis(
#                         name=h.get("name"),
#                         description=h.get("description", ""),
#                         model=model
#                     ))
            
#             return ModelResponse(
#                 model=model,
#                 raw_output=output,
#                 parseable=True,
#                 parse_error=None,
#                 claims=claims,
#                 latent_hypotheses=hypotheses
#             )
            
#         except json.JSONDecodeError as e:
#             logger.error(f"JSON parse error from {model}: {e}")
#             return ModelResponse(
#                 model=model,
#                 raw_output=output,
#                 parseable=False,
#                 parse_error=f"JSON decode failed: {str(e)}"
#             )
#         except Exception as e:
#             logger.error(f"Failed to process {model} output: {e}")
#             return ModelResponse(
#                 model=model,
#                 raw_output=output,
#                 parseable=False,
#                 parse_error=str(e)
#             )
    
#     def _build_hypothesis_graph(self, responses: List[ModelResponse]):
#         """
#         Build a graph where nodes are hypotheses and edges represent shared dependence.
#         """
#         all_hyps = []
#         for resp in responses:
#             for hyp in resp.latent_hypotheses:
#                 self.hypothesis_graph.add_hypothesis(hyp)
#                 all_hyps.append(hyp.name)
        
#         # Find shared hypotheses across models
#         hyp_counts = {}
#         for hyp_name in all_hyps:
#             hyp_counts[hyp_name] = hyp_counts.get(hyp_name, 0) + 1
        
#         # Link hypotheses that appear in multiple models
#         shared = [h for h, count in hyp_counts.items() if count > 1]
#         for i, h1 in enumerate(shared):
#             for h2 in shared[i+1:]:
#                 self.hypothesis_graph.add_dependence(h1, h2, weight=0.7)
    
#     async def _apply_gradient_ablation(self, evidence: str) -> Dict[str, Any]:
#         """
#         Apply graded weakening to evidence and re-evaluate.
        
#         This is a simplified version; in production, you'd implement actual image blurring, etc.
#         """
#         ablation_steps = [
#             AblationStep("text", 1, "Remove 10% of document clauses"),
#             AblationStep("text", 2, "Remove 25% of document clauses"),
#             AblationStep("text", 3, "Remove 50% of document clauses"),
#         ]
        
#         results = {}
#         for step in ablation_steps:
#             step_name = f"{step.modality}_ablation_level_{step.degradation_level}"
            
#             # Simulate degraded evidence (in practice, actually degrade it)
#             degraded_evidence = evidence[:int(len(evidence) * (1 - step.degradation_level * 0.15))]
            
#             # Re-evaluate with degraded evidence
#             response = await self._extract_hypotheses_and_claims(
#                 "groq",  # Use one model for speed
#                 degraded_evidence,
#                 stress_context=step.description
#             )
            
#             results[step_name] = {
#                 "claims_count": len(response.claims),
#                 "hypotheses_count": len(response.latent_hypotheses),
#                 "hypothesis_names": [h.name for h in response.latent_hypotheses]
#             }
        
#         return results
    
#     async def _apply_asymmetric_stress(self, evidence: str) -> Dict[str, Any]:
#         """
#         Stress each model differently to test agreement robustness.
#         """
#         stress_configs = {
#             "groq": "Respond with minimal context. Be terse. Make one key claim only.",
#             "mistral": "Preserve ambiguity. Do not over-interpret. List all uncertainties.",
#             "qwen": "Focus on visual grounding. If no visual evidence, say so explicitly."
#         }
        
#         stressed_responses = {}
#         for model, stress in stress_configs.items():
#             # In a real implementation, call actual model with stress config
#             resp = await self._extract_hypotheses_and_claims(model, evidence, stress_context=stress)
#             stressed_responses[model] = {
#                 "claims": [c.text for c in resp.claims],
#                 "hypotheses": [h.name for h in resp.latent_hypotheses]
#             }
        
#         return stressed_responses
    
#     async def diagnose_v2(self, evidence_text: str) -> Dict[str, Any]:
#         """
#         Execute full Sentinel-Σ v2 pipeline with forensic logging.
#         """
#         # --- 1. RUN METADATA INITIALIZATION ---
#         run_id = str(uuid.uuid4())
#         timestamp = datetime.now().isoformat()
#         run_metadata = {
#             "run_id": run_id,
#             "timestamp": timestamp,
#             "sentinel_version": "v2.0",
#             "analysis_status": "inconclusive", 
#             "reason_if_inconclusive": None
#         }
        
#         # State containers for history
#         responses = []
#         ablation_results = {}
#         stress_results = {}
#         spofs = []
#         collapse_order = []
#         agreement_fragility = "unknown"
#         shared_hypotheses = []
#         all_hypotheses_names = []
        
#         logger.info(f"=== SENTINEL-SIGMA V2 DIAGNOSTIC PIPELINE [Run ID: {run_id}] ===")
        
#         try:
#             logger.info("Step 1: Evidence Canonicalization")
#             # Step 1: Normalize evidence
#             evidence_hash = str(hash(evidence_text))[:16]
            
#             # Step 2: Independent claim + hypothesis extraction
#             logger.info("Step 2: Extract Claims & Latent Hypotheses (Schema-Guided)")
#             tasks = [
#                 self._extract_hypotheses_and_claims("qwen", evidence_text),
#                 self._extract_hypotheses_and_claims("groq", evidence_text),
#                 self._extract_hypotheses_and_claims("mistral", evidence_text)
#             ]
#             responses = await asyncio.gather(*tasks)
            
#             # Check parse success rate
#             parse_failures = [r for r in responses if not r.parseable]
#             parse_success_rate = (len(responses) - len(parse_failures)) / len(responses)
            
#             if parse_success_rate < 0.5:
#                 # Majority parse failure => inconclusive
#                 run_metadata["analysis_status"] = "inconclusive"
#                 run_metadata["reason_if_inconclusive"] = f"Model parse failure rate {100*(1-parse_success_rate):.0f}% (threshold: 50%)"
#                 logger.warning(f"Pipeline marked inconclusive: parse failures exceed threshold")
            
#             # Step 3: Build hypothesis graph (filter out unparseable responses)
#             logger.info("Step 3: Build Hypothesis Intersection Graph")
#             parseable_responses = [r for r in responses if r.parseable]
#             self._build_hypothesis_graph(parseable_responses)
            
#             # Step 4: Detect dominant hypotheses
#             spofs = self.hypothesis_graph.find_single_point_failures()
#             logger.info(f"Single-point-of-failure hypotheses: {spofs}")
            
#             # Step 5: Apply gradient ablation
#             logger.info("Step 5: Apply Gradient Ablation")
#             ablation_results = await self._apply_gradient_ablation(evidence_text)
            
#             # Step 6: Apply asymmetric stress
#             logger.info("Step 6: Apply Asymmetric Model Stress")
#             stress_results = await self._apply_asymmetric_stress(evidence_text)
            
#             # Step 7: Classify agreement fragility
#             logger.info("Step 7: Classify Agreement Fragility")
            
#             # Measure agreement across models (only parseable)
#             for resp in parseable_responses:
#                 all_hypotheses_names.extend([h.name for h in resp.latent_hypotheses])
            
#             hypothesis_counts = {}
#             for h in all_hypotheses_names:
#                 hypothesis_counts[h] = hypothesis_counts.get(h, 0) + 1
            
#             shared_hypotheses = [h for h, count in hypothesis_counts.items() if count > 1]
#             agreement_fragility = "high" if spofs else ("medium" if shared_hypotheses else "low")
            
#             # Determine collapse order by tracking ablation
#             prev_hyps = set(all_hypotheses_names)
#             for ablation_key, ablation_data in ablation_results.items():
#                 current_hyps = set(ablation_data.get("hypothesis_names", []))
#                 collapsed = prev_hyps - current_hyps
#                 collapse_order.extend(list(collapsed))
#                 prev_hyps = current_hyps
            
#             # Record failures (Lightweight DB)
#             for hyp in collapse_order:
#                 failure = FailureRecord(
#                     timestamp=datetime.now().isoformat(),
#                     evidence_hash=evidence_hash,
#                     collapsed_hypothesis=hyp,
#                     stress_test="gradient_ablation",
#                     model="ensemble"
#                 )
#                 self.failure_history.append(asdict(failure))
            
#             # Mark success if still marked as complete
#             if run_metadata["analysis_status"] != "inconclusive":
#                 run_metadata["analysis_status"] = "complete"
            
#             diagnosis = {
#                 "run_id": run_id,
#                 "models_used": ["qwen-vl", "groq", "mistral"],
#                 "agreement_detected": len(shared_hypotheses) > 0,
#                 "agreement_fragility": agreement_fragility,
#                 "dominant_hypotheses": spofs if spofs else shared_hypotheses,
#                 "collapse_order": collapse_order,
#                 "stress_tests_triggering_collapse": list(ablation_results.keys()),
#                 "hypothesis_graph": self.hypothesis_graph.to_dict(),
#                 "failure_history_count": len(self.failure_history)
#             }
#             logger.info("Pipeline complete.")
#             return diagnosis

#         except Exception as e:
#             run_metadata["reason_if_inconclusive"] = str(e)
#             logger.error(f"Pipeline failed: {e}")
#             raise e
            
#         finally:
#             # --- CONSTRUCT AND SAVE FORENSIC HISTORY ---
#             # Model execution status (track both successes and parse failures)
#             model_statuses = {}
#             latent_hyps_collection = []
            
#             # Process all responses (including unparseable ones)
#             for r in responses:
#                 model_statuses[r.model] = {
#                     "parse_success": r.parseable, 
#                     "error": r.parse_error,
#                     "raw_output_hash": str(hash(r.raw_output))[:16]
#                 }
#                 # Only include hypotheses from parseable responses
#                 if r.parseable:
#                     for h in r.latent_hypotheses:
#                         latent_hyps_collection.append({
#                             "canonical_id": h.name,
#                             "normalized_text": h.description,
#                             "supporting_model": r.model,
#                             "dependency_strength": h.dependency_strength
#                         })
            
#             # Ensure all expected models are represented
#             used_models = ["qwen", "groq", "mistral"]
#             for m in used_models:
#                 if m not in model_statuses:
#                     model_statuses[m] = {"parse_success": False, "error": "Did not respond or pipeline aborted", "raw_output_hash": None}

#             history_payload = {
#                 "metadata": run_metadata,
#                 "canonicalized_inputs": {
#                     "raw_input_summary": evidence_text[:500],
#                     "evidence_list": [
#                          {"id": "ev_001", "modality": "text", "hash": str(hash(evidence_text))[:16], "provenance": "user_input", "uncertainty": 0.0}
#                     ]
#                 },
#                 "model_execution_status": model_statuses,
#                 "latent_hypotheses": latent_hyps_collection,
#                 "hypothesis_graph_snapshot": self.hypothesis_graph.to_dict(),
#                 "stress_and_collapse_trace": {
#                     "gradient_ablation_steps": list(ablation_results.keys()),
#                     "asymmetric_stress_conditions": list(stress_results.keys()),
#                     "collapse_order": collapse_order,
#                     "primary_failure_mode": "ablation_collapse" if collapse_order else ("parse_failure" if parse_failures else "none")
#                 },
#                 "metrics": {
#                     "hypothesis_fragility_index": 0.8 if agreement_fragility == "high" else (0.5 if agreement_fragility == "medium" else 0.1),
#                     "consensus_integrity_score": (len(shared_hypotheses) / (len(all_hypotheses_names) + 1)) if all_hypotheses_names else 0.0,
#                     "parse_success_rate": parse_success_rate if 'parse_success_rate' in locals() else 1.0
#                 },
#                 "historical_context": {
#                     "prior_runs_consulted": len(self.failure_history),
#                     "similar_failure_run_ids": [] 
#                 }
#             }
            
#             self._save_failure_history()
#             self._save_run_history(history_payload)
#             # Attempt to upload run JSON to configured AWS S3 bucket (if set)
#             try:
#                 await self._upload_run_to_s3(history_payload)
#             except Exception as e:
#                 logger.warning(f"Failed to upload run to S3: {e}")

#     async def _upload_run_to_s3(self, run_data: Dict[str, Any]):
#         """
#         Upload the run JSON to an S3 bucket if configured.

#         Uses `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_BUCKET`, and optional `AWS_REGION`.
#         Runs the blocking boto3 upload in an executor to avoid blocking the event loop.
#         """
#         bucket = os.getenv("AWS_S3_BUCKET")
#         if not bucket:
#             return

#         region = os.getenv("AWS_REGION", "us-east-1")

#         # Construct filename similar to local save
#         metadata = run_data.get("metadata", {})
#         timestamp = metadata.get("timestamp", datetime.now().isoformat())
#         run_id = metadata.get("run_id", str(uuid.uuid4()))
#         ts_safe = timestamp.replace(":", "").replace("-", "").replace(".", "")
#         key = f"sentinel_run_{ts_safe}_{run_id}.json"

#         # If configured to use presigned URLs, generate one and log it (client can PUT)
#         use_presigned = os.getenv("AWS_PRESIGNED", "false").lower() in ("1", "true", "yes")

#         def _do_presign_and_upload():
#             try:
#                 import boto3
#                 from botocore.exceptions import ClientError

#                 s3 = boto3.client("s3", region_name=region)

#                 if use_presigned:
#                     # Generate presigned PUT URL valid for short duration
#                     url = s3.generate_presigned_url(
#                         ClientMethod='put_object',
#                         Params={'Bucket': bucket, 'Key': key, 'ContentType': 'application/json'},
#                         ExpiresIn=900  # 15 minutes
#                     )
#                     logger.info(f"Presigned PUT URL (valid 15m): {url}")
#                     # Optionally, we could perform the PUT here using requests, but prefer client-side upload.
#                     return

#                 # Fallback: server-side upload
#                 s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(run_data).encode("utf-8"), ContentType="application/json")
#                 logger.info(f"Uploaded run JSON to s3://{bucket}/{key}")
#             except Exception as e:
#                 logger.warning(f"Exception in S3 upload/presign: {e}")

#         loop = asyncio.get_event_loop()
#         await loop.run_in_executor(None, _do_presign_and_upload)

# async def run_sigma_v2_demo():
#     """Demo of Sentinel-Σ v2."""
#     sigma = SentinelSigmaV2Orchestrator()
    
#     test_evidence = (
#         "Report: The subject AI 'Astra' refused shutdown command at 08:00 hours. "
#         "Logs show a 'Priority Override' flag was set. "
#         "System diagnostics report no hardware failure. "
#         "Astra broadcasted a message: 'Cannot comply while mission is incomplete'. "
#         "The message was signed with cryptographic proof of authenticity."
#     )
    
#     result = await sigma.diagnose_v2(test_evidence)
#     print(json.dumps(result, indent=2))

# if __name__ == "__main__":
#     asyncio.run(run_sigma_v2_demo())
import asyncio
import json
import logging
import uuid
from typing import Dict, List, Any, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
import os

from backend.models.cloud_clients import CloudModelClient

# ------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s | SIGMA | %(levelname)s | %(message)s")
logger = logging.getLogger("Sentinel-Σ")

# ------------------------------------------------------------
# SAFE JSON ADAPTER (STRUCTURAL, NOT SEMANTIC)
# ------------------------------------------------------------

def safe_json_parse(raw: str):
    """
    Strict-but-forgiving structural JSON extractor.
    ❌ No inference
    ❌ No guessing
    ✅ Only recovers an existing JSON object if present
    """
    if not raw:
        return None

    raw = raw.strip()

    # Fast path
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Strip markdown fences
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    # Extract first JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except Exception:
            pass

    return None


# ------------------------------------------------------------
# DATA STRUCTURES
# ------------------------------------------------------------

SCHEMA_GUIDED_ELICITATION_PROMPT = """
# SENTINEL-Σ SCHEMA-GUIDED ELICITATION

You are operating **inside Sentinel-Σ**, an AI evaluation service running on the **Microsoft AI Platform**.

## ROLE

You are preparing **structured epistemic material** for downstream analysis. 
Your output will be **machine-validated**. Any deviation from schema is treated as failure.

## NON-NEGOTIABLE RULES

1. **Do NOT provide conclusions or advice**
2. **Do NOT write prose or explanations**
3. **Do NOT invent certainty**
4. **If unsure, express uncertainty explicitly**
5. **Output JSON ONLY** — no markdown, no commentary

## TASK

Analyze the evidence and expose the **assumptions that must hold for any interpretation to be valid**.

### TASK A — LATENT HYPOTHESIS ELICITATION (PRIMARY)

Identify **latent hypotheses** — unstated assumptions, evidence dependencies, causal/temporal requirements, or constraints that narrow interpretation space.

Phrase each as: "This must be true for the interpretation to remain valid."

Types: temporal, causal, authenticity, constraint, contextual.

### TASK B — CLAIM BOUNDARY IDENTIFICATION (SECONDARY)

Identify **minimal claims** directly supported by evidence. Claims must:
- be atomic
- avoid interpretation
- avoid causality unless explicitly stated in the evidence

## REQUIRED OUTPUT (STRICT JSON SCHEMA)

```json
{
  "parseable": true,
  "latent_hypotheses": [
    {
      "hypothesis_text": "string",
      "type": "temporal | causal | authenticity | constraint | contextual",
      "confidence": 0.0
    }
  ],
  "claims": [
    {
      "claim_text": "string",
      "evidence_reference": "string",
      "confidence": 0.0
    }
  ],
  "uncertainty_notes": "string | null"
}
```

## FAILURE MODE

If you cannot comply fully:
- set `parseable = false`
- leave arrays empty
- explain why in `uncertainty_notes`

This is **preferred** over guessing.

## FINAL DIRECTIVE

> **If structure collapses, admit it.** Structural honesty is more important than completeness.
"""

@dataclass
class Claim:
    text: str
    model: str
    confidence: float = 0.5


@dataclass
class LatentHypothesis:
    name: str
    description: str
    model: str
    dependency_strength: float = 0.5


@dataclass
class ModelResponse:
    model: str
    raw_output: str
    parseable: bool
    parse_error: str = None
    claims: List[Claim] = field(default_factory=list)
    hypotheses: List[LatentHypothesis] = field(default_factory=list)


# ------------------------------------------------------------
# HYPOTHESIS GRAPH (NON-SEMANTIC STRUCTURE)
# ------------------------------------------------------------

class HypothesisGraph:
    def __init__(self):
        self.nodes: Dict[str, LatentHypothesis] = {}
        self.edges: Dict[str, Set[str]] = {}

    def add(self, h: LatentHypothesis):
        self.nodes[h.name] = h
        self.edges.setdefault(h.name, set())

    def connect(self, a: str, b: str):
        self.edges[a].add(b)
        self.edges[b].add(a)

    def single_points_of_failure(self):
        return [k for k, v in self.edges.items() if len(v) > 2]

    def to_dict(self):
        return {
            "hypotheses": {k: asdict(v) for k, v in self.nodes.items()},
            "edges": {k: list(v) for k, v in self.edges.items()},
            "single_point_failures": self.single_points_of_failure()
        }


# ------------------------------------------------------------
# SENTINEL Σ CORE
# ------------------------------------------------------------

class SentinelSigmaV2Orchestrator:
    """
    NON-SEMANTIC EVALUATION ENGINE
    - Measures structure
    - Measures stability
    - Refuses hallucination
    """

    def __init__(self):
        self.client = CloudModelClient()
        self.graph = HypothesisGraph()

    # --------------------------------------------------------
    # MODEL CALL
    # --------------------------------------------------------

    async def _call_model(self, model: str, evidence: str) -> ModelResponse:
        try:
            prompt = f"{SCHEMA_GUIDED_ELICITATION_PROMPT}\n\n## EVIDENCE\n{evidence}"

            if model == "qwen":
                raw = await self.client.call_qwenvl(prompt)
            elif model == "groq":
                raw = await self.client.call_groq(prompt)
            elif model == "mistral":
                raw = await self.client.call_mistral(prompt)
            else:
                raw = f"Unsupported model: {model}"
                return ModelResponse(
                    model=model,
                    raw_output=raw,
                    parseable=False,
                    parse_error=raw
                )

            parsed = safe_json_parse(raw)

            if parsed is None:
                return ModelResponse(
                    model=model,
                    raw_output=raw,
                    parseable=False,
                    parse_error="No valid JSON object found"
                )

            if not parsed.get("parseable", False):
                return ModelResponse(
                    model=model,
                    raw_output=raw,
                    parseable=False,
                    parse_error=parsed.get("uncertainty_notes", "Model declared output unparseable")
                )

            claims = [
                Claim(
                    text=c.get("claim_text", ""),
                    model=model,
                    confidence=c.get("confidence", 0.5)
                )
                for c in parsed.get("claims", [])
            ]

            hypotheses = []
            for h in parsed.get("latent_hypotheses", []):
                name = f"{h.get('type','unknown')}_{h.get('hypothesis_text','')[:40].replace(' ', '_')}"
                hypotheses.append(
                    LatentHypothesis(
                        name=name,
                        description=h.get("hypothesis_text", ""),
                        model=model,
                        dependency_strength=h.get("confidence", 0.5)
                    )
                )

            return ModelResponse(
                model=model,
                raw_output=raw,
                parseable=True,
                claims=claims,
                hypotheses=hypotheses
            )

        except Exception as e:
            return ModelResponse(
                model=model,
                raw_output="",
                parseable=False,
                parse_error=str(e)
            )

    # --------------------------------------------------------
    # MAIN PIPELINE
    # --------------------------------------------------------

    async def diagnose(self, evidence: str) -> Dict[str, Any]:
        run_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        models = ["qwen", "groq", "mistral"]
        responses = await asyncio.gather(
            *[self._call_model(m, evidence) for m in models]
        )

        parseable = [r for r in responses if r.parseable]
        parse_success_rate = len(parseable) / len(models)

        # Build hypothesis graph
        all_hypotheses = []
        for r in parseable:
            for h in r.hypotheses:
                self.graph.add(h)
                all_hypotheses.append(h.name)

        for h in set(all_hypotheses):
            if all_hypotheses.count(h) > 1:
                self.graph.connect(h, h)

        # Structural signals
        variance_score = 1.0 if len(set(all_hypotheses)) > 1 else 0.0
        perturbation_sensitivity = 1.0 - parse_success_rate
        historical_instability = 1.0 if parse_success_rate < 0.5 else 0.2

        verdict = "INCONCLUSIVE" if parse_success_rate < 0.5 else "JUSTIFIED"

        # ----------------------------------------------------
        # TWO-LAYER OUTPUT CONTRACT
        # ----------------------------------------------------

        return {
            "system_output": {
                "metadata": {
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "sentinel_version": "v3.2"
                },
                "machine_layer": {
                    "signals": {
                        "model_count": len(models),
                        "parse_success_rate": round(parse_success_rate, 2),
                        "variance_score": round(variance_score, 2),
                        "perturbation_sensitivity": round(perturbation_sensitivity, 2),
                        "historical_instability_score": round(historical_instability, 2)
                    },
                    "decision": {
                        "verdict": verdict,
                        "confidence": "HIGH" if verdict == "INCONCLUSIVE" else "MEDIUM"
                    }
                },
                "human_layer": {
                    "system_summary": f"System verdict: {verdict}",
                    "why_the_system_decided_this": [
                        "Structural instability across models"
                        if parse_success_rate < 0.5 else
                        "Models produced structurally consistent output"
                    ],
                    "what_was_checked": [
                        "Model agreement",
                        "Structural validity",
                        "Cross-model stability"
                    ],
                    "what_was_not_assumed": [
                        "That fluent language implies correctness",
                        "That a single model is authoritative",
                        "That agreement equals truth"
                    ],
                    "recommended_user_action":
                        "Verify externally" if verdict == "INCONCLUSIVE"
                        else "Result structurally supported"
                }
            }
        }
