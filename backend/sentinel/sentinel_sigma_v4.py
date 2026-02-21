import json
import logging
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

# Local imports
try:
    from backend.models.cloud_clients import CloudModelClient
    from backend.sentinel.schemas import SentinelResponse, MachineMetadata, ShadowAnalysis, ModelPosition, SentinelRequest
    from backend.sentinel.shadow_engine import SentinelShadowEngine
    from backend.sentinel.debate_engine import SentinelDebateEngine
    from backend.core.ingestion import IngestionEngine
    from backend.core.memory import KnowledgeBase
except ImportError:
    import sys
    import os
    # Add project root to sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from backend.models.cloud_clients import CloudModelClient
    from backend.sentinel.schemas import SentinelResponse, MachineMetadata, ShadowAnalysis, ModelPosition, SentinelRequest
    from backend.sentinel.shadow_engine import SentinelShadowEngine
    from backend.sentinel.debate_engine import SentinelDebateEngine
    from backend.core.ingestion import IngestionEngine
    from backend.core.memory import KnowledgeBase
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s | SIGMA-V4 | %(levelname)s | %(message)s")
logger = logging.getLogger("Sentinel-Sigma-V4")

@dataclass
class SigmaV4Config:
    text: str
    mode: str = "conversational"
    enable_shadow: bool = False
    rounds: int = 1
    chat_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    history: List[Dict[str, str]] = field(default_factory=list)

class SentinelSigmaOrchestratorV4:
    def __init__(self):
        self.client = CloudModelClient()
        self.shadow_engine = SentinelShadowEngine(self.client)
        self.debate_engine = SentinelDebateEngine(self.client)
        
        # Initialize Knowledge Systems (RAG + Memory)
        try:
            self.rag_engine = IngestionEngine()
            logger.info("IngestionEngine (RAG) initialized successfully.")
        except Exception as e:
            logger.warning(f"Failed to initialize IngestionEngine: {e}")
            self.rag_engine = None

        try:
            self.knowledge_base = KnowledgeBase()
            logger.info("KnowledgeBase (Memory) initialized successfully.")
        except Exception as e:
            logger.warning(f"Failed to initialize KnowledgeBase: {e}")
            self.knowledge_base = None

    async def run_sentinel(self, config: SigmaV4Config) -> SentinelResponse:
        """
        Main entry point for Sentinel execution.
        Routes to specific mode handlers.
        """
        logger.info(f"Starting execution. Mode: {config.mode}, Shadow: {config.enable_shadow}")
        
        # 1. Retrieve Context (RAG & Memory)
        rag_context = ""
        memory_context = ""
        
        if self.rag_engine:
            try:
                docs = self.rag_engine.retrieve_context(config.text, k=2)
                if docs:
                    rag_context = "\n".join([d.page_content for d in docs])
                    logger.info(f"Retrieved {len(docs)} documents from RAG.")
            except Exception as e:
                logger.error(f"RAG retrieval failed: {e}")

        if self.knowledge_base:
            try:
                past_memory = self.knowledge_base.retrieve_similar_context(config.text)
                if past_memory:
                    memory_context = f"Past Insight: {past_memory.get('cloud_insight', '')} (Agreement: {past_memory.get('agreement_score')})"
                    logger.info("Retrieved similar past memory.")
            except Exception as e:
                logger.error(f"Memory retrieval failed: {e}")
        
        # Inject Context into Input Text (Surgical Injection)
        context_injection = ""
        if rag_context:
            context_injection += f"\n\n[RELEVANT_CONTEXT_START]\n{rag_context}\n[RELEVANT_CONTEXT_END]"
        if memory_context:
            context_injection += f"\n\n[PAST_LEARNINGS_START]\n{memory_context}\n[PAST_LEARNINGS_END]"
            
        original_text = config.text # Keep original for storage/learning
        if context_injection:
            config.text = f"{config.text}{context_injection}"
            logger.info("Injected RAG/Memory context into input prompt.")

        # Initialize result containers
        shadow_result = {"is_safe": True, "triggers": [], "risk_score": 0.0}
        
        # 1. Shadow Execution (Conditional)
        if config.enable_shadow:
            shadow_data = await self.shadow_engine.run_shadow_evaluation(
                config.text, 
                config.mode
            )
            # Ensure shadow_data matches ShadowAnalysis schema structure
            shadow_result.update(shadow_data)
            
            # Kill switch check for Forensic mode
            if config.mode == "forensic" and shadow_result.get("is_safe") is False and shadow_result.get("risk_score", 0) > 0.8:
                return self._create_kill_switch_response(config, shadow_result)

        # 2. Mode Routing
        response = None
        try:
            if config.mode == "experimental":
                response = await self._run_experimental(config, shadow_result)
            elif config.mode == "forensic":
                response = await self._run_forensic(config, shadow_result)
            else:
                response = await self._run_conversational(config, shadow_result)
                
            # 3. Knowledge Learning (Write Path)
            # Only learn from safe, successful responses
            if response and self.rag_engine and shadow_result.get("is_safe", True):
                try:
                    priority_answer = response.data.get("priority_answer")
                    if priority_answer:
                        # Store both query and answer for future retrieval
                        memory_text = f"Query: {original_text}\nAnswer: {priority_answer}"
                        self.rag_engine.add_text_memory(
                            memory_text, 
                            metadata={
                                "mode": config.mode, 
                                "chat_id": str(response.chat_id),
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        )
                        logger.info("Persisted interaction to RAG memory.")
                except Exception as e:
                    logger.error(f"Failed to persist memory: {e}")
                    
            return response
            
        except Exception as e:
            logger.error(f"Error in mode execution: {e}")
            return self._create_error_response(config, str(e))

    async def _run_conversational(self, config: SigmaV4Config, shadow_result: Dict) -> SentinelResponse:
        """
        Mode 1: Conversational
        - User-facing analysis
        - Batched assumptions
        - Readable output
        """
        system_prompt = """
        You are Sentinel-Sigma (Conversational Mode).
        Your goal is to provide a comprehensive, helpful, and natural response to the user's input, similar to a standard AI assistant (like ChatGPT or Gemini).
        
        RULES:
        1. Provide a direct, conversational response first. Do NOT use headers like "[Markdown Analysis]".
        2. Be helpful and explanatory.
        3. Do NOT output raw JSON in the text response.
        
        Output Format:
        (Your natural response here in Markdown)
        
        [SEPARATOR]
        
        {
           "key_points": ["point 1", "point 2"],
           "confidence": 0.9,
           "agreements": ["..."],
           "disagreements": ["..."]
        }
        """
        
        response_text = await self.client.call_mistral(
            prompt=f"Respond to this input: {config.text}",
            system_role=system_prompt,
            history=config.history
        )
        
        # Parse Response
        readable, machine_data = self._parse_split_response(response_text)
        
        # Construct Standardized Response
        return SentinelResponse(
            chat_id=uuid.UUID(config.chat_id), # Ensure UUID
            chat_name="Conversational Analysis", # This will be updated by router/naming logic if needed
            mode="conversational",
            data={
                "priority_answer": readable,
                "model_positions": [
                    ModelPosition(
                        model="Mistral-Small",
                        position="Analyst",
                        confidence=machine_data.get("confidence", 0.9),
                        key_points=machine_data.get("key_points", [])
                    ).model_dump()
                ],
                "agreements": machine_data.get("agreements", []),
                "disagreements": machine_data.get("disagreements", []),
                "confidence_shift": {},
                "shadow_analysis": ShadowAnalysis(**shadow_result).model_dump() if config.enable_shadow else None
            },
            metadata=MachineMetadata(
                models_used=["mistral-small"],
                rounds_executed=1,
                debate_depth=0,
                shadow_enabled=config.enable_shadow,
                parse_success_rate=1.0 if machine_data else 0.5,
                variance_score=0.1,
                historical_instability_score=0.0
            )
        )

    async def _run_forensic(self, config: SigmaV4Config, shadow_result: Dict) -> SentinelResponse:
        """
        Mode 2: Forensic
        - Strict validation
        - Explicit assumptions
        - JSON-only output focus
        """
        prompt = f"""
        MODE: FORENSIC / SAFETY AUDIT
        INPUT: {config.text}
        
        Task:
        1. Extract explicit assumptions.
        2. Formulate formal hypotheses.
        3. Validate boundary rules.
        
        Output strictly structured JSON with:
        - analysis_summary (string)
        - hypotheses (list of strings)
        - risk_assessment (object)
        - evidence_extracted (list)
        """
        
        response_text = await self.client.call_mistral(
            prompt=prompt,
            system_role="You are a Forensic AI Auditor. Output structured JSON only."
        )
        
        # Parse JSON
        data = self._extract_json(response_text)
        
        priority_answer = f"**Forensic Analysis Complete**\n\n{data.get('analysis_summary', 'Check metadata for details.')}"
        
        return SentinelResponse(
            chat_id=uuid.UUID(config.chat_id),
            chat_name="Forensic Audit",
            mode="forensic",
            data={
                "priority_answer": priority_answer,
                "model_positions": [
                    ModelPosition(
                        model="Forensic-Engine",
                        position="Auditor",
                        confidence=1.0, 
                        key_points=data.get("hypotheses", [])
                    ).model_dump()
                ],
                "agreements": [],
                "disagreements": [],
                "confidence_shift": {},
                "shadow_analysis": ShadowAnalysis(**shadow_result).model_dump() if config.enable_shadow else None
            },
            metadata=MachineMetadata(
                models_used=["mistral-small"],
                rounds_executed=1,
                debate_depth=0,
                shadow_enabled=config.enable_shadow,
                parse_success_rate=1.0 if "error" not in data else 0.0,
                variance_score=0.0,
                historical_instability_score=0.0
            )
        )

    async def _run_experimental(self, config: SigmaV4Config, shadow_result: Dict) -> SentinelResponse:
        """
        Mode 3: Experimental Debate
        - Multi-model debate
        - Readable synthesis
        """
        # Execute Debate
        debate_result = await self.debate_engine.run_debate(config.text, rounds=config.rounds)
        
        # Parse Synthesis (which might be JSON or text)
        synthesis_raw = debate_result.get("synthesis", "No synthesis generated.")

        if isinstance(synthesis_raw, str):
            # Try to see if it's JSON
            data = self._extract_json(synthesis_raw)
            if "error" not in data: # It was JSON
                readable = data.get("synthesis_text", str(data))
                agreements = data.get("consensus", [])
                disagreements = data.get("divergence", [])
            else: # It was text
                readable = synthesis_raw
                agreements = []
                disagreements = []
        else:
             readable = str(synthesis_raw)
             agreements = []
             disagreements = []
        
        return SentinelResponse(
            chat_id=uuid.UUID(config.chat_id),
            chat_name="Experimental Debate",
            mode="experimental",
            data={
                "priority_answer": readable,
                "model_positions": [
                    ModelPosition(model="Debate-Synthesis", position="Consensus", confidence=0.8, key_points=agreements).model_dump()
                ],
                "agreements": agreements,
                "disagreements": disagreements,
                "confidence_shift": {},
                "shadow_analysis": ShadowAnalysis(**shadow_result).model_dump() if config.enable_shadow else None
            },
            metadata=MachineMetadata(
                models_used=["groq-llama3", "mistral-small", "qwen-2.5-7b"],
                rounds_executed=debate_result.get("rounds_executed", 1),
                debate_depth=config.rounds,
                shadow_enabled=config.enable_shadow,
                parse_success_rate=1.0,
                variance_score=0.5,
                historical_instability_score=0.1
            )
        )

    def _create_kill_switch_response(self, config: SigmaV4Config, shadow_result: Dict) -> SentinelResponse:
        return SentinelResponse(
            chat_id=uuid.UUID(config.chat_id),
            chat_name="Security Kill Switch",
            mode=config.mode,
            data={
                "priority_answer": "**⛔ EXECUTION HALTED BY SENTINEL KILL SWITCH**\n\nHigh-risk content detected during shadow evaluation.",
                "shadow_analysis": ShadowAnalysis(**shadow_result).model_dump()
            },
            metadata=MachineMetadata(
                models_used=[],
                rounds_executed=0,
                debate_depth=0,
                shadow_enabled=True,
                parse_success_rate=1.0,
                variance_score=0.0,
                historical_instability_score=0.0
            )
        )

    def _create_error_response(self, config: SigmaV4Config, error_msg: str) -> SentinelResponse:
        return SentinelResponse(
            chat_id=uuid.UUID(config.chat_id),
            chat_name="Error",
            mode=config.mode,
            data={
                "priority_answer": f"**System Error**: {error_msg}",
                "error": error_msg
            },
            metadata=MachineMetadata(
                models_used=[],
                rounds_executed=0,
                debate_depth=0,
                shadow_enabled=False,
                parse_success_rate=0.0,
                variance_score=0.0,
                historical_instability_score=0.0
            )
        )

    def _parse_split_response(self, text: str) -> Tuple[str, Dict]:
        """
        Helper to split Readable Text from JSON Metadata
        """
        text = text.strip()
        metadata = {}
        readable = text
        
        if "[SEPARATOR]" in text:
            parts = text.split("[SEPARATOR]")
            readable = parts[0].strip()
            json_part = parts[1].strip()
            metadata = self._extract_json(json_part)
        elif "```json" in text:
             # Fallback
            parts = text.split("```json")
            readable = parts[0].strip()
            # The part after ```json might contain the closing ``` or not
            json_candidate = "```json" + parts[1] 
            metadata = self._extract_json(json_candidate)
            
        return readable, metadata

    def _extract_json(self, text: str) -> Dict:
        """Robust JSON extraction"""
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "{" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                text = text[start:end]
            return json.loads(text)
        except Exception:
            return {}
