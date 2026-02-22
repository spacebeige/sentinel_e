# SENTINEL-E 3.X REWRITE PROGRESS TRACKER

## COMPLETED FILES (1-9):
1. `/Users/ashwinagarkhed/sentinel_e/backend/core/mode_config.py` - NEW: Mode(STANDARD/RESEARCH), SubMode, DebateRole, ModeConfig
2. `/Users/ashwinagarkhed/sentinel_e/backend/core/confidence_engine.py` - NEW: ConfidenceComponents, ConfidenceTrace, ConfidenceEngine
3. `/Users/ashwinagarkhed/sentinel_e/backend/core/stress_engine.py` - NEW: StressEngine with 4 vectors
4. `/Users/ashwinagarkhed/sentinel_e/backend/core/debate_orchestrator.py` - REWRITTEN: Groq/Mistral/Qwen names, roles, judge scoring, disagreement
5. `/Users/ashwinagarkhed/sentinel_e/backend/core/evidence_engine.py` - UPGRADED: claim extraction, overlap matrix, evidence strength, traceability
6. `/Users/ashwinagarkhed/sentinel_e/backend/core/behavioral_analytics.py` - UPGRADED: authority_mimicry, overconfidence_inflation
7. `/Users/ashwinagarkhed/sentinel_e/backend/core/session_intelligence.py` - UPGRADED: cognitive_drift, topic_stability, confidence_volatility, debate_history_graph, model_reliability_scores
8. `/Users/ashwinagarkhed/sentinel_e/backend/core/knowledge_learner.py` - UPGRADED: get_model_weight() with gradual decay, get_all_model_weights()

## REMAINING:
9. omega_kernel.py - COMPLETE REWRITE (replace OmegaConfig with ModeConfig, add ConfidenceEngine/StressEngine, add _process_stress, update all mode routing)
10. omega_formatter.py - ADD format_stress method; update "experimental" to "research" in output labels
11. schemas.py + main.py - Update for new modes/role_map/stress
12. Validate build

## OMEGA_KERNEL.PY REWRITE PLAN:
- Replace imports: OmegaConfig → ModeConfig from mode_config, add ConfidenceEngine, StressEngine
- Replace VALID_MODES/OMEGA_MODES/VALID_SUB_MODES with imports from mode_config
- Replace OmegaConfig dataclass with usage of ModeConfig
- __init__: Add self.confidence_engine = ConfidenceEngine(), self.stress_engine (lazy)
- process(): Read ModeConfig, route: STANDARD→_process_standard, RESEARCH→sub_mode
- _process_standard: Use ConfidenceEngine.compute_standard(), single answer + confidence + risk line
- _process_debate: Pass config.role_map to debate.run_debate(), use ConfidenceEngine.compute_debate_consensus()
- _process_glass: Interpretive with confidence_trace, behavioral risk, kill_override diagnostic
- _process_evidence: Use evidence.run_full_evidence_analysis()
- _process_stress: NEW - Use StressEngine
- Knowledge integration: confidence *= knowledge_learner.get_model_weight(model_name)
- Keep all helper methods

## KEY PATTERNS:
- OmegaCognitiveKernel.__init__(sigma_orchestrator): self.sigma = sigma_orchestrator
- CloudModelClient: self.sigma.client has call_groq, call_mistral, call_qwenvl
- SigmaV4Config(text, mode, enable_shadow, rounds, chat_id, history) → self.sigma.run_sentinel(sigma_config)
- llm_result.data.get("priority_answer", "") for text output
- self.boundary.evaluate(text, context_observations, session_data) returns dict with severity_score, risk_level, explanation
- self.boundary.evaluate_debate_boundaries(model_positions, agreements, disagreements)
- self.boundary.get_trend() returns dict with trend
- self.session = SessionIntelligence(), self.reasoning = MultiPassReasoningEngine()
- self.session.state.behavioral_risk_profile = profile_dict
- self.session.update_behavioral_risk(profile_dict) - NEW 2.0 method
- self.session.record_debate_round(summary_dict) - NEW 2.0 method
- self.formatter methods: format_standard, format_debate, format_glass, format_kill, format_evidence, format_coding_task
- Need to ADD format_stress to formatter

## OMEGA_FORMATTER KEY:
- format_standard(data) → markdown string
- format_debate(data) → markdown string with model-by-model breakdown
- format_glass(data) → markdown string with analysis sections
- format_kill(data) → markdown string diagnostic
- format_evidence(data) → markdown string with sources
- format_coding_task(data) → markdown string

## SCHEMAS:
- SentinelRequest at backend/sentinel/schemas.py: text, mode, sub_mode, enable_shadow, rounds, chat_id
- Need to add: role_map optional dict, kill_override bool

## MAIN.PY INTEGRATION:
- /api/sentinel endpoint: creates OmegaConfig → omega_kernel.process()
- omega_sessions dict[str, OmegaCognitiveKernel]
- Knowledge learner at app.state.knowledge_learner
- Need to: pass knowledge_learner to kernel, update OmegaConfig→ModeConfig
