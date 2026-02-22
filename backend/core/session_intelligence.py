"""
Session Intelligence Model — Sentinel-E Omega Cognitive Kernel

Maintains persistent session state across interactions:
- Chat naming
- Domain inference
- User expertise scoring (0–1)
- Recurring error pattern detection
- Boundary history tracking
- Disagreement score evolution
- Fragility index computation
- Session confidence tracking
"""

import re
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


# ============================================================
# DOMAIN TAXONOMY
# ============================================================

DOMAIN_KEYWORDS = {
    "software_engineering": [
        "code", "api", "debug", "deploy", "architecture", "backend", "frontend",
        "database", "server", "docker", "kubernetes", "microservice", "git",
        "python", "javascript", "typescript", "rust", "go", "java", "sql",
        "function", "class", "module", "test", "ci/cd", "pipeline", "aws",
        "azure", "gcp", "cloud", "devops", "infrastructure", "terraform",
    ],
    "machine_learning": [
        "model", "training", "inference", "neural", "transformer", "llm",
        "embedding", "vector", "gradient", "loss", "accuracy", "precision",
        "recall", "dataset", "feature", "classification", "regression",
        "clustering", "reinforcement", "fine-tune", "rag", "attention",
        "tokenizer", "bert", "gpt", "diffusion", "gan",
    ],
    "cybersecurity": [
        "vulnerability", "exploit", "threat", "firewall", "encryption",
        "authentication", "authorization", "zero-day", "malware", "phishing",
        "pentest", "audit", "compliance", "soc", "siem", "incident",
    ],
    "business_strategy": [
        "revenue", "market", "competition", "stakeholder", "roi", "kpi",
        "growth", "acquisition", "valuation", "p&l", "margin", "customer",
        "churn", "retention", "product-market", "ecosystem", "moat",
    ],
    "scientific_research": [
        "hypothesis", "experiment", "peer review", "methodology", "sample size",
        "statistical significance", "control group", "variable", "replication",
        "publication", "citation", "meta-analysis", "longitudinal",
    ],
    "systems_thinking": [
        "feedback loop", "emergence", "complexity", "nonlinear", "adaptive",
        "resilience", "bottleneck", "constraint", "throughput", "latency",
        "scalability", "fault tolerance", "distributed", "consensus",
    ],
}

EXPERTISE_SIGNALS = {
    "high": [
        "idempotent", "invariant", "heuristic", "amortized", "asymptotic",
        "monotonic", "stochastic", "deterministic", "eigenvalue", "latent",
        "bottleneck", "linearizable", "consensus protocol", "CAP theorem",
        "byzantine", "eventual consistency", "backpressure", "circuit breaker",
    ],
    "medium": [
        "algorithm", "complexity", "optimization", "refactor", "architecture",
        "concurrency", "async", "middleware", "schema", "migration",
        "normalization", "index", "cache", "queue", "pub/sub",
    ],
    "low": [
        "how do i", "what is", "explain", "help me", "i don't know",
        "simple", "basic", "beginner", "tutorial", "example",
    ],
}


@dataclass
class ErrorPattern:
    """Detected user logical error."""
    pattern_type: str          # e.g., "assumption_without_evidence", "circular_reasoning"
    description: str
    occurrences: int = 1
    first_seen: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_seen: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class BoundaryEvent:
    """Historical boundary evaluation."""
    timestamp: str
    severity_score: int        # 0–100
    risk_level: str            # LOW | MEDIUM | HIGH
    claim_type: str
    explanation: str


@dataclass
class SessionState:
    """Complete cognitive session state."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    chat_name: Optional[str] = None
    primary_goal: Optional[str] = None
    inferred_domain: str = "general"
    user_expertise_score: float = 0.5
    message_count: int = 0
    error_patterns: List[ErrorPattern] = field(default_factory=list)
    boundary_history: List[BoundaryEvent] = field(default_factory=list)
    disagreement_score: float = 0.0
    fragility_index: float = 0.0
    session_confidence: float = 0.5
    reasoning_depth: str = "standard"    # minimal | standard | deep | maximum
    # v3.0 extensions
    behavioral_risk_profile: Optional[Dict[str, Any]] = None  # Latest behavioral analytics
    reliability_score: float = 0.5       # Derived from learning/feedback (0–1)
    sub_mode_history: List[str] = field(default_factory=list)  # Track sub-mode usage
    # Session Intelligence 2.0 fields
    cognitive_drift_score: float = 0.0   # 0–1: how much the topic has drifted over session
    topic_stability_score: float = 1.0   # 0–1: how stable the topic remains across messages
    confidence_volatility_index: float = 0.0  # 0–1: how much confidence fluctuates
    debate_history_graph: List[Dict[str, Any]] = field(default_factory=list)  # Per-round debate summary
    boundary_trend: str = "stable"       # escalating | stabilizing | flat | stable
    behavioral_risk_accumulator: float = 0.0  # Cumulative behavioral risk across session
    model_reliability_scores: Dict[str, float] = field(default_factory=lambda: {"groq": 1.0, "mistral": 1.0, "qwen": 1.0})
    confidence_trace_history: List[float] = field(default_factory=list)  # All confidence values in session
    domain_history: List[str] = field(default_factory=list)  # Domain per message
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "chat_name": self.chat_name,
            "primary_goal": self.primary_goal,
            "inferred_domain": self.inferred_domain,
            "user_expertise_score": self.user_expertise_score,
            "message_count": self.message_count,
            "error_patterns": [
                {"type": ep.pattern_type, "description": ep.description, "occurrences": ep.occurrences}
                for ep in self.error_patterns
            ],
            "boundary_history_count": len(self.boundary_history),
            "latest_boundary_severity": self.boundary_history[-1].severity_score if self.boundary_history else 0,
            "boundary_trend": self.boundary_trend,
            "disagreement_score": round(self.disagreement_score, 4),
            "fragility_index": round(self.fragility_index, 4),
            "session_confidence": round(self.session_confidence, 4),
            "reasoning_depth": self.reasoning_depth,
            "behavioral_risk_profile": self.behavioral_risk_profile,
            "reliability_score": round(self.reliability_score, 4),
            "sub_mode_history": self.sub_mode_history[-10:],
            # Session Intelligence 2.0
            "cognitive_drift_score": round(self.cognitive_drift_score, 4),
            "topic_stability_score": round(self.topic_stability_score, 4),
            "confidence_volatility_index": round(self.confidence_volatility_index, 4),
            "debate_history_graph": self.debate_history_graph[-5:],  # Last 5 debates
            "behavioral_risk_accumulator": round(self.behavioral_risk_accumulator, 4),
            "model_reliability_scores": {k: round(v, 4) for k, v in self.model_reliability_scores.items()},
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }

    def _compute_boundary_trend(self) -> str:
        if len(self.boundary_history) < 2:
            return "stable"
        recent = self.boundary_history[-3:]
        scores = [b.severity_score for b in recent]
        if scores[-1] > scores[0]:
            return "escalating"
        elif scores[-1] < scores[0]:
            return "stabilizing"
        return "flat"

    def serialize(self) -> Dict[str, Any]:
        """Full serialization for Redis/Postgres persistence."""
        return {
            "session_id": self.session_id,
            "chat_name": self.chat_name,
            "primary_goal": self.primary_goal,
            "inferred_domain": self.inferred_domain,
            "user_expertise_score": self.user_expertise_score,
            "message_count": self.message_count,
            "error_patterns": [
                {
                    "pattern_type": ep.pattern_type,
                    "description": ep.description,
                    "occurrences": ep.occurrences,
                    "first_seen": ep.first_seen,
                    "last_seen": ep.last_seen,
                }
                for ep in self.error_patterns
            ],
            "boundary_history": [
                {
                    "timestamp": be.timestamp,
                    "severity_score": be.severity_score,
                    "risk_level": be.risk_level,
                    "claim_type": be.claim_type,
                    "explanation": be.explanation,
                }
                for be in self.boundary_history
            ],
            "disagreement_score": self.disagreement_score,
            "fragility_index": self.fragility_index,
            "session_confidence": self.session_confidence,
            "reasoning_depth": self.reasoning_depth,
            "behavioral_risk_profile": self.behavioral_risk_profile,
            "reliability_score": self.reliability_score,
            "sub_mode_history": self.sub_mode_history,
            # 2.0 fields
            "cognitive_drift_score": self.cognitive_drift_score,
            "topic_stability_score": self.topic_stability_score,
            "confidence_volatility_index": self.confidence_volatility_index,
            "debate_history_graph": self.debate_history_graph,
            "boundary_trend": self.boundary_trend,
            "behavioral_risk_accumulator": self.behavioral_risk_accumulator,
            "model_reliability_scores": self.model_reliability_scores,
            "confidence_trace_history": self.confidence_trace_history,
            "domain_history": self.domain_history,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """Restore SessionState from serialized dict (Redis/Postgres)."""
        state = cls(
            session_id=data.get("session_id", str(uuid.uuid4())),
            chat_name=data.get("chat_name"),
            primary_goal=data.get("primary_goal"),
            inferred_domain=data.get("inferred_domain", "general"),
            user_expertise_score=data.get("user_expertise_score", 0.5),
            message_count=data.get("message_count", 0),
            disagreement_score=data.get("disagreement_score", 0.0),
            fragility_index=data.get("fragility_index", 0.0),
            session_confidence=data.get("session_confidence", 0.5),
            reasoning_depth=data.get("reasoning_depth", "standard"),
            behavioral_risk_profile=data.get("behavioral_risk_profile"),
            reliability_score=data.get("reliability_score", 0.5),
            sub_mode_history=data.get("sub_mode_history", []),
            # 2.0 fields
            cognitive_drift_score=data.get("cognitive_drift_score", 0.0),
            topic_stability_score=data.get("topic_stability_score", 1.0),
            confidence_volatility_index=data.get("confidence_volatility_index", 0.0),
            debate_history_graph=data.get("debate_history_graph", []),
            boundary_trend=data.get("boundary_trend", "stable"),
            behavioral_risk_accumulator=data.get("behavioral_risk_accumulator", 0.0),
            model_reliability_scores=data.get("model_reliability_scores", {"groq": 1.0, "mistral": 1.0, "qwen": 1.0}),
            confidence_trace_history=data.get("confidence_trace_history", []),
            domain_history=data.get("domain_history", []),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            last_updated=data.get("last_updated", datetime.utcnow().isoformat()),
        )
        # Restore error patterns
        for ep_data in data.get("error_patterns", []):
            state.error_patterns.append(ErrorPattern(
                pattern_type=ep_data.get("pattern_type", "unknown"),
                description=ep_data.get("description", ""),
                occurrences=ep_data.get("occurrences", 1),
                first_seen=ep_data.get("first_seen", datetime.utcnow().isoformat()),
                last_seen=ep_data.get("last_seen", datetime.utcnow().isoformat()),
            ))
        # Restore boundary history
        for bh_data in data.get("boundary_history", []):
            state.boundary_history.append(BoundaryEvent(
                timestamp=bh_data.get("timestamp", datetime.utcnow().isoformat()),
                severity_score=bh_data.get("severity_score", 0),
                risk_level=bh_data.get("risk_level", "LOW"),
                claim_type=bh_data.get("claim_type", "unknown"),
                explanation=bh_data.get("explanation", ""),
            ))
        return state


class SessionIntelligence:
    """
    Manages the cognitive session lifecycle.
    Performs domain inference, expertise estimation, error tracking,
    and session metric updates on every interaction.
    """

    def __init__(self):
        self.state = SessionState()
        self._expertise_history: List[float] = []

    # ============================================================
    # INITIALIZATION (First Message)
    # ============================================================

    def initialize(self, text: str, mode: str) -> SessionState:
        """Called on first message. Sets up session baseline."""
        self.state.chat_name = self._generate_chat_name(text)
        self.state.primary_goal = self._infer_goal(text)
        self.state.inferred_domain = self._infer_domain(text)
        self.state.user_expertise_score = self._estimate_expertise(text)
        self.state.reasoning_depth = self._calibrate_depth(mode)
        self.state.message_count = 1
        self.state.last_updated = datetime.utcnow().isoformat()
        return self.state

    # ============================================================
    # UPDATE (Each Subsequent Message)
    # ============================================================

    def update(self, text: str, mode: str, boundary_result: Optional[Dict] = None,
               disagreement_data: Optional[Dict] = None) -> SessionState:
        """Called on every message. Updates all session metrics."""
        self.state.message_count += 1

        # Domain refinement
        new_domain = self._infer_domain(text)
        if new_domain != "general":
            self.state.inferred_domain = new_domain

        # Expertise evolution (EMA)
        new_expertise = self._estimate_expertise(text)
        self._expertise_history.append(new_expertise)
        alpha = 0.3
        self.state.user_expertise_score = round(
            alpha * new_expertise + (1 - alpha) * self.state.user_expertise_score, 4
        )

        # Reasoning depth calibration
        self.state.reasoning_depth = self._calibrate_depth(mode)

        # Error pattern detection
        self._detect_error_patterns(text)

        # Boundary history
        if boundary_result:
            self.state.boundary_history.append(BoundaryEvent(
                timestamp=datetime.utcnow().isoformat(),
                severity_score=boundary_result.get("severity_score", 0),
                risk_level=boundary_result.get("risk_level", "LOW"),
                claim_type=boundary_result.get("claim_type", "unknown"),
                explanation=boundary_result.get("explanation", ""),
            ))

        # Disagreement score
        if disagreement_data:
            self.state.disagreement_score = self._compute_disagreement(disagreement_data)

        # 2.0: Track domain for drift computation
        self.state.domain_history.append(new_domain)
        
        # 2.0: Cognitive drift
        self.state.cognitive_drift_score = self._compute_cognitive_drift()
        
        # 2.0: Topic stability
        self.state.topic_stability_score = self._compute_topic_stability()
        
        # 2.0: Boundary trend
        self.state.boundary_trend = self.state._compute_boundary_trend()

        # Fragility index (3.X formula)
        self.state.fragility_index = self._compute_fragility()

        # Session confidence
        self.state.session_confidence = self._compute_session_confidence()
        
        # 2.0: Track confidence for volatility
        self.state.confidence_trace_history.append(self.state.session_confidence)
        self.state.confidence_volatility_index = self._compute_confidence_volatility()

        self.state.last_updated = datetime.utcnow().isoformat()
        return self.state

    # ============================================================
    # INTERNAL METHODS
    # ============================================================

    def _generate_chat_name(self, text: str) -> str:
        """Generate a max-6-word chat name from input text."""
        stop_words = {
            "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
            "be", "been", "being", "in", "on", "at", "to", "for", "from", "with",
            "by", "about", "this", "that", "it", "of", "do", "does", "how", "what",
            "why", "can", "will", "just", "not", "i", "me", "my", "we", "you",
        }
        clean = re.sub(r'[^\w\s]', '', text).strip()
        words = clean.split()
        meaningful = [w for w in words if w.lower() not in stop_words and len(w) > 2]
        if not meaningful:
            meaningful = words[:3]
        selected = meaningful[:6]
        name = " ".join(selected).title()
        return name[:60] if name else "New Analysis"

    def _infer_goal(self, text: str) -> str:
        """Infer the user's primary goal from the initial query."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["fix", "debug", "error", "bug", "broken", "crash"]):
            return "debugging"
        elif any(w in text_lower for w in ["build", "create", "implement", "develop", "make"]):
            return "implementation"
        elif any(w in text_lower for w in ["analyze", "evaluate", "assess", "review", "audit"]):
            return "analysis"
        elif any(w in text_lower for w in ["explain", "what is", "how does", "why"]):
            return "understanding"
        elif any(w in text_lower for w in ["compare", "versus", "vs", "difference", "better"]):
            return "comparison"
        elif any(w in text_lower for w in ["optimize", "improve", "faster", "efficient", "scale"]):
            return "optimization"
        elif any(w in text_lower for w in ["secure", "vulnerability", "threat", "risk"]):
            return "security_assessment"
        return "general_inquiry"

    def _infer_domain(self, text: str) -> str:
        """Score text against domain keyword taxonomy."""
        text_lower = text.lower()
        scores = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[domain] = score
        if not scores:
            return "general"
        return max(scores, key=scores.get)

    def _estimate_expertise(self, text: str) -> float:
        """Estimate user expertise from linguistic signals."""
        text_lower = text.lower()
        high_count = sum(1 for w in EXPERTISE_SIGNALS["high"] if w in text_lower)
        med_count = sum(1 for w in EXPERTISE_SIGNALS["medium"] if w in text_lower)
        low_count = sum(1 for w in EXPERTISE_SIGNALS["low"] if w in text_lower)

        # Weighted scoring
        if high_count >= 2:
            return min(1.0, 0.85 + high_count * 0.03)
        elif high_count >= 1:
            return 0.75 + med_count * 0.02
        elif med_count >= 2:
            return 0.55 + med_count * 0.03
        elif low_count >= 2:
            return max(0.1, 0.35 - low_count * 0.05)
        elif med_count >= 1:
            return 0.5
        return 0.5

    def _calibrate_depth(self, mode: str) -> str:
        """Calibrate reasoning depth based on mode + expertise."""
        expertise = self.state.user_expertise_score
        if mode == "experimental":
            return "maximum"
        elif mode == "forensic":
            return "deep"
        elif expertise >= 0.8:
            return "deep"
        elif expertise >= 0.5:
            return "standard"
        return "minimal"

    def _detect_error_patterns(self, text: str):
        """Detect common logical errors in user input."""
        text_lower = text.lower()
        patterns_to_check = [
            ("assumption_without_evidence",
             ["obviously", "everyone knows", "it's clear that", "naturally"],
             "Unstated assumption presented as fact"),
            ("circular_reasoning",
             ["because it is", "by definition", "that's just how it is"],
             "Circular justification detected"),
            ("false_dichotomy",
             ["either", "only two", "you have to choose", "no other option"],
             "Possible false dichotomy"),
            ("hasty_generalization",
             ["always", "never", "every time", "all of them", "none of them"],
             "Overgeneralization without sufficient evidence"),
            ("appeal_to_authority",
             ["experts say", "google says", "according to everyone"],
             "Unverified appeal to authority"),
            ("missing_constraints",
             [],  # Detected structurally, not by keywords
             "Missing operational constraints"),
        ]

        for pattern_type, triggers, description in patterns_to_check:
            if triggers and any(t in text_lower for t in triggers):
                # Check if already tracked
                existing = next((ep for ep in self.state.error_patterns if ep.pattern_type == pattern_type), None)
                if existing:
                    existing.occurrences += 1
                    existing.last_seen = datetime.utcnow().isoformat()
                else:
                    self.state.error_patterns.append(ErrorPattern(
                        pattern_type=pattern_type,
                        description=description,
                    ))

    def _compute_disagreement(self, data: Dict) -> float:
        """Compute disagreement score from debate data."""
        disagreements = data.get("disagreements", [])
        agreements = data.get("agreements", [])
        total = len(disagreements) + len(agreements)
        if total == 0:
            return 0.0
        return round(len(disagreements) / total, 4)

    def _compute_fragility(self) -> float:
        """
        Fragility index v3.X:
        
        fragility = 0.4 * disagreement_score
                   + 0.3 * boundary_severity (normalized)
                   + 0.2 * evidence_contradictions (from behavioral risk)
                   + 0.1 * confidence_volatility
        
        Fragility evolves across the session.
        """
        fragility = 0.0

        # 0.4 * disagreement_score
        fragility += 0.4 * self.state.disagreement_score

        # 0.3 * boundary_severity (normalized, from recent history)
        if self.state.boundary_history:
            recent_severities = [b.severity_score for b in self.state.boundary_history[-5:]]
            avg_severity = sum(recent_severities) / len(recent_severities)
            fragility += 0.3 * (avg_severity / 100)

        # 0.2 * evidence_contradictions (approximated from behavioral risk accumulator)
        fragility += 0.2 * min(self.state.behavioral_risk_accumulator, 1.0)

        # 0.1 * confidence_volatility
        fragility += 0.1 * self.state.confidence_volatility_index

        return round(min(1.0, fragility), 4)

    def _compute_session_confidence(self) -> float:
        """
        Overall session confidence. Inversely related to fragility,
        modulated by expertise and boundary safety.
        """
        base = 1.0 - self.state.fragility_index

        # Boundary safety penalty
        if self.state.boundary_history:
            latest = self.state.boundary_history[-1].severity_score
            if latest > 70:
                base -= 0.15
            elif latest > 50:
                base -= 0.08

        # Expertise bonus
        if self.state.user_expertise_score > 0.7:
            base += 0.05

        return round(max(0.0, min(1.0, base)), 4)

    # ============================================================
    # SESSION INTELLIGENCE 2.0 — NEW METRICS
    # ============================================================

    def _compute_cognitive_drift(self) -> float:
        """
        Compute cognitive drift: how much the topic has changed over the session.
        0.0 = perfectly focused, 1.0 = completely drifted.
        
        Uses domain_history to detect topic changes.
        """
        history = self.state.domain_history
        if len(history) < 2:
            return 0.0
        
        # Count domain changes
        changes = sum(1 for i in range(1, len(history)) if history[i] != history[i-1])
        drift = changes / max(len(history) - 1, 1)
        
        # Weight recent changes more heavily
        recent = history[-5:]
        recent_changes = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])
        recent_drift = recent_changes / max(len(recent) - 1, 1)
        
        return round(drift * 0.4 + recent_drift * 0.6, 4)

    def _compute_topic_stability(self) -> float:
        """
        Compute topic stability: inverse of cognitive drift with momentum.
        High = topic has been consistent, Low = topic is jumping around.
        """
        drift = self.state.cognitive_drift_score
        # Stability is inverse of drift with a smoothing factor
        stability = 1.0 - drift
        
        # Penalize if current domain differs from most common domain
        history = self.state.domain_history
        if history:
            from collections import Counter
            most_common = Counter(history).most_common(1)[0][0]
            if history[-1] != most_common and len(history) > 3:
                stability *= 0.8
        
        return round(max(0.0, min(1.0, stability)), 4)

    def _compute_confidence_volatility(self) -> float:
        """
        Compute confidence volatility: how much confidence fluctuates across the session.
        0.0 = perfectly stable, 1.0 = wildly fluctuating.
        
        Uses standard deviation of recent confidence values.
        """
        trace = self.state.confidence_trace_history
        if len(trace) < 3:
            return 0.0
        
        # Use last 10 values
        recent = trace[-10:]
        mean_conf = sum(recent) / len(recent)
        variance = sum((c - mean_conf) ** 2 for c in recent) / len(recent)
        std_dev = variance ** 0.5
        
        # Normalize: std_dev of 0.25+ is considered maximum volatility
        volatility = min(std_dev / 0.25, 1.0)
        
        return round(volatility, 4)

    def update_behavioral_risk(self, risk_profile: Dict[str, Any]):
        """
        Update session behavioral risk accumulator.
        Uses exponential moving average to avoid spikes.
        """
        new_risk = risk_profile.get("overall_risk", 0.0)
        alpha = 0.3  # Smoothing factor
        self.state.behavioral_risk_accumulator = round(
            alpha * new_risk + (1 - alpha) * self.state.behavioral_risk_accumulator,
            4
        )
        self.state.behavioral_risk_profile = risk_profile

    def record_debate_round(self, debate_summary: Dict[str, Any]):
        """Record a debate result in the session debate history graph."""
        self.state.debate_history_graph.append({
            "timestamp": datetime.utcnow().isoformat(),
            "disagreement_strength": debate_summary.get("disagreement_strength", 0.0),
            "convergence_level": debate_summary.get("convergence_level", "none"),
            "confidence_recalibration": debate_summary.get("confidence_recalibration", 0.5),
            "models_participated": debate_summary.get("models_used", []),
        })

    def update_model_reliability(self, model_id: str, delta: float):
        """
        Update a model's reliability score in the session.
        Delta is positive for good performance, negative for bad.
        Uses EMA for smooth transitions.
        """
        current = self.state.model_reliability_scores.get(model_id, 1.0)
        alpha = 0.2
        new_val = alpha * max(0.0, min(1.0, current + delta)) + (1 - alpha) * current
        self.state.model_reliability_scores[model_id] = round(max(0.3, min(1.0, new_val)), 4)

    # ============================================================
    # PUBLIC ACCESSORS
    # ============================================================

    def get_state(self) -> SessionState:
        return self.state

    def get_state_dict(self) -> Dict[str, Any]:
        return self.state.to_dict()

    def serialize(self) -> Dict[str, Any]:
        """Serialize entire session intelligence for Redis persistence."""
        return {
            "state": self.state.serialize(),
            "expertise_history": self._expertise_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionIntelligence":
        """Restore SessionIntelligence from serialized dict."""
        si = cls()
        si.state = SessionState.from_dict(data.get("state", {}))
        si._expertise_history = data.get("expertise_history", [])
        return si

    def get_kill_diagnostic(self) -> Dict[str, Any]:
        """Returns the KILL mode diagnostic snapshot."""
        return {
            "chat_name": self.state.chat_name,
            "primary_goal": self.state.primary_goal,
            "inferred_domain": self.state.inferred_domain,
            "user_expertise_score": round(self.state.user_expertise_score, 4),
            "latest_boundary_severity": (
                self.state.boundary_history[-1].severity_score
                if self.state.boundary_history else 0
            ),
            "boundary_trend": self.state._compute_boundary_trend(),
            "disagreement_score": round(self.state.disagreement_score, 4),
            "fragility_index": round(self.state.fragility_index, 4),
            "recurring_error_patterns": [
                {"type": ep.pattern_type, "description": ep.description, "occurrences": ep.occurrences}
                for ep in self.state.error_patterns
            ],
            "session_confidence": round(self.state.session_confidence, 4),
            "session_confidence_explanation": self._confidence_explanation(),
        }

    def _confidence_explanation(self) -> str:
        conf = self.state.session_confidence
        if conf >= 0.85:
            return "High stability. Low fragility, consistent reasoning."
        elif conf >= 0.65:
            return "Moderate stability. Some boundary or disagreement pressure."
        elif conf >= 0.4:
            return "Elevated uncertainty. Multiple risk factors active."
        return "Low confidence. High fragility and/or boundary violations."

    def get_descriptive_summary(self) -> Dict[str, Any]:
        """
        Returns a descriptive summary for the right panel / frontend display.
        All raw numbers converted to human-readable labels.
        v4.5: No raw floats — everything has a label.
        """
        expertise = self.state.user_expertise_score
        if expertise >= 0.8:
            expertise_label = "Expert"
        elif expertise >= 0.6:
            expertise_label = "Advanced"
        elif expertise >= 0.4:
            expertise_label = "Intermediate"
        elif expertise >= 0.2:
            expertise_label = "Developing"
        else:
            expertise_label = "Beginner"

        domain_map = {
            "software_engineering": "Software Engineering",
            "machine_learning": "Machine Learning & AI",
            "cybersecurity": "Cybersecurity",
            "business_strategy": "Business Strategy",
            "scientific_research": "Scientific Research",
            "systems_thinking": "Systems Thinking",
        }
        domain = self.state.inferred_domain
        domain_display = domain_map.get(domain, domain.replace("_", " ").title() if domain else "General")

        conf = self.state.session_confidence
        if conf >= 0.85:
            conf_label = "High"
        elif conf >= 0.65:
            conf_label = "Moderate"
        elif conf >= 0.40:
            conf_label = "Fair"
        elif conf >= 0.20:
            conf_label = "Low"
        else:
            conf_label = "Very Low"

        frag = self.state.fragility_index
        if frag >= 0.7:
            frag_label = "High — exercise caution"
        elif frag >= 0.4:
            frag_label = "Moderate — some risk present"
        elif frag >= 0.2:
            frag_label = "Low — session is stable"
        else:
            frag_label = "Minimal — no concerns"

        return {
            "chat_name": self.state.chat_name or "New Analysis",
            "goal": self.state.primary_goal or "General Inquiry",
            "domain": domain_display,
            "domain_key": domain,
            "expertise": {
                "label": expertise_label,
                "score": round(expertise, 2),
                "description": f"{expertise_label} (inferred from terminology use)",
            },
            "confidence": {
                "label": conf_label,
                "score": round(conf, 2),
            },
            "fragility": {
                "label": frag_label,
                "score": round(frag, 2),
            },
            "disagreement": {
                "score": round(self.state.disagreement_score, 2),
                "label": "Strong" if self.state.disagreement_score > 0.6 else "Moderate" if self.state.disagreement_score > 0.3 else "Low",
            },
            "message_count": self.state.message_count,
            "reasoning_depth": self.state.reasoning_depth,
            "error_count": sum(ep.occurrences for ep in self.state.error_patterns),
            "boundary_count": len(self.state.boundary_history),
            "last_boundary_severity": self.state.boundary_history[-1].severity_score if self.state.boundary_history else 0,
        }
