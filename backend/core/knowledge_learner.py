"""
Knowledge Learner: Learn from past boundary violations and model behavior.

Responsibilities:
- Analyze historical boundary violations
- Detect patterns in model behavior
- Suggest threshold adjustments based on feedback
- Track temporal trends in grounding completeness
- Provide model-specific risk assessments
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger("KnowledgeLearner")


class ModelBehaviorPattern:
    """Represents learned behavior pattern for a model."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.violation_history = []  # List of (timestamp, severity_score, severity_level)
        self.claim_type_stats = defaultdict(lambda: {"count": 0, "avg_severity": 0.0})
        self.feedback_correlation = {"up": 0, "down": 0, "neutral": 0}
        self.last_updated = datetime.utcnow()
        
    def add_violation(self, severity_score: float, severity_level: str, claim_type: str):
        """Record a new boundary violation."""
        self.violation_history.append({
            "timestamp": datetime.utcnow(),
            "severity_score": severity_score,
            "severity_level": severity_level,
            "claim_type": claim_type,
        })
        
        # Update claim-type statistics
        stats = self.claim_type_stats[claim_type]
        count = stats["count"]
        old_avg = stats["avg_severity"]
        new_avg = (old_avg * count + severity_score) / (count + 1)
        stats["count"] = count + 1
        stats["avg_severity"] = new_avg
        
        self.last_updated = datetime.utcnow()
    
    def add_feedback(self, feedback: str):
        """Record user feedback correlation."""
        if feedback in ["up", "down"]:
            self.feedback_correlation[feedback] += 1
        else:
            self.feedback_correlation["neutral"] += 1
    
    def get_risk_profile(self) -> Dict[str, Any]:
        """Generate risk profile for this model."""
        if not self.violation_history:
            return {
                "model_name": self.model_name,
                "violation_count": 0,
                "mean_severity": 0.0,
                "max_severity": 0.0,
                "risk_level": "low",
                "days_tracked": 0,
                "claim_type_concerns": {},
                "feedback_sentiment": self.feedback_correlation
            }
        
        severities = [v["severity_score"] for v in self.violation_history]
        mean_severity = sum(severities) / len(severities)
        max_severity = max(severities)
        
        # Determine risk level
        if mean_severity >= 80:
            risk_level = "critical"
        elif mean_severity >= 70:
            risk_level = "high"
        elif mean_severity >= 50:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Identify claim types of concern
        claim_type_concerns = {}
        for claim_type, stats in self.claim_type_stats.items():
            if stats["avg_severity"] >= 70:
                claim_type_concerns[claim_type] = {
                    "count": stats["count"],
                    "avg_severity": round(stats["avg_severity"], 2),
                    "risk": "high" if stats["avg_severity"] >= 80 else "medium"
                }
        
        # Calculate days tracked safely
        start_time = self.violation_history[0]["timestamp"]
        if isinstance(start_time, str):
            try:
                start_time = datetime.fromisoformat(start_time)
            except ValueError:
                start_time = datetime.utcnow()

        return {
            "model_name": self.model_name,
            "violation_count": len(self.violation_history),
            "mean_severity": round(mean_severity, 2),
            "max_severity": round(max_severity, 2),
            "risk_level": risk_level,
            "days_tracked": (datetime.utcnow() - start_time).days,
            "claim_type_concerns": claim_type_concerns,
            "feedback_sentiment": self.feedback_correlation,
        }
    
    def should_increase_threshold(self) -> bool:
        """Suggest lowering threshold if many violations + negative feedback."""
        profile = self.get_risk_profile()
        
        # If consistently high violations AND negative feedback, yes
        negative_feedback = self.feedback_correlation.get("down", 0)
        total_feedback = sum(self.feedback_correlation.values())
        
        if total_feedback == 0:
            return False
        
        negative_ratio = negative_feedback / total_feedback
        
        return (profile["violation_count"] >= 5 and 
                profile["mean_severity"] >= 70 and 
                negative_ratio >= 0.6)
    
    def should_decrease_threshold(self) -> bool:
        """Suggest raising threshold if few violations + positive feedback."""
        profile = self.get_risk_profile()
        
        # If low violations AND high satisfaction, maybe relax
        positive_feedback = self.feedback_correlation.get("up", 0)
        total_feedback = sum(self.feedback_correlation.values())
        
        if total_feedback == 0:
            return False
        
        positive_ratio = positive_feedback / total_feedback
        
        return (profile["violation_count"] <= 2 and 
                profile["mean_severity"] < 50 and 
                positive_ratio >= 0.8)


class KnowledgeLearner:
    """
    Learn from past runs to improve future decisions.
    
    Maintains in-memory model behavior patterns.
    Persists state to local JSON file for durability.
    """
    
    def __init__(self, persistence_file: str = "sentinel_knowledge.json"):
        self.model_patterns: Dict[str, ModelBehaviorPattern] = {}
        self.global_violation_count = 0
        self.refusal_count = 0
        self.feedback_store = []
        self.persistence_file = persistence_file
        self.load_state()

    def update_feedback(self, model_name: str, feedback: str):
        """Records user feedback for a model's output."""
        if model_name not in self.model_patterns:
            self.model_patterns[model_name] = ModelBehaviorPattern(model_name)
            
        self.model_patterns[model_name].add_feedback(feedback)
        
        # Also store raw feedback in store
        self.feedback_store.append({
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": model_name,
            "feedback": feedback
        })
        
        self.save_state()
        logger.info(f"Feedback '{feedback}' recorded for model {model_name}")
        
    def load_state(self):
        """Load state from persistence file."""
        import json
        import os
        if not os.path.exists(self.persistence_file):
            return
            
        try:
            with open(self.persistence_file, 'r') as f:
                data = json.load(f)
                
            self.global_violation_count = data.get("global_violation_count", 0)
            self.refusal_count = data.get("refusal_count", 0)
            self.feedback_store = data.get("feedback_store", [])
            
            # Reconstruct model patterns
            for name, p_data in data.get("model_patterns", {}).items():
                pattern = ModelBehaviorPattern(name)
                
                # Convert timestamp strings back to datetime objects
                violation_history = []
                for v in p_data.get("violation_history", []):
                    v_copy = v.copy()
                    if isinstance(v_copy.get("timestamp"), str):
                        v_copy["timestamp"] = datetime.fromisoformat(v_copy["timestamp"])
                    violation_history.append(v_copy)
                pattern.violation_history = violation_history
                
                # claim_type_stats key fix (since json keys are strings)
                pattern.claim_type_stats = defaultdict(lambda: {"count": 0, "avg_severity": 0.0}, p_data.get("claim_type_stats", {}))
                pattern.feedback_correlation = p_data.get("feedback_correlation", {"up": 0, "down": 0, "neutral": 0})
                
                self.model_patterns[name] = pattern
                
            logger.info(f"Loaded knowledge state from {self.persistence_file}")
        except Exception as e:
            logger.error(f"Failed to load knowledge state: {e}")

    def save_state(self):
        """Save state to persistence file."""
        import json
        try:
            # Serialize patterns
            patterns_data = {}
            for name, pattern in self.model_patterns.items():
                patterns_data[name] = {
                    "violation_history": [
                        {**v, "timestamp": v["timestamp"].isoformat() if isinstance(v["timestamp"], datetime) else v["timestamp"]}
                        for v in pattern.violation_history
                    ], # Serialize dates
                    "claim_type_stats": dict(pattern.claim_type_stats),
                    "feedback_correlation": pattern.feedback_correlation
                }
                
            data = {
                "global_violation_count": self.global_violation_count,
                "refusal_count": self.refusal_count,
                "feedback_store": self.feedback_store,
                "model_patterns": patterns_data
            }
            
            with open(self.persistence_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save knowledge state: {e}")

    def record_boundary_violation(
        self,
        model_name: str,
        severity_score: float,
        severity_level: str,
        claim_type: str,
        run_id: str
    ):
        """Record a boundary violation for learning."""
        if model_name not in self.model_patterns:
            self.model_patterns[model_name] = ModelBehaviorPattern(model_name)
        
        self.model_patterns[model_name].add_violation(severity_score, severity_level, claim_type)
        self.global_violation_count += 1
        
        logger.info(f"Violation recorded: {model_name} | "
                   f"Severity: {severity_score} | Type: {claim_type}")
        self.save_state()
    
    def record_refusal_decision(
        self,
        model_name: str,
        run_id: str,
        boundary_severity: float,
        refusal_reason: str
    ):
        """Record a refusal decision."""
        self.refusal_count += 1
        logger.info(f"Refusal decision: {model_name} | "
                   f"Severity: {boundary_severity} | Reason: {refusal_reason}")
        self.save_state()
    
    def record_feedback(
        self,
        run_id: str,
        model_name: Optional[str],
        feedback: str,
        reason: Optional[str] = None
    ):
        """Record user feedback for correlation analysis."""
        if model_name and model_name in self.model_patterns:
            self.model_patterns[model_name].add_feedback(feedback)
        
        self.feedback_store.append({
            "run_id": run_id,
            "model_name": model_name,
            "feedback": feedback,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        logger.info(f"Feedback recorded: {feedback} for run {run_id}")
        self.save_state()
    
    def get_model_risk_profile(self, model_name: str) -> Dict[str, Any]:
        """Get learned risk profile for a model."""
        if model_name not in self.model_patterns:
            return {"model_name": model_name, "status": "no_history"}
        
        return self.model_patterns[model_name].get_risk_profile()
    
    def get_all_risk_profiles(self) -> List[Dict[str, Any]]:
        """Get risk profiles for all tracked models."""
        return [pattern.get_risk_profile() for pattern in self.model_patterns.values()]
    
    def suggest_threshold_adjustments(self) -> Dict[str, Dict[str, Any]]:
        """
        Suggest refusal threshold adjustments based on learned patterns.
        
        Returns:
            Dict mapping model name to suggested action
        """
        suggestions = {}
        
        for model_name, pattern in self.model_patterns.items():
            if pattern.should_increase_threshold():
                suggestions[model_name] = {
                    "action": "lower_threshold",
                    "reason": "High violation rate with negative feedback",
                    "current_profile": pattern.get_risk_profile()
                }
            elif pattern.should_decrease_threshold():
                suggestions[model_name] = {
                    "action": "raise_threshold",
                    "reason": "Low violation rate with positive feedback",
                    "current_profile": pattern.get_risk_profile()
                }
        
        return suggestions
    
    def get_claim_type_risk_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze risk by claim type across all models.
        
        Returns:
            Dict mapping claim type to aggregate risk metrics
        """
        claim_type_stats = defaultdict(lambda: {
            "total_count": 0,
            "violation_count": 0,
            "models_affected": set(),
            "avg_severity": 0.0
        })
        
        for pattern in self.model_patterns.values():
            for claim_type, stats in pattern.claim_type_stats.items():
                ctype_data = claim_type_stats[claim_type]
                ctype_data["total_count"] += stats["count"]
                ctype_data["models_affected"].add(pattern.model_name)
                
                # Update running average
                old_count = ctype_data["violation_count"]
                old_avg = ctype_data["avg_severity"]
                new_avg = (old_avg * old_count + stats["avg_severity"] * stats["count"]) / (old_count + stats["count"])
                ctype_data["avg_severity"] = new_avg
                ctype_data["violation_count"] += stats["count"]
        
        # Convert sets to lists and format
        result = {}
        for claim_type, data in claim_type_stats.items():
            result[claim_type] = {
                "total_violations": data["violation_count"],
                "models_affected": list(data["models_affected"]),
                "violation_rate": round(data["violation_count"] / max(data["total_count"], 1), 3),
                "avg_severity": round(data["avg_severity"], 2),
                "risk_level": (
                    "critical" if data["avg_severity"] >= 80 else
                    "high" if data["avg_severity"] >= 70 else
                    "medium" if data["avg_severity"] >= 50 else
                    "low"
                )
            }
        
        return result
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get overall learning summary."""
        self.load_state()  # Refresh state from persistence file
        feedback_stats = {"up": 0, "down": 0, "neutral": 0}
        for fb in self.feedback_store:
            feedback_stats[fb["feedback"]] = feedback_stats.get(fb["feedback"], 0) + 1
        
        return {
            "total_violations_recorded": self.global_violation_count,
            "total_refusals_issued": self.refusal_count,
            "models_tracked": len(self.model_patterns),
            "feedback_collected": len(self.feedback_store),
            "feedback_breakdown": feedback_stats,
            "user_satisfaction": (
                feedback_stats["up"] / max(sum(feedback_stats.values()), 1)
                if sum(feedback_stats.values()) > 0 else None
            ),
            "threshold_suggestions": self.suggest_threshold_adjustments(),
            "claim_type_risks": self.get_claim_type_risk_summary(),
            "model_weights": {
                name: self.get_model_weight(name)
                for name in self.model_patterns
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ============================================================
    # 3.X — ADAPTIVE MODEL WEIGHT SYSTEM
    # ============================================================

    def get_model_weight(self, model_name: str) -> float:
        """
        Get the adaptive influence weight for a model (0.5–1.0).
        
        Models with repeated:
        - High boundary severity
        - High disagreement instability
        - Negative feedback
        must gradually lose influence weight.
        
        Decay is GRADUAL (exponential moving average), not abrupt.
        Weight persists across sessions (via save_state/load_state).
        
        Returns:
            float between 0.5 and 1.0 (1.0 = full trust, 0.5 = minimum trust)
        """
        if model_name not in self.model_patterns:
            return 1.0  # Unknown model = full trust by default
        
        pattern = self.model_patterns[model_name]
        
        # Start from full trust
        weight = 1.0
        
        # Factor 1: Violation severity decay
        # More violations at higher severity = more weight reduction
        if pattern.violation_history:
            recent_violations = pattern.violation_history[-20:]  # Last 20 violations
            severity_scores = [v["severity_score"] for v in recent_violations]
            avg_severity = sum(severity_scores) / len(severity_scores)
            
            # Penalty: gradual decay based on average severity
            # Severity 0-50: minimal penalty, 50-80: moderate, 80+: significant
            severity_penalty = 0.0
            if avg_severity > 80:
                severity_penalty = 0.20
            elif avg_severity > 70:
                severity_penalty = 0.15
            elif avg_severity > 50:
                severity_penalty = 0.08
            elif avg_severity > 30:
                severity_penalty = 0.03
            
            # Scale by violation count (more violations = more of the penalty applies)
            violation_factor = min(len(recent_violations) / 10, 1.0)
            weight -= severity_penalty * violation_factor
        
        # Factor 2: Negative feedback decay
        total_feedback = sum(pattern.feedback_correlation.values())
        if total_feedback > 0:
            negative_ratio = pattern.feedback_correlation["down"] / total_feedback
            positive_ratio = pattern.feedback_correlation["up"] / total_feedback
            
            # Negative feedback reduces weight, positive increases slightly
            feedback_penalty = negative_ratio * 0.15 - positive_ratio * 0.05
            weight -= feedback_penalty
        
        # Factor 3: Temporal decay — recent violations matter more
        if pattern.violation_history:
            recent_cutoff = datetime.utcnow() - timedelta(hours=24)
            very_recent = [
                v for v in pattern.violation_history
                if (isinstance(v["timestamp"], datetime) and v["timestamp"] > recent_cutoff)
            ]
            if len(very_recent) > 3:
                # Many recent violations — additional penalty
                weight -= 0.05
        
        # Clamp to [0.5, 1.0] — never fully distrust
        weight = round(max(0.5, min(1.0, weight)), 4)
        
        return weight

    def get_all_model_weights(self) -> Dict[str, float]:
        """Get adaptive weights for all known models."""
        weights = {}
        known_models = ["groq", "llama70b", "qwen", "Groq", "Llama70B", "Qwen"]
        
        for model_name in known_models:
            weights[model_name.lower()] = self.get_model_weight(model_name)
        
        # Also include any models tracked in patterns
        for model_name in self.model_patterns:
            weights[model_name.lower()] = self.get_model_weight(model_name)
        
        return weights
