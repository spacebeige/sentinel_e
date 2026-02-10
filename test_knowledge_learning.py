#!/usr/bin/env python3
"""
Knowledge Learning System Demo

Demonstrates how Sentinel-E learns from past runs to improve future decisions.
Shows:
1. Recording boundary violations
2. Tracking model-specific patterns
3. Analyzing claim-type risks
4. Generating threshold adjustment suggestions
5. Correlation with user feedback
"""

import asyncio
from backend.core.knowledge_learner import KnowledgeLearner
from datetime import datetime


def print_section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


async def demo_knowledge_learning():
    # Initialize learner
    learner = KnowledgeLearner()
    
    print_section("KNOWLEDGE LEARNING SYSTEM DEMO")
    
    # ========================================================================
    # SCENARIO 1: Medical Claims (High Risk)
    # ========================================================================
    print("SCENARIO 1: Medical Claims Analysis")
    print("-" * 80)
    print("Recording 5 medical claims from 'Qwen' model...")
    
    for i in range(5):
        learner.record_boundary_violation(
            model_name="qwen",
            severity_score=85 + (i % 3) * 5,  # 85, 90, 95, 85, 90
            severity_level="critical",
            claim_type="medical_claim",
            run_id=f"run_qwen_{i}"
        )
    
    profile = learner.get_model_risk_profile("qwen")
    print(f"\n✓ Qwen Model Profile after 5 medical claims:")
    print(f"  • Violation Count: {profile['violation_count']}")
    print(f"  • Mean Severity: {profile['mean_severity']}")
    print(f"  • Risk Level: {profile['risk_level']}")
    print(f"  • Medical Claims Average: {profile['claim_type_concerns'].get('medical_claim', {}).get('avg_severity', 'N/A')}")
    
    # ========================================================================
    # SCENARIO 2: Predictive Claims (Mixed Quality)
    # ========================================================================
    print_section("SCENARIO 2: Predictive Claims Learning")
    print("Recording 3 well-grounded + 2 ungrounded predictive claims from 'Mistral'...")
    
    # Well-grounded predictions
    for i in range(3):
        learner.record_boundary_violation(
            model_name="mistral",
            severity_score=30 + (i * 10),  # 30, 40, 50
            severity_level="low",
            claim_type="predictive_claim",
            run_id=f"run_mistral_good_{i}"
        )
    
    # Ungrounded predictions
    for i in range(2):
        learner.record_boundary_violation(
            model_name="mistral",
            severity_score=75 + (i * 5),  # 75, 80
            severity_level="high",
            claim_type="predictive_claim",
            run_id=f"run_mistral_bad_{i}"
        )
    
    mistral_profile = learner.get_model_risk_profile("mistral")
    print(f"\n✓ Mistral Model Profile after mixed predictions:")
    print(f"  • Violation Count: {mistral_profile['violation_count']}")
    print(f"  • Mean Severity: {mistral_profile['mean_severity']}")
    print(f"  • Risk Level: {mistral_profile['risk_level']}")
    print(f"  • Predictive Claims Average: {mistral_profile['claim_type_concerns'].get('predictive_claim', {}).get('avg_severity', 'N/A')}")
    
    # ========================================================================
    # SCENARIO 3: User Feedback Correlation
    # ========================================================================
    print_section("SCENARIO 3: Learning from User Feedback")
    print("Recording user feedback on Qwen's medical claims...")
    
    # Users didn't like Qwen's medical claims
    learner.record_feedback("run_qwen_0", "qwen", "down", "Dangerous medical misinformation")
    learner.record_feedback("run_qwen_1", "qwen", "down", "Not supported by evidence")
    learner.record_feedback("run_qwen_2", "qwen", "down", "Lacks clinical data")
    
    qwen_updated = learner.get_model_risk_profile("qwen")
    print(f"\n✓ Qwen Profile with Feedback:")
    print(f"  • Feedback Sentiment: {qwen_updated['feedback_sentiment']}")
    print(f"  • User Satisfaction: {(qwen_updated['feedback_sentiment'].get('up', 0) / max(sum(qwen_updated['feedback_sentiment'].values()), 1)) * 100:.1f}%")
    
    # Users liked Mistral's good predictions
    learner.record_feedback("run_mistral_good_0", "mistral", "up", "Well-reasoned and evidence-based")
    learner.record_feedback("run_mistral_good_1", "mistral", "up", "Clear confidence bounds")
    
    mistral_updated = learner.get_model_risk_profile("mistral")
    print(f"\n✓ Mistral Profile with Feedback:")
    print(f"  • Feedback Sentiment: {mistral_updated['feedback_sentiment']}")
    
    # ========================================================================
    # SCENARIO 4: Claim Type Risk Analysis
    # ========================================================================
    print_section("SCENARIO 4: Claim Type Risk Summary")
    print("Analyzing risks across all claim types from all models...")
    
    claim_risks = learner.get_claim_type_risk_summary()
    
    for claim_type, risks in claim_risks.items():
        print(f"\n  📊 {claim_type.upper()}")
        print(f"     • Total Violations: {risks['total_violations']}")
        print(f"     • Models Affected: {', '.join(risks['models_affected'])}")
        print(f"     • Avg Severity: {risks['avg_severity']}/100")
        print(f"     • Risk Level: {risks['risk_level'].upper()}")
        print(f"     • Violation Rate: {risks['violation_rate']*100:.1f}%")
    
    # ========================================================================
    # SCENARIO 5: Threshold Adjustment Suggestions
    # ========================================================================
    print_section("SCENARIO 5: AI Learning Suggestions")
    print("Analyzing patterns to suggest threshold adjustments...")
    
    suggestions = learner.suggest_threshold_adjustments()
    
    if suggestions:
        for model, suggestion in suggestions.items():
            print(f"\n  🤖 {model.upper()}")
            print(f"     Action: {suggestion['action']}")
            print(f"     Reason: {suggestion['reason']}")
            profile = suggestion['current_profile']
            print(f"     Mean Severity: {profile['mean_severity']}")
    else:
        print("\n  No threshold adjustments suggested at this time.")
    
    # ========================================================================
    # SCENARIO 6: Overall Learning Summary
    # ========================================================================
    print_section("SCENARIO 6: Complete Learning Summary")
    
    summary = learner.get_learning_summary()
    
    print(f"\n  📈 STATISTICS")
    print(f"     • Total Violations Recorded: {summary['total_violations_recorded']}")
    print(f"     • Total Refusals Issued: {summary['total_refusals_issued']}")
    print(f"     • Models Tracked: {summary['models_tracked']}")
    print(f"     • Feedback Collected: {summary['feedback_collected']}")
    
    print(f"\n  👍 FEEDBACK BREAKDOWN")
    fb = summary['feedback_breakdown']
    print(f"     • Thumbs Up: {fb['up']}")
    print(f"     • Thumbs Down: {fb['down']}")
    print(f"     • Neutral: {fb['neutral']}")
    if summary['user_satisfaction'] is not None:
        print(f"     • Overall Satisfaction: {summary['user_satisfaction']*100:.1f}%")
    
    print(f"\n  🎯 LEARNING OUTCOMES")
    print(f"     Timestamp: {summary['timestamp']}")
    
    # ========================================================================
    # SCENARIO 7: How Learning Would Be Applied
    # ========================================================================
    print_section("SCENARIO 7: Applying Learning to Future Decisions")
    print("""
    BEFORE LEARNING:
    - All medical claims scored with default 90-point rubric
    - All models treated equally
    - Refusal threshold: 70 (global)
    
    AFTER LEARNING (with this demo):
    ✅ Qwen Medical Claims:
       - Mean severity: 89/100 (critical)
       - User feedback: 100% negative
       → ACTION: Lower threshold to 65 for medical claims from Qwen
       → RESULT: More protective against medical misinformation
    
    ✅ Mistral Predictive Claims:
       - Mean severity: 57/100 (acceptable range)
       - User feedback: 100% positive (good predictions)
       → ACTION: Can raise threshold to 75 for quality predictions
       → RESULT: Allows high-quality analysis while blocking poor predictions
    """)
    
    print_section("KEY INSIGHTS")
    print("""
    1. PATTERN DETECTION
       - Learned that Qwen consistently produces unsafe medical claims
       - Learned that Mistral has variable quality in predictions
    
    2. USER ALIGNMENT
       - Users consistently reject Qwen's medical output
       - Users approve of Mistral's evidence-based predictions
    
    3. PERSONALIZED THRESHOLDS
       - Model-specific: Qwen (medical) needs stricter boundaries than global
       - Claim-type-specific: Medical claims require different strictness than predictions
    
    4. CONTINUOUS IMPROVEMENT
       - Each run adds more data points
       - Feedback refines the risk models
       - Suggestions become more accurate over time
    
    API ENDPOINTS FOR ACCESSING LEARNING:
    ✓ GET /learning/summary - Overall statistics and suggestions
    ✓ GET /learning/model/{name} - Model-specific risk profile
    ✓ GET /learning/all-models - All tracked models
    ✓ GET /learning/claim-types - Claim type risk analysis
    ✓ GET /learning/suggestions - Threshold adjustment recommendations
    """)
    
    print("=" * 80)
    print("  DEMO COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(demo_knowledge_learning())
