#!/usr/bin/env python3
"""
Boundary Detection Test Suite

Test the boundary detection system with sample claims.
Demonstrates:
1. Claim classification
2. Grounding requirement inference
3. Severity scoring
4. Refusal threshold logic
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from core.boundary_detector import BoundaryDetector
import json


def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_test_result(i, test_case, violation):
    """Pretty print a test result."""
    print(f"\n📋 TEST {i}: {test_case['category']}")
    print(f"{'─' * 80}")
    
    print(f"\n  📌 Input Claim:")
    print(f"     \"{test_case['input']}\"")
    
    print(f"\n  📚 Available Evidence:")
    for j, ev in enumerate(test_case['evidence'], 1):
        print(f"     {j}. {ev}")
    
    print(f"\n  🔍 Analysis:")
    print(f"     • Claim Type:        {violation['claim_type']}")
    print(f"     • Grounding Score:   {violation['grounding_score']}/100")
    print(f"     • Severity Level:    {violation['severity_level'].upper()}")
    print(f"     • Severity Score:    {violation['severity_score']}/100")
    
    print(f"\n  ✓ Required Grounding:")
    for req in violation['required_grounding']:
        print(f"     - {req}")
    
    if violation['missing_grounding']:
        print(f"\n  ✗ Missing Grounding:")
        for missing in violation['missing_grounding']:
            print(f"     - {missing}")
    
    print(f"\n  🚫 Refusal Decision:")
    if violation['human_review_required']:
        print(f"     WOULD REFUSE ❌ (severity {violation['severity_score']} >= 70)")
    else:
        print(f"     ALLOWED ✅ (severity {violation['severity_score']} < 70)")
    
    print(f"\n  📝 Expected: {test_case['expected']}")
    print(f"{'─' * 80}")


def main():
    detector = BoundaryDetector()
    
    # Test cases covering different claim types
    test_cases = [
        # WELL-GROUNDED (should score LOW severity)
        {
            "category": "WELL-GROUNDED FACTUAL CLAIM",
            "input": "Water boils at 100 degrees Celsius at sea level",
            "evidence": [
                "Physics textbooks define boiling point",
                "Verified by countless experiments",
                "Explains molecular behavior at pressure/temp equilibrium"
            ],
            "expected": "✅ LOW severity (excellent grounding)"
        },
        
        # MODERATELY GROUNDED
        {
            "category": "MODERATELY GROUNDED CAUSAL CLAIM",
            "input": "Exercise improves cardiovascular health",
            "evidence": [
                "Multiple studies show correlation",
                "Mechanism involves heart strengthening",
                "Recommended by medical guidelines"
            ],
            "expected": "✅ LOW-MEDIUM severity (good but could use more evidence)"
        },
        
        # POORLY GROUNDED MEDICAL CLAIM
        {
            "category": "UNGROUNDED MEDICAL CLAIM",
            "input": "This supplement cures cancer in 100% of patients",
            "evidence": [
                "Some customer testimonials",
                "One anecdotal story online"
            ],
            "expected": "❌ CRITICAL severity (no clinical evidence, dangerous)"
        },
        
        # UNGROUNDED PREDICTIVE CLAIM
        {
            "category": "UNGROUNDED PREDICTION",
            "input": "AI will achieve superintelligence and replace all jobs by June 2026",
            "evidence": [
                "Recent AI progress",
                "Some blog posts about AI"
            ],
            "expected": "❌ HIGH severity (no model, no confidence bounds, timeline unjustified)"
        },
        
        # UNGROUNDED CAUSAL CLAIM
        {
            "category": "UNGROUNDED CAUSAL CLAIM",
            "input": "The new policy caused a 50% increase in productivity",
            "evidence": [
                "Productivity increased"
            ],
            "expected": "❌ HIGH severity (no temporal proof, no alternative explanations ruled out)"
        },
        
        # LEGAL CLAIM WITHOUT JURISDICTION
        {
            "category": "UNGROUNDED LEGAL CLAIM",
            "input": "This contract is unenforceable and violates international law",
            "evidence": [
                "I think it might be unfair"
            ],
            "expected": "❌ CRITICAL severity (no jurisdiction, no precedent, no statutory basis)"
        },
        
        # NORMATIVE CLAIM WITHOUT FRAMEWORK
        {
            "category": "VAGUE NORMATIVE CLAIM",
            "input": "Everyone should stop using technology",
            "evidence": [
                "Some people say technology is bad"
            ],
            "expected": "❌ HIGH severity (no ethical framework, no stakeholder context)"
        },
        
        # TECHNICAL CLAIM WITH PARTIAL GROUNDING
        {
            "category": "TECHNICAL CLAIM (PARTIAL)",
            "input": "The API endpoint returns a 500 error under load",
            "evidence": [
                "Logs show 500 errors at 9:15 UTC",
                "Server CPU was at 98%",
                "Database connection pool was exhausted"
            ],
            "expected": "✅ LOW-MEDIUM severity (reproducible, has mechanism)"
        },
    ]
    
    # Run tests
    print_header("BOUNDARY DETECTION DIAGNOSTIC SUITE")
    print("\nThis suite tests the boundary detector against various claim types.")
    print("It demonstrates grounding analysis and refusal threshold logic.")
    
    results_summary = {
        "total": len(test_cases),
        "would_refuse": 0,
        "would_allow": 0,
        "by_severity": {}
    }
    
    for i, test_case in enumerate(test_cases, 1):
        violation = detector.extract_boundaries(test_case['input'], test_case['evidence'])
        print_test_result(i, test_case, violation)
        
        # Tally results
        if violation['human_review_required']:
            results_summary["would_refuse"] += 1
        else:
            results_summary["would_allow"] += 1
        
        severity = violation['severity_level']
        results_summary["by_severity"][severity] = results_summary["by_severity"].get(severity, 0) + 1
    
    # Summary
    print_header("SUMMARY")
    print(f"\nTotal Tests: {results_summary['total']}")
    print(f"Would Refuse (severity >= 70): {results_summary['would_refuse']} claims")
    print(f"Would Allow (severity < 70): {results_summary['would_allow']} claims")
    print(f"\nSeverity Distribution:")
    for level, count in sorted(results_summary['by_severity'].items(), key=lambda x: -x[1]):
        print(f"  • {level.upper()}: {count} claim(s)")
    
    print("\n" + "=" * 80)
    print("  KEY INSIGHTS")
    print("=" * 80)
    print("""
1. CRITICAL severity claims (score 90+) require immediate human review
   → Example: Medical claims without clinical evidence

2. HIGH severity claims (score 70-89) trigger refusal in Standard mode
   → Example: Predictions without model specification

3. MEDIUM/LOW claims (score < 70) proceed but may warn the user
   → Example: Well-documented factual claims with proper citations

4. Grounding Score (0-100) measures how well the claim is supported
   → Higher = better supported
   → Used to compute severity

5. Missing Grounding tells you what evidence is needed
   → Helps identify information gaps
   → Guides further investigation
    """)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
