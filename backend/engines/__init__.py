"""
Sentinel-E Engine Modules — Production Cognitive Governance System

engines/
  mode_controller.py      — Centralized mode routing with trigger word detection
  aggregation_engine.py   — Parallel 3-model aggregation (Standard mode)
  forensic_evidence_engine.py — 5-phase triangular forensic pipeline (Evidence mode)
  blind_audit_engine.py   — Blind cross-model structural audit (Glass mode)
  dynamic_boundary.py     — Computed boundary metrics (replaces hardcoded values)

No mode logic in UI layer. UI only renders results.
"""
