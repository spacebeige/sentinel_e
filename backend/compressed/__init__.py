"""
Sentinel-E Compressed Pipeline v2 — role-based multi-model reasoning via LangGraph.
Providers: Groq + Gemini Flash + Qwen (no OpenRouter).
~6 API calls: Analysis(Groq 70B) → Critique(Mixtral ∥ Gemma ∥ Qwen) → Synthesis(Gemini) → Verification(Groq 8B).
"""
