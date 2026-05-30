with open("src/app/config/runtime.ts", "a") as f:
    f.write("""

export const ORCHESTRATION_MODE_MAP: Record<string, { endpoint: string, mode: string, orchestration: boolean }> = {
  debate: {
    endpoint: "/api/mco/run",
    mode: "debate",
    orchestration: true,
  },
  glass: {
    endpoint: "/api/mco/run",
    mode: "glass",
    orchestration: true,
  },
  evidence: {
    endpoint: "/api/mco/run",
    mode: "evidence",
    orchestration: true,
  },
  synthesis: {
    endpoint: "/api/mco/run",
    mode: "synthesis",
    orchestration: true,
  },
};

export const MODEL_RUNTIME_MAP: Record<string, { provider: string, model: string }> = {
  "llama-3-3-70b": {
    provider: "groq",
    model: "llama-3.3-70b-versatile",
  },
  "gemini-flash-2-0": {
    provider: "google",
    model: "gemini-2.0-flash",
  },
  "deepseek": {
    provider: "deepseek",
    model: "deepseek-chat",
  },
  "qwen3-32b": {
    provider: "alibaba",
    model: "qwen3-32b",
  },
  "llama-4-scout-17b": {
    provider: "meta",
    model: "llama-4-scout-17b",
  },
  "qwen-2-5-vl-7b": {
    provider: "alibaba",
    model: "qwen-2.5-vl-7b",
  },
  "llama-3-1-8b-instant": {
    provider: "groq",
    model: "llama-3.1-8b-instant",
  },
  "mistral-large-3-675b": {
    provider: "mistral",
    model: "mistral-large-3-675b",
  }
};
""")
