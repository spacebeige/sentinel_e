export interface ModelDefinition {
  id: string;
  name: string;
  category: string;
  description: string;
  capabilities: string;
  provider: string;
  color: string;
}

export interface OrchestrationMode {
  id: string;
  label: string;
  description: string;
  color: string;
  placeholder: string;
  icon?: string;
  orchestrationType?: string;
  isExperimental?: boolean;
  borderClass?: string;
  showProIndicator?: boolean;
}

export interface ProActivationRules {
  requiresPro: boolean;
  disablesModels: boolean;
  enablesOrchestration: boolean;
}

// ── FALLBACK CONSTANTS MUST BE DECLARED FIRST ──────────────────────────────
export const FALLBACK_MODEL_COLOR = "#8b5cf6";
export const FALLBACK_MODE_COLOR = "#8b5cf6";
export const FALLBACK_EVIDENCE_COLOR = "#06b6d4";
export const FALLBACK_TIMELINE_COLOR = "#10b981";


// ── RUNTIME MAPS & CONFIGS ───────────────────────────────────────────────
export const MODELS: ModelDefinition[] = [
  {
    id: "llama-3-3-70b",
    name: "Llama 3.3 70B",
    category: "Conceptual",
    description: "Deep conceptual reasoning with large context support.",
    capabilities: "Concept analysis · code reasoning · efficient inference",
    provider: "Meta",
    color: "#0ea5e9"
  },
  {
    id: "qwen3-32b",
    name: "Qwen3 32B",
    category: "Conceptual",
    description: "Strong structured reasoning and contextual understanding.",
    capabilities: "Long-context reasoning · coding · semantic analysis",
    provider: "Alibaba",
    color: "#8b5cf6"
  },
  {
    id: "llama-4-scout-17b",
    name: "Llama 4 Scout 17B",
    category: "General",
    description: "Balanced multi-domain reasoning model.",
    capabilities: "General reasoning · synthesis · versatile output",
    provider: "Meta",
    color: "#3b82f6"
  },
  {
    id: "qwen-2-5-vl-7b",
    name: "Qwen 2.5 VL 7B",
    category: "Multimodal",
    description: "Multimodal reasoning with visual understanding.",
    capabilities: "Vision parsing · image analysis · multilingual support",
    provider: "Alibaba",
    color: "#ec4899"
  },
  {
    id: "gemini-flash-2-0",
    name: "Gemini Flash 2.0",
    category: "General",
    description: "Fast general-purpose reasoning with huge context support.",
    capabilities: "Large context · fast inference · balanced reasoning",
    provider: "Google",
    color: "#10b981"
  },
  {
    id: "llama-3-1-8b-instant",
    name: "Llama 3.1 8B Instant",
    category: "Speed",
    description: "Ultra-fast lightweight reasoning model.",
    capabilities: "Fast responses · quick verification · low latency",
    provider: "Meta",
    color: "#f59e0b"
  },
  {
    id: "mistral-large-675b",
    name: "Mistral Large 3 675B",
    category: "Conceptual",
    description: "Large-scale conceptual and analytical reasoning model.",
    capabilities: "Deep analysis · coding · long-context reasoning",
    provider: "Mistral",
    color: "#f43f5e"
  },
  {
    id: "kimi-k2-thinking",
    name: "Kimi K2 Thinking",
    category: "Conceptual",
    description: "Deep analytical reasoning through NVIDIA-hosted Moonshot Kimi.",
    capabilities: "Long-context reasoning · critique · complex analysis",
    provider: "Moonshot AI",
    color: "#14b8a6"
  }
];

export const DEFAULT_MODEL_CONFIG: ModelDefinition = {
  id: "default",
  name: "Sentinel Default",
  category: "General",
  description: "Fallback default model",
  capabilities: "General capabilities",
  provider: "Unknown",
  color: FALLBACK_MODEL_COLOR
};

export const DEFAULT_MODE_CONFIG: OrchestrationMode = {
  id: "standard",
  label: "Standard",
  description: "Standard orchestration mode",
  color: FALLBACK_MODE_COLOR,
  placeholder: "Ask Sentinel-E anything...",
  icon: "sparkles",
  orchestrationType: "standard",
  isExperimental: false,
  borderClass: "border-white/10",
  showProIndicator: false,
};

export const ALL_RUNTIME_MODES: OrchestrationMode[] = [
  DEFAULT_MODE_CONFIG,
  {
    id: "pro",
    label: "Pro",
    description: "Full MCO orchestration across active runtime pipelines",
    color: "#8b5cf6",
    placeholder: "Run a full Sentinel-E orchestration...",
    icon: "sparkles",
    orchestrationType: "experimental",
    isExperimental: true,
    borderClass: "border-violet-500/40",
    showProIndicator: true,
  },
  {
    id: "debate",
    label: "Debate",
    description: "Argues both sides so you can decide",
    color: "#ef4444",
    placeholder: "Give me a topic to debate...",
    icon: "swords",
    orchestrationType: "experimental",
    isExperimental: true,
    borderClass: "border-red-500/40",
    showProIndicator: true,
  },
  {
    id: "glass",
    label: "Glass",
    description: "Full reasoning chain — nothing hidden",
    color: "#8b5cf6",
    placeholder: "Ask and I'll show my thinking...",
    icon: "gem",
    orchestrationType: "experimental",
    isExperimental: true,
    borderClass: "border-purple-500/40",
    showProIndicator: true,
  },
  {
    id: "evidence",
    label: "Evidence",
    description: "Every claim backed by a cited source",
    color: FALLBACK_EVIDENCE_COLOR,
    placeholder: "What do you need evidence for...",
    icon: "file-search",
    orchestrationType: "experimental",
    isExperimental: true,
    borderClass: "border-cyan-500/40",
    showProIndicator: true,
  },
  {
    id: "synthesis",
    label: "Synthesis",
    description: "Synthesizes complex topics",
    color: FALLBACK_TIMELINE_COLOR,
    placeholder: "Provide topics to synthesize...",
    icon: "git-merge",
    orchestrationType: "experimental",
    isExperimental: true,
    borderClass: "border-emerald-500/40",
    showProIndicator: true,
  }
];


// ── RESOLVER FUNCTIONS ───────────────────────────────────────────────────

export function getModelConfig(modelId?: string | null): ModelDefinition {
  if (!modelId) return DEFAULT_MODEL_CONFIG;
  return MODELS.find(m => m.id === modelId) ?? DEFAULT_MODEL_CONFIG;
}

export function getModeConfig(modeId?: string | null): OrchestrationMode {
  if (!modeId) return DEFAULT_MODE_CONFIG;
  return ALL_RUNTIME_MODES.find(m => m.id === modeId) ?? DEFAULT_MODE_CONFIG;
}

export const PRO_ACTIVATION_RULES: ProActivationRules = {
  requiresPro: true,
  disablesModels: true,
  enablesOrchestration: true,
};

export const ORCHESTRATION_MODE_MAP: Record<string, { endpoint: string, mode: string, orchestration: boolean }> = {
  pro: {
    endpoint: "/api/mco/run",
    mode: "pro",
    orchestration: true,
  },
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
    model: "llama33-70b",
  },
  "gemini-flash-2-0": {
    provider: "google",
    model: "gemini-flash",
  },
  "qwen3-32b": {
    provider: "groq",
    model: "mixtral-8x7b",
  },
  "llama-4-scout-17b": {
    provider: "groq",
    model: "llama4-scout",
  },
  "qwen-2-5-vl-7b": {
    provider: "qwen",
    model: "qwen-2.5-vl",
  },
  "llama-3-1-8b-instant": {
    provider: "groq",
    model: "llama31-8b",
  },
  "mistral-large-675b": {
    provider: "nvidia",
    model: "mistral-large-675b",
  },
  "kimi-k2-thinking": {
    provider: "nvidia",
    model: "kimi-k2-thinking",
  }
};

export function resolveFrontendModelId(runtimeModelId?: string | null): string | null {
  if (!runtimeModelId) return null;

  const directMatch = MODELS.find((model) => model.id === runtimeModelId);
  if (directMatch) return directMatch.id;

  const mappedEntry = Object.entries(MODEL_RUNTIME_MAP).find(([, value]) => value.model === runtimeModelId);
  return mappedEntry ? mappedEntry[0] : null;
}
