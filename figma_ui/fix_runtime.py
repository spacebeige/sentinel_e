import re

with open("src/app/config/runtime.ts", "r") as f:
    content = f.read()

# Make sure DEFAULT_MODEL_CONFIG is defined
if "DEFAULT_MODEL_CONFIG" not in content:
    default_model = """
export const DEFAULT_MODEL_CONFIG: ModelDefinition = {
  id: "default",
  name: "Sentinel Default",
  category: "General",
  description: "Fallback default model",
  capabilities: "General capabilities",
  provider: "Unknown",
  color: FALLBACK_MODEL_COLOR
};

export function getModelConfig(modelId?: string | null): ModelDefinition {
  if (!modelId) return DEFAULT_MODEL_CONFIG;
  return MODELS.find(m => m.id === modelId) ?? DEFAULT_MODEL_CONFIG;
}
"""
    # Insert right after MODELS definition
    content = content.replace("export const FALLBACK_MODEL_COLOR", default_model + "\nexport const FALLBACK_MODEL_COLOR")

with open("src/app/config/runtime.ts", "w") as f:
    f.write(content)
