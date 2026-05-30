import re

with open("src/app/components/ChatPage.tsx", "r") as f:
    content = f.read()

# Add sanitization logic
sanitization_logic = """
    // Sanitize stale local storage
    if (parsed.selectedModel && !getModelConfig(parsed.selectedModel)) {
      parsed.selectedModel = getModelConfig().id;
    }
    if (parsed.selectedMode && !getModeConfig(parsed.selectedMode)) {
      parsed.selectedMode = getModeConfig().id;
    }
"""

# Let's just fix it where setUiState is called from persist
content = content.replace(
    'if (parsed.selectedMode) setSelectedMode(parsed.selectedMode);',
    'if (parsed.selectedMode) { const safeMode = getModeConfig(parsed.selectedMode); setSelectedMode(safeMode.id !== "default" ? safeMode.id : "standard"); }'
)
content = content.replace(
    'if (parsed.selectedModel) setSelectedModel(parsed.selectedModel);',
    'if (parsed.selectedModel) { const safeModel = getModelConfig(parsed.selectedModel); setSelectedModel(safeModel.id !== "default" ? safeModel.id : AVAILABLE_MODELS[0].id); }'
)

# And add the debug log as requested by user
debug_log = """
  useEffect(() => {
    console.log({
      runtimeTier,
      selectedModel,
      selectedMode,
      resolvedModel: getModelConfig(selectedModel),
      resolvedMode: getModeConfig(selectedMode),
    });
  }, [runtimeTier, selectedModel, selectedMode]);
"""

content = content.replace(
    'const [isOrchestrationExpanded, setIsOrchestrationExpanded] = useState(false);',
    'const [isOrchestrationExpanded, setIsOrchestrationExpanded] = useState(false);\n' + debug_log
)

with open("src/app/components/ChatPage.tsx", "w") as f:
    f.write(content)
