import re

with open("src/app/components/ChatPage.tsx", "r") as f:
    content = f.read()

# 1. State Restoration Gating (ChatPage.tsx:265)
restore_pattern = r"const saved = restore\(\);\n\s*if \(saved\.chatId\) \{\n\s*setCurrentChatId\(saved\.chatId\);\n\s*if \(saved\.mode\) setSelectedMode\(saved\.subMode \?\? saved\.mode\);\n\s*\}"
new_restore = """const saved = restore();
    if (saved.chatId) {
      setCurrentChatId(saved.chatId);
      if (saved.mode) {
        const restoredMode = saved.subMode ?? saved.mode;
        setSelectedMode(restoredMode);
        // Force Pro mode true if the restored mode is an experimental orchestration mode
        if (restoredMode !== "standard") {
          setIsProMode(true);
        }
      }
    }"""
content = re.sub(restore_pattern, new_restore, content, count=1)


# 2. Add effectiveMode near the top of the component render (around activeModel)
active_model_pattern = r"const activeModel = AVAILABLE_MODELS\.find\(m => m\.id === selectedModel\) \|\| AVAILABLE_MODELS\[0\];"
new_active_model = """const activeModel = AVAILABLE_MODELS.find(m => m.id === selectedModel) || AVAILABLE_MODELS[0];
  // Calculate effective mode: if Pro is off, we strictly render and execute as 'standard'
  const effectiveMode = isProMode ? selectedMode : "standard";"""
content = re.sub(active_model_pattern, new_active_model, content, count=1)


# 3. Fix payload execution in handleSend and generateFallbackResponse (replace selectedMode with effectiveMode inside the function)
# We can't safely use string replace for selectedMode because it's used in state setters (setSelectedMode). 
# But we can replace specific instances.
# In handleSend, we have: const activeConfig = getModeConfig(selectedMode);
content = content.replace("const activeConfig = getModeConfig(selectedMode);", "const activeConfig = getModeConfig(effectiveMode);")
content = content.replace("console.log({ selectedMode, payloadMode: activeConfig.orchestrationType, payloadSubMode: activeConfig.id });", "console.log({ effectiveMode, payloadMode: activeConfig.orchestrationType, payloadSubMode: activeConfig.id });")
content = content.replace("if (selectedMode === \"debate\" && response.omega_metadata) setDebateState((p) => mergeDebateResult(p, response.omega_metadata));", "if (effectiveMode === \"debate\" && response.omega_metadata) setDebateState((p) => mergeDebateResult(p, response.omega_metadata));")
content = content.replace("if (selectedMode === \"glass\" && response.omega_metadata) setGlassState((p) => mergeGlassState(p, response.omega_metadata));", "if (effectiveMode === \"glass\" && response.omega_metadata) setGlassState((p) => mergeGlassState(p, response.omega_metadata));")
content = content.replace("if (selectedMode === \"evidence\" && response.omega_metadata) setEvidenceState((p) => mergeEvidenceState(p, response.omega_metadata));", "if (effectiveMode === \"evidence\" && response.omega_metadata) setEvidenceState((p) => mergeEvidenceState(p, response.omega_metadata));")
content = content.replace("mode: response.sub_mode || selectedMode,", "mode: response.sub_mode || effectiveMode,")

content = content.replace("const currentMode = selectedMode;", "const currentMode = effectiveMode;")
content = content.replace("mode: selectedMode,", "mode: effectiveMode,") # This handles fallback message generation

# 4. Fix visual styles in the UI (Input Area)
content = content.replace("border ${getModeConfig(selectedMode).borderClass}", "border ${getModeConfig(effectiveMode).borderClass}")
content = content.replace("boxShadow: selectedMode && selectedMode !== \"standard\"", "boxShadow: effectiveMode && effectiveMode !== \"standard\"")
content = content.replace("4px ${getModeConfig(selectedMode).color}", "4px ${getModeConfig(effectiveMode).color}")
content = content.replace("placeholder={selectedMode && selectedMode !== \"standard\"", "placeholder={effectiveMode && effectiveMode !== \"standard\"")
content = content.replace("? getModeConfig(selectedMode).placeholder", "? getModeConfig(effectiveMode).placeholder")
content = content.replace(": selectedMode && selectedMode !== \"standard\"", ": effectiveMode && effectiveMode !== \"standard\"")
content = content.replace("? getModeConfig(selectedMode).color", "? getModeConfig(effectiveMode).color")
content = content.replace("{getModeConfig(selectedMode).isExperimental ? \"Pro\" : \"Standard\"}", "{getModeConfig(effectiveMode).isExperimental ? \"Pro\" : \"Standard\"}")

# 5. Dropdown trigger color references
content = content.replace("color: isProMode ? getModeConfig(selectedMode).color : activeModel.color,", "color: isProMode ? getModeConfig(effectiveMode).color : activeModel.color,")
content = content.replace("background: isProMode ? getModeConfig(selectedMode).color : activeModel.color", "background: isProMode ? getModeConfig(effectiveMode).color : activeModel.color")
content = content.replace("{isProMode ? getModeConfig(selectedMode).label : activeModel.name}", "{isProMode ? getModeConfig(effectiveMode).label : activeModel.name}")

# 6. Ensure Session Panel / Mode Badge in chat uses effectiveMode
content = content.replace("mode={getModeConfig(selectedMode).orchestrationType || \"standard\"}", "mode={getModeConfig(effectiveMode).orchestrationType || \"standard\"}")
content = content.replace("subMode={selectedMode}", "subMode={effectiveMode}")

with open("src/app/components/ChatPage.tsx", "w") as f:
    f.write(content)

