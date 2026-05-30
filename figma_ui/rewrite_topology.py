import re

with open("src/app/components/ChatPage.tsx", "r") as f:
    content = f.read()

# 1. Update State Declarations
state_pattern = r"const \[isProMode, setIsProMode\] = useState\(false\);"
new_states = """const [runtimeTier, setRuntimeTier] = useState<"standard" | "pro">("standard");
  const [isOrchestrationExpanded, setIsOrchestrationExpanded] = useState(false);
  const isProMode = runtimeTier === "pro";"""
content = re.sub(state_pattern, new_states, content, count=1)

# 2. Update debug logging to include new states
debug_pattern = r"console\.log\(\"DEBUG interaction states:\", \{ isModelDropdownOpen, isModeDropdownOpen, isProMode, selectedMode \}\);"
new_debug = """console.log("DEBUG interaction states:", { isModelDropdownOpen, isModeDropdownOpen, runtimeTier, isOrchestrationExpanded, selectedMode });"""
content = re.sub(debug_pattern, new_debug, content)

# 3. Update State Restoration Gating (ChatPage.tsx:265)
restore_pattern = r"if \(restoredMode !== \"standard\"\) \{\n\s*setIsProMode\(true\);\n\s*\}"
new_restore = """if (restoredMode !== "standard") {
          setRuntimeTier("pro");
          setIsOrchestrationExpanded(true);
        }"""
content = re.sub(restore_pattern, new_restore, content)

# 4. Remove old mode toggle logic in Top Nav (around line 988)
# The current center block:
top_nav_pattern = r"\{/\* CENTER \*/\}.*?\{/\* RIGHT \*/\}"
new_top_nav = """{/* CENTER */}
          <div className="absolute left-1/2 -translate-x-1/2 flex items-center gap-1 p-1 z-50 rounded-full" 
               style={{ 
                 background: isDark ? "rgba(18,18,24,0.72)" : "rgba(255,255,255,0.72)",
                 backdropFilter: "blur(24px) saturate(180%)",
                 border: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.06)",
                 boxShadow: isDark ? "0 10px 30px rgba(0,0,0,0.35)" : "0 6px 20px rgba(0,0,0,0.06)",
               }}>
            <button
              onClick={() => { setRuntimeTier("standard"); setIsOrchestrationExpanded(false); setIsModelDropdownOpen(false); setIsModeDropdownOpen(false); }}
              className={`px-4 py-1.5 rounded-full text-sm font-semibold transition-all duration-300 ${runtimeTier === "standard" ? (isDark ? "bg-white/10 text-white shadow-sm" : "bg-black/5 text-black shadow-sm") : (isDark ? "text-white/50 hover:text-white/80" : "text-black/50 hover:text-black/80")}`}
            >
              Sentinel Standard
            </button>
            <button
              onClick={() => { setRuntimeTier("pro"); }}
              className={`px-4 py-1.5 rounded-full text-sm font-semibold transition-all duration-300 ${runtimeTier === "pro" ? (isDark ? "bg-white/10 text-white shadow-sm" : "bg-black/5 text-black shadow-sm") : (isDark ? "text-white/50 hover:text-white/80" : "text-black/50 hover:text-black/80")}`}
            >
              Sentinel Pro
            </button>
          </div>

          {/* RIGHT */}"""
content = re.sub(top_nav_pattern, new_top_nav, content, flags=re.DOTALL)

# 5. Fix effectiveMode calculation
effective_mode_pattern = r"const effectiveMode = isProMode \? selectedMode : \"standard\";"
new_effective_mode = """// effectiveMode depends on both Pro Tier and Orchestration Expansion
  const effectiveMode = (runtimeTier === "pro" && isOrchestrationExpanded) ? selectedMode : "standard";"""
content = re.sub(effective_mode_pattern, new_effective_mode, content)

# 6. Hide/Show logic inside the cockpit (Input Area)
# The `+` button logic:
plus_button_pattern = r"<button[\s\S]*?onClick=\{.*?setIsProMode.*?\}[\s\S]*?>[\s\S]*?Plus[\s\S]*?</button>"

new_plus_button = """{runtimeTier === "pro" && (
                  <button
                    type="button"
                    onClick={() => {
                      const willExpand = !isOrchestrationExpanded;
                      setIsOrchestrationExpanded(willExpand);
                      if (willExpand) {
                        if (selectedMode === "standard") setSelectedMode("debate");
                        setIsModeDropdownOpen(true);
                        setIsModelDropdownOpen(false);
                      } else {
                        setSelectedMode("standard");
                        setIsModeDropdownOpen(false);
                      }
                    }}
                    className={`w-9 h-9 rounded-full flex items-center justify-center transition-all duration-300 relative group overflow-hidden ${
                      isOrchestrationExpanded
                        ? "bg-white text-black shadow-[0_0_15px_rgba(255,255,255,0.4)] hover:shadow-[0_0_20px_rgba(255,255,255,0.6)]"
                        : isDark
                        ? "bg-white/5 text-white/60 hover:bg-white/10 hover:text-white border border-white/5"
                        : "bg-black/5 text-black/60 hover:bg-black/10 hover:text-black border border-black/5"
                    }`}
                    title={isOrchestrationExpanded ? "Revert to Model Runtime" : "Advanced Orchestration Expansion Control"}
                  >
                    <Plus className={`w-5 h-5 transition-transform duration-500 ease-out ${isOrchestrationExpanded ? 'rotate-[135deg] scale-110' : 'group-hover:scale-110'}`} />
                  </button>
                )}"""
content = re.sub(plus_button_pattern, new_plus_button, content, count=1, flags=re.DOTALL)

# Update dropdowns visibility gating
# Current Model dropdown button visibility:
model_trigger_pattern = r"\{!isProMode && \([\s\S]*?isModelDropdownOpen \? \([\s\S]*?\}\)\}"
new_model_trigger = """{runtimeTier === "pro" && !isOrchestrationExpanded && (
                  <div className="relative" ref={modelTriggerRef}>
                    <button
                      type="button"
                      onClick={() => setIsModelDropdownOpen(!isModelDropdownOpen)}
                      className={`h-9 px-3 rounded-full flex items-center gap-2 transition-all duration-300 border ${
                        isModelDropdownOpen
                          ? isDark ? "bg-white/10 border-white/20 text-white" : "bg-black/5 border-black/10 text-black"
                          : isDark ? "bg-white/5 border-white/5 text-white/70 hover:bg-white/10 hover:text-white" : "bg-transparent border-black/5 text-black/70 hover:bg-black/5 hover:text-black"
                      }`}
                    >
                      <div className="w-2 h-2 rounded-full" style={{ background: activeModel.color }} />
                      <span className="text-sm font-medium">{activeModel.name}</span>
                      <ChevronDown className={`w-3 h-3 transition-transform duration-300 ${isModelDropdownOpen ? 'rotate-180' : ''}`} />
                    </button>
                  </div>
                )}"""
content = re.sub(r"\{\!isProMode && \([\s\S]*?modelTriggerRef[\s\S]*?</button>\s*</div>\s*\)\}", new_model_trigger, content)

mode_trigger_pattern = r"\{isProMode && \([\s\S]*?modeTriggerRef[\s\S]*?</button>\s*</div>\s*\)\}"
new_mode_trigger = """{runtimeTier === "pro" && isOrchestrationExpanded && (
                  <div className="relative" ref={modeTriggerRef}>
                    <button
                      type="button"
                      onClick={() => setIsModeDropdownOpen(!isModeDropdownOpen)}
                      className={`h-9 px-3 rounded-full flex items-center gap-2 transition-all duration-300 border ${
                        isModeDropdownOpen
                          ? isDark ? "bg-white/10 border-white/20 text-white" : "bg-black/5 border-black/10 text-black"
                          : isDark ? "bg-white/5 border-white/5 text-white/70 hover:bg-white/10 hover:text-white" : "bg-transparent border-black/5 text-black/70 hover:bg-black/5 hover:text-black"
                      }`}
                      style={{
                        borderColor: isOrchestrationExpanded ? `${getModeConfig(effectiveMode).color}40` : undefined,
                        color: isOrchestrationExpanded ? getModeConfig(effectiveMode).color : undefined,
                      }}
                    >
                      <div className="w-2 h-2 rounded-full" style={{ background: getModeConfig(effectiveMode).color }} />
                      <span className="text-sm font-medium">{getModeConfig(effectiveMode).label}</span>
                      <ChevronDown className={`w-3 h-3 transition-transform duration-300 ${isModeDropdownOpen ? 'rotate-180' : ''}`} />
                    </button>
                  </div>
                )}"""
content = re.sub(mode_trigger_pattern, new_mode_trigger, content)

with open("src/app/components/ChatPage.tsx", "w") as f:
    f.write(content)

