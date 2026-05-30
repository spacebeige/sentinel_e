import re

with open("src/app/components/ChatPage.tsx", "r") as f:
    content = f.read()

# 1. State Topology Redesign
# Add runtimeTier and isOrchestrationExpanded. We replace `isPlusOpen` with `isOrchestrationExpanded`.
content = content.replace(
    'const [isPlusOpen, setIsPlusOpen] = useState(false);',
    'const [runtimeTier, setRuntimeTier] = useState<"standard" | "pro">("standard");\n  const [isOrchestrationExpanded, setIsOrchestrationExpanded] = useState(false);'
)

# Replace all isPlusOpen with isOrchestrationExpanded
content = content.replace('isPlusOpen', 'isOrchestrationExpanded')
content = content.replace('setIsPlusOpen', 'setIsOrchestrationExpanded')

# Remove `isProMode` from useChatInteraction
content = content.replace('isProMode, setIsProMode', '/* removed isProMode */')
content = content.replace('isProMode: saved.isProMode ?? false,', '')
content = content.replace('setIsProMode(saved.isProMode ?? false);', 'setRuntimeTier(saved.runtimeTier || "standard");\n      setIsOrchestrationExpanded(saved.isOrchestrationExpanded || false);')

# In save/restore logic:
content = content.replace('isProMode,', 'runtimeTier, isOrchestrationExpanded,')

# Compute effective mode
content = content.replace('const availableModels = AVAILABLE_MODELS;', 'const availableModels = AVAILABLE_MODELS;\n  const effectiveMode = (runtimeTier === "pro" && isOrchestrationExpanded) ? selectedMode : "standard";')

# 2. Redesign handleSend logic
handle_send_old = """if (isProMode && selectedMode) {
          response = await runExperimental(userText, selectedMode, 6, currentChatId || undefined, glassState.killOverride, attachedFile || undefined, ac.signal);
        } else {
          response = await runStandard(userText, activeModel.id, currentChatId || undefined, attachedFile || undefined, ac.signal);
        }"""
handle_send_new = """if (runtimeTier === "pro" && isOrchestrationExpanded) {
          response = await runExperimental(userText, selectedMode, 6, currentChatId || undefined, glassState.killOverride, attachedFile || undefined, ac.signal);
        } else {
          response = await runStandard(userText, runtimeTier === "pro" ? activeModel.id : "llama-3-1-8b-instant", currentChatId || undefined, attachedFile || undefined, ac.signal);
        }"""
content = content.replace(handle_send_old, handle_send_new)

# 3. Top Navigation: Runtime Tier Controller
top_nav_old = """<div
                className="flex items-center justify-center gap-2 transition-all duration-300 pointer-events-auto"
                style={{
                  height: "40px",
                  padding: "0 16px",
                  borderRadius: "18px",
                  fontSize: "14px",
                  fontWeight: 600,
                  background: isDark ? "rgba(18,18,24,0.72)" : "rgba(255,255,255,0.72)",
                  backdropFilter: "blur(24px) saturate(180%)",
                  WebkitBackdropFilter: "blur(24px) saturate(180%)",
                  border: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.06)",
                  boxShadow: isDark
                    ? "0 10px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05)"
                    : "0 6px 20px rgba(0,0,0,0.06), inset 0 1px 0 rgba(255,255,255,0.7)",
                  color: isDark ? "#f5f5f7" : "#1d1d1f",
                }}
              >
                Sentinel {isProMode ? "Pro" : "Standard"}
              </div>"""

top_nav_new = """<div className="flex items-center justify-center gap-1 p-1 transition-all duration-300 pointer-events-auto"
                style={{
                  height: "40px",
                  borderRadius: "20px",
                  background: isDark ? "rgba(18,18,24,0.72)" : "rgba(255,255,255,0.72)",
                  backdropFilter: "blur(24px) saturate(180%)",
                  border: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.06)",
                  boxShadow: isDark ? "0 10px 30px rgba(0,0,0,0.35)" : "0 6px 20px rgba(0,0,0,0.06)",
                }}>
                <button
                  onClick={() => { setRuntimeTier("standard"); setIsOrchestrationExpanded(false); }}
                  className={`px-4 py-1 h-full rounded-full text-sm font-semibold transition-all duration-300 ${runtimeTier === "standard" ? (isDark ? "bg-white/15 text-white shadow-sm" : "bg-black/10 text-black shadow-sm") : (isDark ? "text-white/50 hover:text-white/80" : "text-black/50 hover:text-black/80")}`}
                >
                  Sentinel Standard
                </button>
                <button
                  onClick={() => { setRuntimeTier("pro"); }}
                  className={`px-4 py-1 h-full rounded-full text-sm font-semibold transition-all duration-300 ${runtimeTier === "pro" ? (isDark ? "bg-white/15 text-white shadow-sm" : "bg-black/10 text-black shadow-sm") : (isDark ? "text-white/50 hover:text-white/80" : "text-black/50 hover:text-black/80")}`}
                >
                  Sentinel Pro
                </button>
              </div>"""
content = content.replace(top_nav_old, top_nav_new)

# 4. Fix Cockpit Input Area styling (use effectiveMode)
content = content.replace('{getModeConfig(selectedMode).borderClass}', '{getModeConfig(effectiveMode).borderClass}')
content = content.replace('selectedMode && selectedMode !== "standard"', 'effectiveMode && effectiveMode !== "standard"')
content = content.replace('? getModeConfig(selectedMode).color', '? getModeConfig(effectiveMode).color')
content = content.replace('? getModeConfig(selectedMode).placeholder', '? getModeConfig(effectiveMode).placeholder')

# 5. Fix UI controls: The model dropdown and `+` should ONLY show in Pro
# In ChatPage.tsx lines 1395 to 1464:
# I need to conditionally wrap `+` button and model dropdown.
# I will just replace `isProMode` with `runtimeTier === "pro"` everywhere first.
content = content.replace('isProMode', '(runtimeTier === "pro")')

# Specifically for the model dropdown (which currently shows activeModel.name)
# It was conditionally rendered with `{isProMode && (` - wait, it already was!
content = content.replace('{/* Pro model label */}\n                {(runtimeTier === "pro") && (', '{/* Pro model label */}\n                {runtimeTier === "pro" && !isOrchestrationExpanded && (')

# And for the orchestration modes menu:
content = content.replace('{/* Modes menu */}', '{/* Modes menu */}')
# Wait, the mode menu logic uses `isPlusOpen` (now `isOrchestrationExpanded`). We just need to make sure `+` works.

with open("src/app/components/ChatPage.tsx", "w") as f:
    f.write(content)

