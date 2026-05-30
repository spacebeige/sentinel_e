import re

with open("src/app/components/ChatPage.tsx", "r") as f:
    content = f.read()

# 1. Add states
state_declarations = r"const \[isProActive, setIsProActive\] = useState\(false\);"
new_state = """const [isProActive, setIsProActive] = useState(false);
  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);
  const [isModeDropdownOpen, setIsModeDropdownOpen] = useState(false);"""
content = re.sub(state_declarations, new_state, content, count=1)

# 2. Rewrite the left actions section
pattern = r"\{/\*\s*Left actions\s*\*/\}([\s\S]*?)\{/\*\s*Textarea\s*\*/\}"

new_left_actions = """{/* Left actions */}
                <div className="flex items-center gap-0.5 pb-0.5 relative" style={{ overflow: "visible" }}>
                  
                  {/* Dynamic Dropdown Panels */}
                  <AnimatePresence>
                    {isModeDropdownOpen && isProActive && (
                      <motion.div
                        key="mode-dropdown"
                        initial={{ opacity: 0, y: 8, scale: 0.96 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 8, scale: 0.96 }}
                        transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
                        className="absolute z-[140] pointer-events-auto w-[240px] p-3"
                        style={{
                          bottom: "calc(100% + 12px)",
                          left: "0",
                          background: isDark ? "rgba(18,18,24,0.78)" : "rgba(255,255,255,0.72)",
                          backdropFilter: "blur(30px) saturate(180%)",
                          WebkitBackdropFilter: "blur(30px) saturate(180%)",
                          border: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.06)",
                          borderRadius: "22px",
                          boxShadow: isDark 
                            ? "0 18px 40px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05)"
                            : "0 10px 30px rgba(0,0,0,0.08), inset 0 1px 0 rgba(255,255,255,0.7)",
                        }}
                      >
                         <div className="mb-2 px-2 text-[11px] font-semibold tracking-wider uppercase" style={{ color: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)" }}>Orchestration Modes</div>
                         <div className="flex flex-col gap-1.5">
                           {ALL_RUNTIME_MODES.filter(m => m.id !== "standard").map((m) => {
                             const isActive = selectedMode === m.id;
                             return (
                               <button
                                 key={m.id}
                                 onClick={() => { setSelectedMode(m.id); setIsModeDropdownOpen(false); }}
                                 className="flex items-center gap-2.5 p-2 rounded-xl transition-all text-left"
                                 style={{
                                   background: isActive ? m.color : (isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.03)"),
                                   border: isActive ? `1px solid ${m.color}` : "1px solid transparent",
                                   color: isActive ? "#ffffff" : (isDark ? "#fff" : "#000"),
                                 }}
                                 onMouseEnter={(e) => { if (!isActive) e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.05)"; }}
                                 onMouseLeave={(e) => { if (!isActive) e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.03)"; }}
                               >
                                 <div className="w-2.5 h-2.5 rounded-full" style={{ background: isActive ? "#fff" : m.color }} />
                                 <div className="text-[13px] font-semibold">{m.label}</div>
                               </button>
                             );
                           })}
                         </div>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  <AnimatePresence>
                    {isModelDropdownOpen && !isProActive && (
                      <motion.div
                        key="model-dropdown"
                        initial={{ opacity: 0, y: 8, scale: 0.96 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 8, scale: 0.96 }}
                        transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
                        className="absolute z-[140] pointer-events-auto w-[240px] p-3"
                        style={{
                          bottom: "calc(100% + 12px)",
                          left: "0",
                          background: isDark ? "rgba(18,18,24,0.78)" : "rgba(255,255,255,0.72)",
                          backdropFilter: "blur(30px) saturate(180%)",
                          WebkitBackdropFilter: "blur(30px) saturate(180%)",
                          border: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.06)",
                          borderRadius: "22px",
                          boxShadow: isDark 
                            ? "0 18px 40px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05)"
                            : "0 10px 30px rgba(0,0,0,0.08), inset 0 1px 0 rgba(255,255,255,0.7)",
                        }}
                      >
                         <div className="mb-2 px-2 text-[11px] font-semibold tracking-wider uppercase" style={{ color: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)" }}>Available Models</div>
                         <div className="flex flex-col gap-1.5">
                           {AVAILABLE_MODELS.map((model) => {
                             const isSelected = selectedModel === model.id;
                             return (
                               <button
                                 key={model.id}
                                 onClick={() => { setSelectedModel(model.id); setIsModelDropdownOpen(false); }}
                                 className="flex items-center gap-2 p-2 rounded-xl transition-all text-left"
                                 style={{
                                   background: isSelected ? (isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.06)") : "transparent",
                                 }}
                                 onMouseEnter={(e) => { if(!isSelected) e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)"; }}
                                 onMouseLeave={(e) => { if(!isSelected) e.currentTarget.style.background = "transparent"; }}
                               >
                                 <div className="w-2 h-2 rounded-full" style={{ background: isSelected ? "#3b82f6" : (isDark ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.2)") }} />
                                 <div className="flex-1">
                                   <div className="text-[12px] font-medium" style={{ color: isDark ? "#fff" : "#000" }}>{model.name}</div>
                                   <div className="text-[9px] font-medium opacity-50 uppercase tracking-wider">{model.provider}</div>
                                 </div>
                               </button>
                             );
                           })}
                         </div>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Pro Activation Toggle (+) */}
                  <button
                    onClick={() => {
                      const willBePro = !isProActive;
                      setIsProActive(willBePro);
                      if (willBePro) {
                        setSelectedMode("debate");
                        setIsModelDropdownOpen(false); // force close
                      } else {
                        setSelectedMode("standard");
                        setIsModeDropdownOpen(false); // force close
                      }
                    }}
                    className="flex items-center justify-center rounded-full transition-all duration-300 flex-shrink-0 relative group"
                    style={{ 
                      width: "40px",
                      height: "40px",
                      background: isProActive ? "rgba(59, 130, 246, 0.15)" : (isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.05)"),
                      color: isProActive ? "#3b82f6" : textSecondary,
                      backdropFilter: "blur(12px)",
                      WebkitBackdropFilter: "blur(12px)",
                      boxShadow: isDark ? "inset 0 1px 0 rgba(255,255,255,0.05)" : "none",
                    }}
                    onMouseEnter={(e) => { 
                      e.currentTarget.style.transform = "scale(1.04)"; 
                      if (!isProActive) e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.08)";
                    }}
                    onMouseLeave={(e) => { 
                      e.currentTarget.style.transform = "scale(1)";
                      if (!isProActive) e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.05)";
                    }}
                  >
                    <Plus className="w-5 h-5 relative z-10" />
                    {isProActive && (
                      <div className="absolute inset-0 rounded-full animate-pulse-slow" style={{ background: `radial-gradient(circle at center, #3b82f640, transparent 70%)` }} />
                    )}
                  </button>

                  {/* Paperclip */}
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className={`p-2 rounded-full transition-all ${attachedFile ? "text-[#3b82f6]" : ""}`}
                    style={{ color: attachedFile ? "#3b82f6" : textSecondary }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.05)"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
                  >
                    <Paperclip className="w-4.5 h-4.5" />
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    className="hidden"
                    onChange={handleFileSelect}
                    accept=".txt,.pdf,.md,.json,.csv,.py,.js,.ts,.jsx,.tsx"
                  />

                  {/* Dynamic Selector Dropdown Trigger */}
                  <button
                    onClick={() => {
                      if (isProActive) {
                        setIsModeDropdownOpen(!isModeDropdownOpen);
                      } else {
                        setIsModelDropdownOpen(!isModelDropdownOpen);
                      }
                    }}
                    className="hidden sm:flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl mb-0.5 flex-shrink-0 transition-all"
                    style={{
                      background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.04)",
                      border: `1px solid ${borderColor}`,
                      color: isProActive ? getModeConfig(selectedMode).color : activeModel.color,
                    }}
                  >
                    <div className="w-2 h-2 rounded-full" style={{ background: isProActive ? getModeConfig(selectedMode).color : activeModel.color }} />
                    <span style={{ fontSize: "11px", fontWeight: 600 }}>
                      {isProActive ? getModeConfig(selectedMode).label : activeModel.name}
                    </span>
                    <ChevronDown className="w-3 h-3" style={{ color: textSecondary }} />
                  </button>
                </div>

                {/* Textarea */}"""

content = re.sub(pattern, new_left_actions, content, count=1)

# Fix click-outside handler at the end of the file
click_outside_pattern = r"\{\(modeDropdownOpen \|\| isDropdownOpen\)\s*&&\s*\([\s\S]*?\}\s*\)\s*\}"
new_click_outside = """{(modeDropdownOpen || isModelDropdownOpen || isModeDropdownOpen) && (
        <div
          className="fixed inset-0 z-10"
          onClick={() => { setModeDropdownOpen(false); setIsModelDropdownOpen(false); setIsModeDropdownOpen(false); }}
        />
      )}"""
content = re.sub(click_outside_pattern, new_click_outside, content, count=1)

# Any other stray isDropdownOpen traces? Let's just catch them if they exist
content = content.replace("isDropdownOpen", "isModelDropdownOpen") # fallback for any misses

with open("src/app/components/ChatPage.tsx", "w") as f:
    f.write(content)

