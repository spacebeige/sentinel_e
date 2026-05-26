import re

with open("/Users/ashwinagarkhed/sentinel_e/figma_ui/src/app/components/ChatPage.tsx", "r") as f:
    content = f.read()

# Change aside to md:relative to prevent tablet overlap
old_aside_class = "className={`absolute lg:relative left-0 top-0 h-full z-40 flex flex-col flex-shrink-0 overflow-hidden transition-all duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] ${"
new_aside_class = "className={`absolute md:relative left-0 top-0 h-full z-40 flex flex-col flex-shrink-0 overflow-hidden transition-all duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] ${"
content = content.replace(old_aside_class, new_aside_class)
# Fix translate for mobile (use md instead of lg)
content = content.replace('sidebarOpen ? "translate-x-0 w-[260px]" : "-translate-x-full lg:translate-x-0 lg:w-[68px]"', 'sidebarOpen ? "translate-x-0 w-[260px]" : "-translate-x-full md:translate-x-0 md:w-[68px]"')
content = content.replace('className="fixed inset-0 z-30 lg:hidden"', 'className="fixed inset-0 z-30 md:hidden"')

# Ensure the mode dropdown shifts smoothly (the dropdown itself is the one being clicked)
# "The Sentinel dropdown should behave like ChatGPT model selector UX"
# Let's make the dropdown standard/pro selector more robust
old_dropdown_button = """            {/* Mode dropdown */}
            <div className="relative">
              <button
                onClick={() => setModeDropdownOpen(!modeDropdownOpen)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl transition-all"
                style={{
                  background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.05)",
                  border: `1px solid ${borderColor}`,
                  color: textPrimary,
                }}
              >
                <Sparkles className="w-3.5 h-3.5 text-[#3b82f6]" />
                <span style={{ fontFamily: "'Inter', sans-serif", fontSize: "13px", fontWeight: 600 }}>
                  Sentinel
                </span>
                {isProMode && (
                  <span className="px-1.5 py-0.5 rounded-md text-[9px] font-bold tracking-wider" style={{ background: "rgba(139,92,246,0.15)", color: "#8b5cf6" }}>PRO</span>
                )}
                <ChevronDown className={`w-3.5 h-3.5 transition-transform ${modeDropdownOpen ? "rotate-180" : ""}`} style={{ color: textSecondary }} />
              </button>"""

new_dropdown_button = """            {/* Mode dropdown */}
            <div className="relative">
              <button
                onClick={() => setModeDropdownOpen(!modeDropdownOpen)}
                className="group flex items-center gap-2 px-3 py-1.5 rounded-xl transition-all duration-300 hover:shadow-sm"
                style={{
                  background: isDark ? "rgba(255,255,255,0.04)" : "transparent",
                  border: isDark ? `1px solid rgba(255,255,255,0.08)` : `1px solid transparent`,
                  color: textPrimary,
                }}
              >
                <span className="font-semibold text-[15px] flex items-center gap-2" style={{ fontFamily: "'Inter', sans-serif" }}>
                  Sentinel {isProMode ? <span className="text-[#3b82f6]">Pro</span> : <span className="text-gray-500">Standard</span>}
                </span>
                <ChevronDown className={`w-4 h-4 transition-transform duration-300 ${modeDropdownOpen ? "rotate-180" : ""}`} style={{ color: textSecondary }} />
              </button>"""
content = content.replace(old_dropdown_button, new_dropdown_button)

old_dropdown_menu = """              <AnimatePresence>
                {modeDropdownOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: -6, scale: 0.97 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -6, scale: 0.97 }}
                    transition={{ duration: 0.15, ease: [0.16, 1, 0.3, 1] }}
                    className="absolute left-0 top-full mt-1.5 w-52 rounded-2xl overflow-hidden z-50"
                    style={{
                      background: isDark ? "#0f1117" : "#ffffff",
                      border: `1px solid ${borderColor}`,
                      boxShadow: isDark ? "0 16px 40px rgba(0,0,0,0.5)" : "0 16px 40px rgba(0,0,0,0.1)",
                    }}
                  >
                    <div className="p-1.5">
                      {[
                        { id: "standard", label: "Standard", sub: "Simple AI experience", pro: false },
                        { id: "pro", label: "Pro", sub: "Full orchestration · multi-model", pro: true },
                      ].map((m) => (
                        <button
                          key={m.id}
                          onClick={() => { setIsProMode(m.pro); setModeDropdownOpen(false); if (!m.pro) setActiveSubMode(null); }}
                          className="w-full text-left px-3 py-2 rounded-xl transition-colors flex items-center justify-between"
                          onMouseEnter={(e) => { e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)"; }}
                          onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
                        >
                          <div>
                            <div style={{ fontSize: "13px", fontWeight: 600, color: textPrimary }}>{m.label}</div>
                            <div style={{ fontSize: "10px", color: textSecondary, marginTop: "2px" }}>{m.sub}</div>
                          </div>
                          {isProMode === m.pro && <Check className="w-3.5 h-3.5 text-[#3b82f6]" />}
                        </button>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>"""

new_dropdown_menu = """              <AnimatePresence>
                {modeDropdownOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: -6, scale: 0.97 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -6, scale: 0.97 }}
                    transition={{ duration: 0.15, ease: [0.16, 1, 0.3, 1] }}
                    className="absolute left-0 top-full mt-2 w-[260px] rounded-2xl overflow-hidden z-50"
                    style={{
                      background: isDark ? "#1a1d26" : "#ffffff",
                      border: isDark ? "1px solid rgba(255,255,255,0.1)" : "1px solid rgba(0,0,0,0.08)",
                      boxShadow: isDark ? "0 12px 40px rgba(0,0,0,0.4)" : "0 12px 40px rgba(0,0,0,0.12)",
                    }}
                  >
                    <div className="p-2 space-y-1">
                      {[
                        { id: "standard", label: "Sentinel Standard", sub: "Simple AI experience", pro: false },
                        { id: "pro", label: "Sentinel Pro", sub: "Full orchestration · multi-model", pro: true },
                      ].map((m) => (
                        <button
                          key={m.id}
                          onClick={() => { setIsProMode(m.pro); setModeDropdownOpen(false); if (!m.pro) setActiveSubMode(null); }}
                          className="w-full text-left px-3 py-3 rounded-xl transition-all duration-200 flex items-center justify-between group"
                          style={{
                            background: isProMode === m.pro ? (isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.05)") : "transparent"
                          }}
                          onMouseEnter={(e) => { if(isProMode !== m.pro) e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.02)"; }}
                          onMouseLeave={(e) => { if(isProMode !== m.pro) e.currentTarget.style.background = "transparent"; }}
                        >
                          <div>
                            <div style={{ fontSize: "14px", fontWeight: 600, color: textPrimary }}>{m.label}</div>
                            <div style={{ fontSize: "12px", color: textSecondary, marginTop: "2px" }}>{m.sub}</div>
                          </div>
                          {isProMode === m.pro && <Check className="w-4 h-4 text-[#3b82f6]" />}
                        </button>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>"""

content = content.replace(old_dropdown_menu, new_dropdown_menu)

with open("/Users/ashwinagarkhed/sentinel_e/figma_ui/src/app/components/ChatPage.tsx", "w") as f:
    f.write(content)

print("Dropdown and mobile overlap fixed")
