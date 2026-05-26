import re

with open("/Users/ashwinagarkhed/sentinel_e/figma_ui/src/app/components/ChatPage.tsx", "r") as f:
    content = f.read()

# 1. Input dock safe area padding
old_input_dock = """        {/* ── INPUT DOCK ──────────────────────────────────────────────── */}
        <div
          className="relative px-4 pb-4 pt-2"
          style={{
            background: isDark
              ? "linear-gradient(to top, #08090e 60%, transparent)"
              : "linear-gradient(to top, #ffffff 60%, transparent)",
          }}
        >"""
new_input_dock = """        {/* ── INPUT DOCK ──────────────────────────────────────────────── */}
        <div
          className="relative px-4 pb-[calc(1rem+env(safe-area-inset-bottom))] pt-2"
          style={{
            background: isDark
              ? "linear-gradient(to top, #08090e 60%, transparent)"
              : "linear-gradient(to top, #ffffff 60%, transparent)",
          }}
        >"""
content = content.replace(old_input_dock, new_input_dock)

# 2. Plus button for Pro mode
old_plus = """                  {/* Plus — opens model selector in Pro mode */}
                  <button
                    onClick={() => isProMode ? setShowModelSelector(!showModelSelector) : fileInputRef.current?.click()}
                    className="p-2 rounded-full transition-all"
                    style={{ color: showModelSelector ? "#3b82f6" : textSecondary }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.05)"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
                  >
                    <Plus className="w-5 h-5" />
                  </button>"""
new_plus = """                  {/* Plus — opens model selector in Pro mode */}
                  {isProMode && (
                    <button
                      onClick={() => setShowModelSelector(!showModelSelector)}
                      className="p-2 rounded-full transition-all"
                      style={{ color: showModelSelector ? "#3b82f6" : textSecondary }}
                      onMouseEnter={(e) => { e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.05)"; }}
                      onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
                    >
                      <Plus className="w-5 h-5 flex-shrink-0" />
                    </button>
                  )}"""
content = content.replace(old_plus, new_plus)

with open("/Users/ashwinagarkhed/sentinel_e/figma_ui/src/app/components/ChatPage.tsx", "w") as f:
    f.write(content)
print("Input dock patch applied.")
