import re

with open("src/app/components/ChatPage.tsx", "r") as f:
    content = f.read()

# 1. Rename isProActive to isProMode
content = content.replace("isProActive", "isProMode")
content = content.replace("setIsProActive", "setIsProMode")

# 2. Remove the old modeDropdownOpen states and refs (lines ~184-210)
# We can do this with regex replacing the block
state_pattern = r"const \[modeDropdownOpen, setModeDropdownOpen\] = useState\(false\);\n\s*const \[modeDropdownCoords, setModeDropdownCoords\] = useState\(\{ top: 0, left: 0 \}\);\n\s*const modeTriggerRef = useRef<HTMLButtonElement>\(null\);\n\n\s*useEffect\(\(\) => \{\n[\s\S]*?\}, \[modeDropdownOpen\]\);"
content = re.sub(state_pattern, "", content)

# 3. Replace the Center Nav block entirely
nav_pattern = r"\{\/\*\s*CENTER\s*\*\/\}([\s\S]*?)\{\/\*\s*RIGHT\s*\*\/\}"
new_nav = """{/* CENTER */}
          <div className="absolute left-1/2 -translate-x-1/2 flex items-center gap-3 z-50">
            <div className="relative">
              <div
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
                Sentinel {getModeConfig(selectedMode).isExperimental ? "Pro" : "Standard"}
              </div>
            </div>
          </div>

          {/* RIGHT */}"""
content = re.sub(nav_pattern, new_nav, content, count=1)

# 4. Replace stale setIsDropdownOpen at line 275 and others
content = content.replace("setIsDropdownOpen(false)", "setIsModelDropdownOpen(false); setIsModeDropdownOpen(false);")

# 5. Fix scroll container click handler
# The scroll container was `onClick={() => { setModeDropdownOpen(false); setIsDropdownOpen(false); }}`
content = content.replace("setModeDropdownOpen(false); setIsModelDropdownOpen(false); setIsModeDropdownOpen(false);", "setIsModelDropdownOpen(false); setIsModeDropdownOpen(false);")
content = content.replace("setModeDropdownOpen(false);", "")

# 6. Fix Click outside to close dropdowns block
click_outside_pattern = r"\{\(modeDropdownOpen \|\| isModelDropdownOpen \|\| isModeDropdownOpen\)\s*&&\s*\([\s\S]*?\}\s*\)\s*\}"
new_click_outside = """{(isModelDropdownOpen || isModeDropdownOpen) && (
        <div
          className="fixed inset-0 z-10"
          onClick={() => { setIsModelDropdownOpen(false); setIsModeDropdownOpen(false); }}
        />
      )}"""
content = re.sub(click_outside_pattern, new_click_outside, content, count=1)

# 7. Add debug log inside component body
debug_log = """
  useEffect(() => {
    console.log({
      isModelDropdownOpen,
      isModeDropdownOpen,
      isProMode,
      selectedMode,
    });
  }, [isModelDropdownOpen, isModeDropdownOpen, isProMode, selectedMode]);
"""
# insert right after `const [isProMode, setIsProMode] = useState(false);` etc.
state_vars_pattern = r"const \[isModeDropdownOpen, setIsModeDropdownOpen\] = useState\(false\);"
content = re.sub(state_vars_pattern, f"const [isModeDropdownOpen, setIsModeDropdownOpen] = useState(false);\n{debug_log}", content, count=1)


with open("src/app/components/ChatPage.tsx", "w") as f:
    f.write(content)

