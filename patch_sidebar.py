import re

with open("/Users/ashwinagarkhed/sentinel_e/figma_ui/src/app/components/ChatPage.tsx", "r") as f:
    content = f.read()

# 1. Remove the AnimatePresence and {sidebarOpen && ( around motion.aside
# Line 629: <AnimatePresence initial={false}>
# Line 630: {sidebarOpen && (
# ...
# Line 805: )}
# Line 806: </AnimatePresence>

content = content.replace("<AnimatePresence initial={false}>\n        {sidebarOpen && (", "")
content = content.replace("          </motion.aside>\n        )}\n      </AnimatePresence>", "          </aside>")

# 2. Change motion.aside to a standard aside with dynamic tailwind width
old_aside = """          <motion.aside
            initial={{ x: -280, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -280, opacity: 0 }}
            transition={{ type: "spring", damping: 28, stiffness: 260 }}
            className="absolute lg:relative left-0 top-0 h-full z-40 flex flex-col flex-shrink-0 overflow-hidden"
            style={{
              width: "260px",
              background: sidebarBg,
              borderRight: `1px solid ${borderColor}`,
            }}
          >"""
new_aside = """          <aside
            className={`absolute lg:relative left-0 top-0 h-full z-40 flex flex-col flex-shrink-0 overflow-hidden transition-all duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] ${
              sidebarOpen ? "translate-x-0 w-[260px]" : "-translate-x-full lg:translate-x-0 lg:w-[68px]"
            }`}
            style={{
              background: sidebarBg,
              borderRight: `1px solid ${borderColor}`,
            }}
          >"""
content = content.replace(old_aside, new_aside)

# 3. Sidebar header - hide text if closed
old_header = """              <span
                style={{
                  fontFamily: "'Inter', sans-serif",
                  fontSize: "13px",
                  fontWeight: 600,
                  color: textPrimary,
                  letterSpacing: "-0.01em",
                }}
              >
                Sentinel-E
              </span>"""
new_header = """              {sidebarOpen ? (
                <span
                  style={{
                    fontFamily: "'Inter', sans-serif",
                    fontSize: "13px",
                    fontWeight: 600,
                    color: textPrimary,
                    letterSpacing: "-0.01em",
                  }}
                >
                  Sentinel-E
                </span>
              ) : (
                <div className="w-6 h-6 flex-shrink-0 rounded-lg bg-gradient-to-br from-[#3b82f6] to-[#06b6d4] flex items-center justify-center">
                  <span className="text-white text-[10px] font-bold">S</span>
                </div>
              )}"""
content = content.replace(old_header, new_header)

# 4. Search and New Chat button - conditionally hide text
old_search = """            {/* Search */}
            <div className="px-3 py-2.5">
              <div
                className="flex items-center gap-2 px-3 py-2 rounded-xl"
                style={{ background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)" }}
              >
                <Search className="w-3.5 h-3.5 flex-shrink-0" style={{ color: textSecondary }} />
                <input
                  type="text"
                  placeholder="Search chats..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="flex-1 bg-transparent outline-none"
                  style={{
                    fontFamily: "'Inter', sans-serif",
                    fontSize: "13px",
                    fontWeight: 400,
                    color: textPrimary,
                  }}
                />
                {searchQuery && (
                  <button onClick={() => setSearchQuery("")}>
                    <X className="w-3 h-3" style={{ color: textSecondary }} />
                  </button>
                )}
              </div>
            </div>"""
new_search = """            {/* Search */}
            <div className="px-3 py-2.5">
              <button
                onClick={() => !sidebarOpen && setSidebarOpen(true)}
                className={`flex items-center gap-2 rounded-xl transition-all ${sidebarOpen ? 'px-3 py-2' : 'p-2.5 justify-center'}`}
                style={{ background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)", width: "100%" }}
              >
                <Search className="w-3.5 h-3.5 flex-shrink-0" style={{ color: textSecondary }} />
                {sidebarOpen && (
                  <>
                    <input
                      type="text"
                      placeholder="Search chats..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="flex-1 bg-transparent outline-none"
                      style={{
                        fontFamily: "'Inter', sans-serif",
                        fontSize: "13px",
                        fontWeight: 400,
                        color: textPrimary,
                      }}
                    />
                    {searchQuery && (
                      <div onClick={(e) => { e.stopPropagation(); setSearchQuery(""); }} className="cursor-pointer">
                        <X className="w-3 h-3" style={{ color: textSecondary }} />
                      </div>
                    )}
                  </>
                )}
              </button>
            </div>"""
content = content.replace(old_search, new_search)

old_new_chat = """            {/* New Chat button */}
            <div className="px-3 pb-2">
              <button
                onClick={handleNewChat}
                className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl transition-all"
                style={{
                  background: isDark ? "rgba(59,130,246,0.1)" : "rgba(59,130,246,0.06)",
                  border: "1px solid rgba(59,130,246,0.2)",
                  color: "#3b82f6",
                }}
              >
                <Plus className="w-4 h-4" />
                <span style={{ fontFamily: "'Inter', sans-serif", fontSize: "13px", fontWeight: 600 }}>New Chat</span>
              </button>
            </div>"""
new_new_chat = """            {/* New Chat button */}
            <div className="px-3 pb-2">
              <button
                onClick={handleNewChat}
                className={`w-full flex items-center transition-all ${sidebarOpen ? 'gap-2.5 px-3 py-2.5 rounded-xl' : 'justify-center p-2.5 rounded-xl'}`}
                style={{
                  background: isDark ? "rgba(59,130,246,0.1)" : "rgba(59,130,246,0.06)",
                  border: "1px solid rgba(59,130,246,0.2)",
                  color: "#3b82f6",
                }}
              >
                <Plus className="w-4 h-4 flex-shrink-0" />
                {sidebarOpen && <span style={{ fontFamily: "'Inter', sans-serif", fontSize: "13px", fontWeight: 600 }}>New Chat</span>}
              </button>
            </div>"""
content = content.replace(old_new_chat, new_new_chat)


# 5. History section - hide if closed
old_history_start = """            {/* Chat history */}
            <div className="flex-1 overflow-y-auto px-2 py-1">"""
new_history_start = """            {/* Chat history */}
            <div className={`flex-1 overflow-y-auto py-1 ${sidebarOpen ? 'px-2' : 'px-0 opacity-0 pointer-events-none'}`}>"""
content = content.replace(old_history_start, new_history_start)

# 6. Settings footer
old_settings = """            {/* Sidebar footer */}
            <div
              className="px-3 py-3"
              style={{ borderTop: `1px solid ${borderColor}` }}
            >
              <button
                className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl transition-colors"
                style={{ color: textSecondary }}
                onMouseEnter={(e) => { e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.04)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
              >
                <Settings className="w-4 h-4" />
                <span style={{ fontSize: "13px", fontWeight: 500 }}>Settings</span>
              </button>
            </div>"""
new_settings = """            {/* Sidebar footer */}
            <div
              className="px-3 py-3"
              style={{ borderTop: `1px solid ${borderColor}` }}
            >
              <button
                className={`w-full flex items-center transition-colors ${sidebarOpen ? 'gap-2.5 px-3 py-2.5 rounded-xl' : 'justify-center p-2.5 rounded-xl'}`}
                style={{ color: textSecondary }}
                onMouseEnter={(e) => { e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.04)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
              >
                <Settings className="w-4 h-4 flex-shrink-0" />
                {sidebarOpen && <span style={{ fontSize: "13px", fontWeight: 500 }}>Settings</span>}
              </button>
            </div>"""
content = content.replace(old_settings, new_settings)

with open("/Users/ashwinagarkhed/sentinel_e/figma_ui/src/app/components/ChatPage.tsx", "w") as f:
    f.write(content)
print("Sidebar patch applied.")
