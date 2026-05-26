import re

with open("/Users/ashwinagarkhed/sentinel_e/figma_ui/src/app/components/ChatPage.tsx", "r") as f:
    content = f.read()

old_style = """                        style={message.role === "user" ? {
                          borderRadius: "20px 20px 5px 20px",
                          background: isDark ? "#1d2030" : "#1d1d1f",
                          color: "#ffffff",
                          padding: "12px 16px",
                        } : {
                          borderRadius: "20px 20px 20px 5px",
                          background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.03)",
                          color: textPrimary,
                          border: `1px solid ${msgMode ? msgMode.color + "25" : borderColor}`,
                          borderLeft: msgMode ? `3px solid ${msgMode.color}` : message.mode === "kill" ? "3px solid #ef4444" : `1px solid ${borderColor}`,
                          boxShadow: isDark ? "0 2px 12px rgba(0,0,0,0.2)" : "0 1px 8px rgba(0,0,0,0.04)",
                          overflow: "visible",
                        }}"""

new_style = """                        style={message.role === "user" ? {
                          borderRadius: "20px 20px 5px 20px",
                          background: isDark ? "rgb(32, 32, 36)" : "rgb(240, 240, 245)",
                          color: isDark ? "#f5f5f7" : "#1d1d1f",
                          padding: "12px 16px",
                          boxShadow: isDark ? "0 4px 12px rgba(0,0,0,0.3)" : "0 2px 8px rgba(0,0,0,0.03)",
                          border: isDark ? "1px solid rgba(255,255,255,0.05)" : "1px solid rgba(0,0,0,0.04)",
                        } : {
                          borderRadius: "20px 20px 20px 5px",
                          background: isDark ? "rgb(24, 24, 28)" : "#ffffff",
                          color: textPrimary,
                          border: `1px solid ${msgMode ? msgMode.color + "25" : borderColor}`,
                          borderLeft: msgMode ? `3px solid ${msgMode.color}` : message.mode === "kill" ? "3px solid #ef4444" : `1px solid ${borderColor}`,
                          boxShadow: isDark ? "0 4px 16px rgba(0,0,0,0.4)" : "0 4px 16px rgba(0,0,0,0.04)",
                          overflow: "visible",
                        }}"""

content = content.replace(old_style, new_style)

# Also fix the text content color logic inside user bubble which forces #ffffff
old_content_color = """                            style={{
                              fontFamily: "'Inter', sans-serif",
                              fontSize: "15px",
                              lineHeight: 1.6,
                              fontWeight: 400,
                              color: message.role === "user" ? "#ffffff" : textPrimary,
                            }}"""
new_content_color = """                            style={{
                              fontFamily: "'Inter', sans-serif",
                              fontSize: "15px",
                              lineHeight: 1.6,
                              fontWeight: 400,
                              color: message.role === "user" ? (isDark ? "#f5f5f7" : "#1d1d1f") : textPrimary,
                            }}"""
content = content.replace(old_content_color, new_content_color)

with open("/Users/ashwinagarkhed/sentinel_e/figma_ui/src/app/components/ChatPage.tsx", "w") as f:
    f.write(content)

print("Bubbles updated.")
