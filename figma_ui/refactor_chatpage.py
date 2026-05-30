import re

with open("src/app/components/ChatPage.tsx", "r") as f:
    content = f.read()

# Replace isPlusOpen with isDropdownOpen
content = content.replace("isPlusOpen", "isDropdownOpen")
content = content.replace("setIsPlusOpen", "setIsDropdownOpen")

# In handleSend payload generation (around line 592):
# Ensure mode is standard if !isProActive, and mode is selectedMode if isProActive
# We need to replace the payload construction logic
payload_regex = r"const\s+payload\s*=\s*{[\s\S]*?};"

def payload_replacement(match):
    original = match.group(0)
    # Rebuild the payload safely
    return """const payload = {
      message: trimmed,
      chat_id: currentChatId || undefined,
      session_id: "test-session",
      model: isProActive ? "orchestration-auto" : activeModel.id,
      mode: isProActive ? getModeConfig(selectedMode).id : "standard",
      sub_mode: undefined,
      metadata: {
        client_timestamp: new Date().toISOString(),
        client_version: "5.0.0",
        orchestration_requested: isProActive
      }
    };"""

content = re.sub(payload_regex, payload_replacement, content, count=1)

with open("src/app/components/ChatPage.tsx", "w") as f:
    f.write(content)
