import re

with open("src/app/components/ChatPage.tsx", "r") as f:
    content = f.read()

# Replace msgMode access
content = content.replace(
    'const msgMode = message.mode ? proSubModes.find(m => m.id === message.mode) : null;',
    'const msgMode = message.mode && message.mode !== "standard" ? getModeConfig(message.mode) : null;'
)

# Replace the constant definition (remove it)
content = re.sub(
    r'const proSubModes = \[\s*\{\s*id: "debate".*?\}\s*\];',
    '',
    content,
    flags=re.DOTALL
)

with open("src/app/components/ChatPage.tsx", "w") as f:
    f.write(content)
